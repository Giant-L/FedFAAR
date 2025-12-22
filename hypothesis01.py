import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import copy
import numpy as np
from collections import OrderedDict

# --- 1. 环境设置与导入 ---
# 将当前目录添加到路径，以便导入 models 和 utils
sys.path.append(os.getcwd())

from models.Nets import SimpleCNN
# 假设 get_celeba_dataset_raw 在 utils.raw_celeba 中，如果没有请根据你的 utils 修改
from utils.raw_celeba import get_celeba_dataset_raw 
from utils.options import args_parser

# --- 导入 FAAR 核心模块 ---
from faar.locate_DoubtDegree import get_doubt_score
from faar.treatment_MorRorO import select_neurons_by_temperature, treatment_mask

# ================= 配置区域 =================
MODEL_PATH = './save/FedAvg_SimpleCNN_Biased20_celeba_Beta0.5/model_round_199.pth'
PROBE_PATH = './probe_balanced_2000.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 待测试的方法列表
METHODS = ['linear', 'log_odds', 'shape_rate', 'complex', 'jsd']
# 待测试的剪枝比例
RATIOS = [0.10, 0.20, 0.30, 0.50]
# ===========================================

# 复制一份 test_fairness 以便独立运行
def test_fairness_standalone(model, loader, device):
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    
    counts = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    corrects = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    
    total = 0
    correct = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            # 兼容 (img, label) 或 (img, label, sensitive)
            if len(batch) == 3:
                targets = batch[1].to(device).long()
                sensitives = batch[2].to(device).long()
            else:
                labels = batch[1].to(device)
                if labels.dim() > 1:
                    targets = labels[:, 0].long()
                    sensitives = labels[:, 1].long()
                else:
                    targets = labels.long()
                    sensitives = torch.zeros_like(targets)

            outputs = model(imgs)
            if isinstance(outputs, dict): outputs = outputs['output']
            
            loss = loss_func(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # 统计子组
            batch_correct = (predicted == targets)
            for s in [0, 1]:
                for t in [0, 1]:
                    mask = (sensitives == s) & (targets == t)
                    if mask.sum() > 0:
                        counts[s][t] += mask.sum().item()
                        corrects[s][t] += batch_correct[mask].sum().item()

    acc_global = 100.0 * correct / (total + 1e-6)
    avg_loss = test_loss / (total + 1e-6)

    # 计算各项指标
    res = {}
    res['acc'] = acc_global
    res['loss'] = avg_loss
    
    # 子组 Acc
    def get_acc(s, t): return 100.0 * corrects[s][t] / (counts[s][t] + 1e-6)
    
    res['acc_m_s'] = get_acc(1, 1)   # Male Smile (Hard)
    res['acc_m_ns'] = get_acc(1, 0)  # Male NoSmile (Easy)
    res['acc_f_s'] = get_acc(0, 1)   # Female Smile (Easy)
    res['acc_f_ns'] = get_acc(0, 0)  # Female NoSmile (Hard)
    
    # 总体 Acc
    total_m = counts[1][0] + counts[1][1]
    total_f = counts[0][0] + counts[0][1]
    res['acc_m'] = 100.0 * (corrects[1][0] + corrects[1][1]) / (total_m + 1e-6)
    res['acc_f'] = 100.0 * (corrects[0][0] + corrects[0][1]) / (total_f + 1e-6)
    
    res['gap'] = abs(res['acc_m'] - res['acc_f'])
    
    return res

def print_result_line(title, res):
    print(f"{title:<20} | Glb: {res['acc']:5.2f}% | Gap: {res['gap']:5.2f}% | "
          f"M_Smile: {res['acc_m_s']:5.2f}% (Hard) | M_NoSmile: {res['acc_m_ns']:5.2f}% | "
          f"F_NoSmile: {res['acc_f_ns']:5.2f}%")

# 提取指定层激活并按性别分组
def extract_activations_for_diagnosis(model, probe_loader, target_layer_name, device):
    model.eval()
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # 注册 Hook
    handle = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            handle = module.register_forward_hook(get_activation(name))
            break
    
    if handle is None:
        raise ValueError(f"Layer {target_layer_name} not found in model.")

    act_male_list = []
    act_female_list = []

    with torch.no_grad():
        for batch in probe_loader:
            imgs = batch[0].to(device)
            sensitives = batch[2].to(device) # Probe pt 包含 sensitive 在 index 2
            
            _ = model(imgs)
            
            # [Batch, Channel, H, W] -> [Batch, Channel] (Global Average Pooling)
            act = activation[target_layer_name].mean(dim=[2, 3])
            
            act_male_list.append(act[sensitives == 1])
            act_female_list.append(act[sensitives == 0])

    handle.remove()
    
    # 拼接
    act_male = torch.cat(act_male_list, dim=0)
    act_female = torch.cat(act_female_list, dim=0)
    
    return act_male, act_female

def main():
    print("🚀 [Hypothesis-01] Starting Experiment...")
    
    # 1. 模拟 Args 并加载测试集 (用于评估结果)
    args = args_parser()
    args.device = DEVICE
    args.bs = 100
    
    print("📂 Loading Standard CelebA Test Set (for Evaluation)...")
    _, dataset_test, _ = get_celeba_dataset_raw(args)
    test_loader = DataLoader(dataset_test, batch_size=100, shuffle=False, num_workers=4)
    
    # 2. 加载 Probe 数据集 (用于诊断)
    print(f"📂 Loading Balanced Probe from {PROBE_PATH} (for Diagnosis)...")
    probe_data = torch.load(PROBE_PATH)
    probe_dataset = TensorDataset(probe_data['x'], probe_data['y'], probe_data['s'])
    probe_loader = DataLoader(probe_dataset, batch_size=100, shuffle=False)
    
    # 3. 初始化模型并加载权重
    print(f"🔄 Loading Model from {MODEL_PATH}...")
    # 注意：SimpleCNN 初始化可能需要 num_classes
    net = SimpleCNN(num_classes=2).to(DEVICE)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        net.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 备份原始参数，用于每次重置
    original_state_dict = copy.deepcopy(net.state_dict())
    
    # 4. 寻找目标层 (通常是最后一个卷积层)
    target_layer_name = None
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer_name = name # 不断更新，最后保留的就是最后一个 Conv
    
    print(f"🎯 Target Layer for Surgery: {target_layer_name}")

    # ================= BASELINE =================
    print("\n📊 --- Baseline Performance (Before Surgery) ---")
    res_base = test_fairness_standalone(net, test_loader, DEVICE)
    print_result_line("Original Model", res_base)
    print("-" * 120)

    # ================= 诊断阶段 =================
    print(f"\n🔍 Diagnosing Neurons using Probe Data (Layer: {target_layer_name})...")
    act_m, act_f = extract_activations_for_diagnosis(net, probe_loader, target_layer_name, DEVICE)
    print(f"   Collected Activations: Male={act_m.shape}, Female={act_f.shape}")

    # ================= 实验循环 =================
    
    # 存储结果以便最后汇总
    summary = []

    for method in METHODS:
        print(f"\n🧪 Testing Method: [ {method.upper()} ]")
        
        # 1. 计算分数 (只算一次)
        scores = get_doubt_score(act_m, act_f, method=method)
        
        for ratio in RATIOS:
            # 重置模型
            net.load_state_dict(copy.deepcopy(original_state_dict))
            
            # 计算需要切除的数量
            num_filters = scores.shape[0]
            top_k = int(num_filters * ratio)
            if top_k < 1: top_k = 1
            
            # 2. 锁定神经元 (Hard Selection, Temperature=0)
            indices = select_neurons_by_temperature(scores, temperature=0.0, top_k=top_k)
            
            # 3. 执行手术 (Mask)
            treatment_mask(net, target_layer_name, indices)
            
            # 4. 评估
            res = test_fairness_standalone(net, test_loader, DEVICE)
            
            tag = f"{method}@{int(ratio*100)}%"
            print_result_line(tag, res)
            
            summary.append({
                'method': method,
                'ratio': ratio,
                'gap': res['gap'],
                'acc': res['acc'],
                'hard_acc': res['acc_m_s']
            })

    # ================= 最佳结果汇总 =================
    print("\n🏆 === Best Results Summary ===")
    # 按 Hard Acc (Male Smile) 排序
    best_hard = sorted(summary, key=lambda x: x['hard_acc'], reverse=True)[0]
    # 按 Gap 排序
    best_gap  = sorted(summary, key=lambda x: x['gap'])[0]
    
    print(f"🔥 Best Hard Acc improvement: {best_hard['method']} (Top {best_hard['ratio']:.0%}) -> Hard Acc: {best_hard['hard_acc']:.2f}% (Base: {res_base['acc_m_s']:.2f}%)")
    print(f"⚖️ Best Gap reduction      : {best_gap['method']} (Top {best_gap['ratio']:.0%}) -> Gap: {best_gap['gap']:.2f}% (Base: {res_base['gap']:.2f}%)")

if __name__ == "__main__":
    main()