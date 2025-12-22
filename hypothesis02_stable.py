import torch
import copy
import sys
import os
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# --- 环境设置 ---
sys.path.append(os.getcwd())
from models.Nets import SimpleCNN
from utils.raw_celeba import get_celeba_dataset_raw 
from utils.options import args_parser
from faar.locate_DoubtDegree import get_doubt_score
from faar.treatment_MorRorO import select_neurons_by_temperature, treatment_mask

# ================= 配置区域 =================
MODEL_PATH = './save/FedAvg_SimpleCNN_Biased20_celeba_Beta0.5/model_round_199.pth'
PROBE_PATH = './probe_balanced_2000.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 实验方法
METHODS = ['random', 'linear', 'log_odds', 'shape_rate', 'complex', 'jsd']

# 2. 剪枝比例
RATIOS = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

# 3. Random 方法重复次数
RANDOM_REPEAT = 3

# ================= 辅助函数 =================

def test_fairness_standalone(model, loader, device):
    """ 评估模型并返回所有详细指标 """
    model.eval()
    counts = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    corrects = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    total = 0; correct = 0
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            # 兼容性标签解析
            if len(batch) == 3:
                targets = batch[1].to(device).long(); sensitives = batch[2].to(device).long()
            else:
                labels = batch[1].to(device)
                if labels.dim() > 1: targets = labels[:,0].long(); sensitives = labels[:,1].long()
                else: targets = labels.long(); sensitives = torch.zeros_like(targets)

            outputs = model(imgs)
            if isinstance(outputs, dict): outputs = outputs['output']
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0); correct += (predicted == targets).sum().item()
            
            batch_correct = (predicted == targets)
            for s in [0, 1]:
                for t in [0, 1]:
                    mask = (sensitives == s) & (targets == t)
                    if mask.sum() > 0:
                        counts[s][t] += mask.sum().item()
                        corrects[s][t] += batch_correct[mask].sum().item()

    # 计算各项指标
    res = {}
    res['glb'] = 100.0 * correct / (total + 1e-6)
    
    def get_acc(s, t): return 100.0 * corrects[s][t] / (counts[s][t] + 1e-6)
    res['m_s'] = get_acc(1, 1) # Male Smile (Hard)
    res['m_ns'] = get_acc(1, 0) # Male NoSmile
    res['f_s'] = get_acc(0, 1) # Female Smile
    res['f_ns'] = get_acc(0, 0) # Female NoSmile (Hard)
    
    # 补充 Male / Female 整体 Acc
    res['male'] = 100.0 * (corrects[1][0] + corrects[1][1]) / (counts[1][0] + counts[1][1] + 1e-6)
    res['female'] = 100.0 * (corrects[0][0] + corrects[0][1]) / (counts[0][0] + counts[0][1] + 1e-6)
    
    # Gap 计算
    res['gap'] = abs(res['male'] - res['female'])
    
    return res

def extract_activations_for_layer(model, probe_loader, layer_name, device):
    model.eval()
    activation = {}
    def get_activation(name):
        def hook(model, input, output): activation[name] = output.detach()
        return hook

    layer_module = dict(model.named_modules())[layer_name]
    handle = layer_module.register_forward_hook(get_activation(layer_name))

    act_male_list, act_female_list = [], []
    with torch.no_grad():
        for batch in probe_loader:
            imgs, sensitives = batch[0].to(device), batch[2].to(device)
            _ = model(imgs)
            act = activation[layer_name].mean(dim=[2, 3])
            act_male_list.append(act[sensitives == 1])
            act_female_list.append(act[sensitives == 0])
    handle.remove()
    return torch.cat(act_male_list), torch.cat(act_female_list)

def main():
    print("🚀 [Hypothesis-02 Final] Full Report with Baselines...")
    
    # 1. 加载数据
    args = args_parser(); args.device = DEVICE; args.bs = 100
    _, dataset_test, _ = get_celeba_dataset_raw(args)
    test_loader = DataLoader(dataset_test, batch_size=200, shuffle=False, num_workers=4)
    
    probe_data = torch.load(PROBE_PATH)
    probe_dataset = TensorDataset(probe_data['x'], probe_data['y'], probe_data['s'])
    probe_loader = DataLoader(probe_dataset, batch_size=100, shuffle=False)
    
    # 2. 加载模型
    net = SimpleCNN(num_classes=2).to(DEVICE)
    try:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ Error loading model: {e}"); return
    original_state_dict = copy.deepcopy(net.state_dict())

    # 3. 锁定层
    target_layers = ['conv3', 'conv4']
    print(f"🎯 Target Layers: {target_layers}")

    # 4. 计算 Baseline (一次性计算，后续重复打印)
    print("\n📊 Calculating Baseline...")
    res_base = test_fairness_standalone(net, test_loader, DEVICE)
    base_hard = res_base['m_s']
    
    # 构造 Baseline 的打印字符串 (formatted string)
    base_str = f"{'Baseline':<10} | {res_base['glb']:<8.2f} | {res_base['gap']:<8.2f} | {res_base['m_s']:<14.2f} | {res_base['m_ns']:<8.2f} | {res_base['f_s']:<8.2f} | {res_base['f_ns']:<8.2f} | {res_base['male']:<8.2f} | {res_base['female']:<8.2f}"

    # === 层级循环 ===
    for layer_name in target_layers:
        print(f"\n📂 Analysing Layer: [ {layer_name} ]")
        
        # 提取激活
        act_m, act_f = extract_activations_for_layer(net, probe_loader, layer_name, DEVICE)
        num_filters = act_m.shape[1]
        
        # 预计算 FAAR 分数
        method_scores = {}
        for method in METHODS:
            if method != 'random':
                method_scores[method] = get_doubt_score(act_m, act_f, method=method)
        
        # === 比例循环 ===
        for ratio in RATIOS:
            print(f"\n✂️ Ratio: {ratio:.0%} (Pruned {int(num_filters * ratio)}/{num_filters})")
            
            # 表头 (增加了 Male / Female 列)
            header = f"{'Method':<10} | {'Glb Acc':<8} | {'Gap':<8} | {'Hard(M_S)':<14} | {'M_NS':<8} | {'F_S':<8} | {'F_NS':<8} | {'Male':<8} | {'Female':<8}"
            print("-" * 130)
            print(header)
            print("-" * 130)
            
            # 🟢 每一组之前都先打印 Baseline 🟢
            print(base_str) 

            # === 方法循环 ===
            for method in METHODS:
                n_repeats = RANDOM_REPEAT if method == 'random' else 1
                history = {k: [] for k in ['glb', 'gap', 'm_s', 'm_ns', 'f_s', 'f_ns', 'male', 'female']}
                
                for i in range(n_repeats):
                    net.load_state_dict(copy.deepcopy(original_state_dict))
                    top_k = int(num_filters * ratio)
                    if top_k < 1: top_k = 1
                    
                    if method == 'random':
                        indices = torch.randperm(num_filters)[:top_k].to(DEVICE)
                    else:
                        scores = method_scores[method]
                        indices = select_neurons_by_temperature(scores, temperature=0.0, top_k=top_k)
                    
                    treatment_mask(net, layer_name, indices)
                    res = test_fairness_standalone(net, test_loader, DEVICE)
                    
                    for k in history.keys(): history[k].append(res[k])

                # 格式化输出
                mean_hard = np.mean(history['m_s'])
                diff = mean_hard - base_hard
                
                # Hard Acc 显示变化幅度
                if method == 'random':
                    hard_str = f"{mean_hard:.2f} ({diff:+.2f}%)" # 简化 Random 显示，太长会乱
                else:
                    hard_str = f"{mean_hard:.2f} ({diff:+.2f}%)"

                def fmt(key):
                    val = np.mean(history[key])
                    return f"{val:.2f}" # 保持简洁，去掉 % 号，表头已有

                row_str = f"{method:<10} | "
                row_str += f"{fmt('glb'):<8} | "
                row_str += f"{fmt('gap'):<8} | "
                row_str += f"{hard_str:<14} | " # 重点列
                row_str += f"{fmt('m_ns'):<8} | "
                row_str += f"{fmt('f_s'):<8} | "
                row_str += f"{fmt('f_ns'):<8} | "
                row_str += f"{fmt('male'):<8} | "
                row_str += f"{fmt('female'):<8}"
                
                print(row_str)

if __name__ == "__main__":
    main()