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

# 2. 保留比例 (Keep Ratio)
# 意思是：保留 Top X%，Mask 掉剩余的 (100-X)%
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

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
    print("🚀 [Hypothesis-02 Inverse] Keep Top-K Doubtful, Mask Others...")
    
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

    # 3. 自动识别所有卷积层
    target_layers = []
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            target_layers.append(name)
    print(f"🎯 Target Layers: {target_layers}")

    # 4. 计算 Baseline
    print("\n📊 Calculating Baseline...")
    res_base = test_fairness_standalone(net, test_loader, DEVICE)
    base_hard = res_base['m_s']
    
    base_str = f"{'Baseline':<10} | {res_base['glb']:<8.2f} | {res_base['gap']:<8.2f} | {res_base['m_s']:<14.2f} | {res_base['m_ns']:<8.2f} | {res_base['f_s']:<8.2f} | {res_base['f_ns']:<8.2f} | {res_base['male']:<8.2f} | {res_base['female']:<8.2f}"

    # === 层级循环 ===
    for layer_name in target_layers:
        print(f"\n📂 Analysing Layer: [ {layer_name} ]")
        
        # 提取激活
        act_m, act_f = extract_activations_for_layer(net, probe_loader, layer_name, DEVICE)
        num_filters = act_m.shape[1]
        all_indices = set(range(num_filters)) # 所有神经元的索引集合
        
        # 预计算 FAAR 分数
        method_scores = {}
        for method in METHODS:
            if method != 'random':
                method_scores[method] = get_doubt_score(act_m, act_f, method=method)
        
        # === 比例循环 (注意：这里的 Ratio 是保留比例) ===
        for ratio in RATIOS:
            keep_k = int(num_filters * ratio)
            if keep_k < 1: keep_k = 1
            mask_k = num_filters - keep_k # 剪掉的数量
            
            print(f"\n✂️ Ratio: {ratio:.0%} (Keep {keep_k} / Pruned {mask_k})")
            
            header = f"{'Method':<10} | {'Glb Acc':<8} | {'Gap':<8} | {'Hard(M_S)':<14} | {'M_NS':<8} | {'F_S':<8} | {'F_NS':<8} | {'Male':<8} | {'Female':<8}"
            print("-" * 130)
            print(header)
            print("-" * 130)
            print(base_str) 

            # === 方法循环 ===
            for method in METHODS:
                n_repeats = RANDOM_REPEAT if method == 'random' else 1
                history = {k: [] for k in ['glb', 'gap', 'm_s', 'm_ns', 'f_s', 'f_ns', 'male', 'female']}
                
                for i in range(n_repeats):
                    net.load_state_dict(copy.deepcopy(original_state_dict))
                    
                    # --- 核心反转逻辑 ---
                    if method == 'random':
                        # 随机选择 keep_k 个保留
                        keep_indices_list = torch.randperm(num_filters)[:keep_k].tolist()
                    else:
                        scores = method_scores[method]
                        # 选出分数最高的 keep_k 个 (这些是我们认为有信息的)
                        keep_indices_tensor = select_neurons_by_temperature(scores, temperature=0.0, top_k=keep_k)
                        keep_indices_list = keep_indices_tensor.tolist()
                    
                    # 计算补集：要 Mask 掉的索引
                    mask_indices_list = list(all_indices - set(keep_indices_list))
                    mask_indices_tensor = torch.tensor(mask_indices_list).to(DEVICE)
                    
                    # 执行 Mask (注意：传入的是 mask_indices)
                    if len(mask_indices_list) > 0:
                        treatment_mask(net, layer_name, mask_indices_tensor)
                    
                    # Evaluate
                    res = test_fairness_standalone(net, test_loader, DEVICE)
                    for k in history.keys(): history[k].append(res[k])

                # 格式化输出
                mean_hard = np.mean(history['m_s'])
                diff = mean_hard - base_hard
                
                if method == 'random':
                    hard_str = f"{mean_hard:.2f} ({diff:+.2f}%)"
                else:
                    hard_str = f"{mean_hard:.2f} ({diff:+.2f}%)"

                def fmt(key):
                    val = np.mean(history[key])
                    return f"{val:.2f}"

                row_str = f"{method:<10} | "
                row_str += f"{fmt('glb'):<8} | "
                row_str += f"{fmt('gap'):<8} | "
                row_str += f"{hard_str:<14} | "
                row_str += f"{fmt('m_ns'):<8} | "
                row_str += f"{fmt('f_s'):<8} | "
                row_str += f"{fmt('f_ns'):<8} | "
                row_str += f"{fmt('male'):<8} | "
                row_str += f"{fmt('female'):<8}"
                
                print(row_str)

if __name__ == "__main__":
    main()