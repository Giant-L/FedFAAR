import torch
import copy
import sys
import os
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# --- 1. 环境设置 ---
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

# 实验变量
METHODS = ['random', 'linear', 'log_odds', 'shape_rate', 'complex', 'jsd']
RATIOS = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

# ================= 辅助函数 =================

def test_fairness_standalone(model, loader, device):
    """ 快速评估函数 """
    model.eval()
    counts = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    corrects = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    total = 0; correct = 0
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            
            # 🟢 兼容性标签解析逻辑 (Original版本包含这个)
            if len(batch) == 3:
                # 情况 A: 数据集返回 (img, target, sensitive)
                targets = batch[1].to(device).long()
                sensitives = batch[2].to(device).long()
            else:
                # 情况 B: 数据集返回 (img, labels)
                labels = batch[1].to(device)
                if labels.dim() > 1 and labels.shape[1] >= 2:
                    # 如果 label 是 [Batch, 2] -> 切分
                    targets = labels[:, 0].long()
                    sensitives = labels[:, 1].long()
                else:
                    # 如果 label 是 [Batch] -> 只有 target
                    targets = labels.long()
                    sensitives = torch.zeros_like(targets)

            outputs = model(imgs)
            if isinstance(outputs, dict): outputs = outputs['output']
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            batch_correct = (predicted == targets)
            for s in [0, 1]:
                for t in [0, 1]:
                    mask = (sensitives == s) & (targets == t)
                    if mask.sum() > 0:
                        counts[s][t] += mask.sum().item()
                        corrects[s][t] += batch_correct[mask].sum().item()

    acc_global = 100.0 * correct / (total + 1e-6)
    def get_acc(s, t): return 100.0 * corrects[s][t] / (counts[s][t] + 1e-6)
    
    res = {
        'acc': acc_global,
        'acc_m_s': get_acc(1, 1), # Hard: Male Smile
        'acc_m_ns': get_acc(1, 0),
        'acc_f_ns': get_acc(0, 0),
        'acc_f_s': get_acc(0, 1)
    }
    acc_m = (corrects[1][0] + corrects[1][1]) / (counts[1][0] + counts[1][1] + 1e-6) * 100
    acc_f = (corrects[0][0] + corrects[0][1]) / (counts[0][0] + counts[0][1] + 1e-6) * 100
    res['gap'] = abs(acc_m - acc_f)
    return res

def extract_activations_for_layer(model, probe_loader, layer_name, device):
    """ 提取指定层的激活值 """
    model.eval()
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
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
    print("🚀 [Hypothesis-02] Starting Full Spectrum Analysis...")
    print(f"   Config: {len(METHODS)} Methods x {len(RATIOS)} Ratios x All Conv Layers")
    
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
    
    # 3. 自动识别卷积层
    conv_layers = []
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append(name)
    print(f"🔍 Found Layers: {conv_layers}")

    # 4. Baseline
    print("\n📊 --- Original Baseline ---")
    res_base = test_fairness_standalone(net, test_loader, DEVICE)
    print(f"Base | Glb: {res_base['acc']:6.2f}% | Gap: {res_base['gap']:6.2f}% | Hard(M_Smile): {res_base['acc_m_s']:6.2f}%")
    print("=" * 120)

    # 结果容器
    global_results = []

    # === 大循环 ===
    for layer_name in conv_layers:
        print(f"\n📂 Analysing Layer: [ {layer_name} ]")
        
        # 🚨🚨🚨 【必须加这一行】 🚨🚨🚨
        # 确保诊断前，模型是满血状态！
        net.load_state_dict(copy.deepcopy(original_state_dict))

        # 提取激活 (该层只做一次)
        act_m, act_f = extract_activations_for_layer(net, probe_loader, layer_name, DEVICE)
        num_filters = act_m.shape[1]
        
        # 预计算分数 (注：这里是一次性计算完该层所有方法的分数)
        method_scores = {}
        for method in METHODS:
            if method != 'random':
                method_scores[method] = get_doubt_score(act_m, act_f, method=method)
        
        print(f"{'Method':<12} | {'Ratio':<6} | {'Pruned':<6} | {'Glb Acc':<8} | {'Gap':<8} | {'Hard Acc':<8} | {'Diff (vs Base)':<15}")
        print("-" * 100)

        for method in METHODS:
            for ratio in RATIOS:
                # 重置模型
                net.load_state_dict(copy.deepcopy(original_state_dict))
                
                top_k = int(num_filters * ratio)
                if top_k < 1: top_k = 1
                
                # Mask
                if method == 'random':
                    indices = torch.randperm(num_filters)[:top_k].to(DEVICE)
                else:
                    scores = method_scores[method]
                    indices = select_neurons_by_temperature(scores, temperature=0.0, top_k=top_k)
                
                treatment_mask(net, layer_name, indices)
                
                # Evaluate
                res = test_fairness_standalone(net, test_loader, DEVICE)
                
                imp = res['acc_m_s'] - res_base['acc_m_s']
                print(f"{method:<12} | {ratio:<6.0%} | {top_k:<6} | {res['acc']:<8.2f} | {res['gap']:<8.2f} | {res['acc_m_s']:<8.2f} | {imp:+6.2f}%")
                
                global_results.append({
                    'layer': layer_name, 'method': method, 'ratio': ratio,
                    'hard': res['acc_m_s'], 'gap': res['gap'], 'glb': res['acc'], 'imp': imp
                })

    # ================= 5. 汇总报告 =================
    print("\n🏆 ================== FINAL SUMMARY ==================")
    
    for layer_name in conv_layers:
        layer_res = [r for r in global_results if r['layer'] == layer_name]
        best_cfg = sorted(layer_res, key=lambda x: x['hard'], reverse=True)[0]
        
        print(f"\n📍 Layer: {layer_name}")
        print(f"   🔥 Best Config: {best_cfg['method']} @ {best_cfg['ratio']:.0%}")
        print(f"   📈 Hard Acc   : {best_cfg['hard']:.2f}% (Base: {res_base['acc_m_s']:.2f}%) -> Imp: {best_cfg['imp']:+.2f}%")
        print(f"   🌍 Global Acc : {best_cfg['glb']:.2f}%")

    overall_best = sorted(global_results, key=lambda x: x['hard'], reverse=True)[0]
    print(f"\n👑 OVERALL WINNER: {overall_best['layer']} - {overall_best['method']} @ {overall_best['ratio']:.0%}")
    print(f"   Hard Acc: {overall_best['hard']:.2f}% (+{overall_best['imp']:.2f}%)")

if __name__ == "__main__":
    main()