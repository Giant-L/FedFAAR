import torch
import sys
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 环境设置
sys.path.append(os.getcwd())
from models.Nets import SimpleCNN
from faar.locate_DoubtDegree import get_doubt_score
from faar.treatment_MorRorO import select_neurons_by_temperature

# ================= 配置 =================
MODEL_PATH = './save/FedAvg_SimpleCNN_Biased20_celeba_Beta0.5/model_round_199.pth'
PROBE_PATH = './probe_balanced_2000.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_LAYER = 'conv3' 
RATIO = 0.30 

def extract_activations_debug(model, probe_loader, layer_name, device):
    model.eval()
    activation = {}
    def hook(m, i, o): activation['act'] = o.detach()
    layer = dict(model.named_modules())[layer_name]
    h = layer.register_forward_hook(hook)
    
    act_m, act_f = [], []
    
    # 🔍 DEBUG: 打印前几个 batch 的 label 分布，确保没有乱序
    print(f"   [Data Audit] Checking Probe Loader Order...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(probe_loader):
            imgs = batch[0].to(device)
            # 兼容 (img, labels) 或 (img, target, sensitive)
            if len(batch) == 3: 
                sensitives = batch[2].to(device)
            else:
                labels = batch[1].to(device)
                sensitives = labels[:, 1] if labels.dim() > 1 else torch.zeros_like(labels)

            if batch_idx == 0:
                print(f"   -> Batch 0 Sensitives (First 10): {sensitives[:10].tolist()}")
                
            _ = model(imgs)
            act = activation['act'].mean(dim=[2, 3])
            act_m.append(act[sensitives == 1])
            act_f.append(act[sensitives == 0])
            
    h.remove()
    return torch.cat(act_m), torch.cat(act_f)

def main():
    print("🐞 [DEBUG] Starting Audit for Hypothesis-02...")
    
    # 1. 加载数据 (严格按照 hypothesis02 的方式)
    probe_data = torch.load(PROBE_PATH)
    probe_dataset = TensorDataset(probe_data['x'], probe_data['y'], probe_data['s'])
    
    # ⚠️ 关键点：Shuffle 必须为 False 才能保证确定性
    probe_loader = DataLoader(probe_dataset, batch_size=100, shuffle=False)
    
    # 2. 加载模型
    net = SimpleCNN(num_classes=2).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 3. 提取激活值
    print(f"\n1️⃣  Extracting Activations from {TARGET_LAYER}...")
    act_m, act_f = extract_activations_debug(net, probe_loader, TARGET_LAYER, DEVICE)
    
    print(f"   -> Male Act Shape: {act_m.shape}")
    print(f"   -> Female Act Shape: {act_f.shape}")
    print(f"   -> Male Mean (Check Consistency): {act_m.mean().item():.6f}")
    print(f"   -> Female Mean (Check Consistency): {act_f.mean().item():.6f}")
    
    if act_m.mean().item() == act_f.mean().item():
        print("   ❌ WARNING: Male and Female means are identical! Data splitting failed.")
        return

    # 4. 计算分数 & 索引
    print(f"\n2️⃣  Calculating Scores & Indices (Ratio={RATIO})...")
    num_filters = act_m.shape[1]
    top_k = int(num_filters * RATIO)
    
    # Method A: Log Odds
    scores_log = get_doubt_score(act_m, act_f, method='log_odds')
    indices_log = select_neurons_by_temperature(scores_log, temperature=0.0, top_k=top_k)
    val_log, _ = torch.topk(scores_log, top_k)
    
    # Method B: Linear
    scores_lin = get_doubt_score(act_m, act_f, method='linear')
    indices_lin = select_neurons_by_temperature(scores_lin, temperature=0.0, top_k=top_k)
    
    # Method C: Random (模拟一次)
    torch.manual_seed(42)
    indices_rnd = torch.randperm(num_filters)[:top_k].to(DEVICE)

    # 5. 输出对比
    print("-" * 60)
    print(f"Target Filters to Prune (Top {top_k}):")
    print(f"   [Log_Odds]: {sorted(indices_log.tolist())}")
    print(f"   [Linear]  : {sorted(indices_lin.tolist())}")
    print(f"   [Random]  : {sorted(indices_rnd.tolist())}")
    
    print("-" * 60)
    print("Score Values (Log_Odds Top 5):", val_log[:5].tolist())
    
    # 6. 核心判断
    intersection = len(set(indices_log.tolist()) & set(indices_rnd.tolist()))
    print(f"\nOverlap between Log_Odds and Random: {intersection} / {top_k}")
    
    if intersection > top_k * 0.5:
        print("⚠️ Warning: Your method looks like Random Sampling!")
    else:
        print("✅ Check: Your method is selecting distinct neurons.")

if __name__ == "__main__":
    main()