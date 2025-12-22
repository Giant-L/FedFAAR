# locate策略 + reset手段

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# ==============================================================================
# 1. 神经元定位器 (Locators) - 五种方法集成
# ==============================================================================

def score_linear_mixing(act_male, act_female, alpha=0.5):
    rate_m = (act_male > 0).float().mean(dim=0)
    rate_f = (act_female > 0).float().mean(dim=0)
    diff_rate = torch.abs(rate_m - rate_f)
    mu_m = act_male.mean(dim=0)
    mu_f = act_female.mean(dim=0)
    diff_mu = torch.abs(mu_m - mu_f)
    return (1 - alpha) * diff_mu + alpha * diff_rate

def score_mixed_log_odds(act_male, act_female, eps=1e-6):
    rate_m = (act_male > 0).float().mean(dim=0)
    rate_f = (act_female > 0).float().mean(dim=0)
    rate_m = torch.clamp(rate_m, min=eps, max=1-eps)
    rate_f = torch.clamp(rate_f, min=eps, max=1-eps)
    logit_m = torch.log(rate_m / (1 - rate_m))
    logit_f = torch.log(rate_f / (1 - rate_f))
    return torch.abs(logit_m - logit_f)

def score_shape_rate(act_male, act_female, alpha=0.5):
    # 简化版 Wasserstein (假设对齐)
    if act_male.shape[0] == act_female.shape[0]:
        m_sorted, _ = torch.sort(act_male, dim=0)
        f_sorted, _ = torch.sort(act_female, dim=0)
        wass_dist = torch.mean(torch.abs(m_sorted - f_sorted), dim=0)
    else:
        qs = torch.linspace(0, 1, 100).to(act_male.device)
        m_q = torch.quantile(act_male, qs, dim=0)
        f_q = torch.quantile(act_female, qs, dim=0)
        wass_dist = torch.mean(torch.abs(m_q - f_q), dim=0)
    
    rate_m = (act_male > 0).float().mean(dim=0)
    rate_f = (act_female > 0).float().mean(dim=0)
    diff_rate = torch.abs(rate_m - rate_f)
    return (1 - alpha) * wass_dist + alpha * diff_rate

def score_complex_fisher(act_male, act_female):
    mu_m, var_m = act_male.mean(dim=0), act_male.var(dim=0)
    mu_f, var_f = act_female.mean(dim=0), act_female.var(dim=0)
    numerator = torch.pow(mu_m - mu_f, 2)
    denominator = var_m + var_f + 1e-8
    fisher = numerator / denominator
    rate_diff = torch.abs((act_male > 0).float().mean(dim=0) - (act_female > 0).float().mean(dim=0))
    return fisher * (1 + rate_diff)

def score_jsd(act_male, act_female, bins=20):
    # 简化的 JSD 实现 (Batch处理)
    # 为了速度，这里使用 Rate 差异近似，或者你可以保留你原来的循环实现
    # 这里提供一个高效的 Rate-based 近似，因为循环太慢会卡死 Server
    p0_m = (act_male <= 0).float().mean(dim=0)
    p0_f = (act_female <= 0).float().mean(dim=0)
    return torch.abs(p0_m - p0_f) 

def get_doubt_score(act_male, act_female, method='log_odds'):
    if method == 'random':
        return torch.rand(act_male.shape[1]).to(act_male.device)
    elif method == 'linear':
        return score_linear_mixing(act_male, act_female)
    elif method == 'log_odds':
        return score_mixed_log_odds(act_male, act_female)
    elif method == 'shape_rate':
        return score_shape_rate(act_male, act_female)
    elif method == 'complex':
        return score_complex_fisher(act_male, act_female)
    elif method == 'jsd':
        return score_jsd(act_male, act_female)
    else:
        raise ValueError(f"Unknown method: {method}")

# ==============================================================================
# 2. 诊断主程序 (Server Side Diagnosis)
# ==============================================================================

def extract_layer_scores(net, probe_loader, device, method='log_odds'):
    """
    提取所有卷积层的激活值，并计算怀疑度分数。
    返回: {layer_name: score_tensor}
    """
    net.eval()
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hooks = []
    target_layers = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d):
            h = layer.register_forward_hook(get_activation(name))
            hooks.append(h)
            target_layers.append(name)

    # 收集激活值
    act_male = {name: [] for name in target_layers}
    act_female = {name: [] for name in target_layers}
    
    with torch.no_grad():
        for imgs, labels in probe_loader:
            imgs = imgs.to(device)
            # 处理 label 格式
            if labels.dim() > 1:
                sensitives = labels[:, 1].to(device).long()
            else:
                sensitives = torch.zeros(labels.size(0)).long().to(device) # fallback

            _ = net(imgs)
            
            for name in target_layers:
                # Global Average Pooling [B, C, H, W] -> [B, C]
                act = activation[name].mean(dim=[2, 3])
                act_male[name].append(act[sensitives == 1])
                act_female[name].append(act[sensitives == 0])

    for h in hooks: h.remove()

    # 计算分数
    layer_scores = {}
    for name in target_layers:
        am = torch.cat(act_male[name], dim=0)
        af = torch.cat(act_female[name], dim=0)
        
        # 避免空数据
        if len(am) == 0 or len(af) == 0:
            layer_scores[name] = torch.zeros(am.shape[1]).to(device)
        else:
            layer_scores[name] = get_doubt_score(am, af, method=method)
            
    return layer_scores

# ==============================================================================
# 3. 神经元选择 (Temperature Scaling)
# ==============================================================================

def select_neurons_by_temperature(scores, temperature=1.0, top_k=None):
    """
    基于温度采样。
    tau -> 0: Hard Top-K
    tau -> inf: Random
    """
    scores = scores.float()
    
    # 极低温度：确定性选择
    if temperature < 1e-4:
        _, indices = torch.topk(scores, top_k)
        return indices
    
    # 温度缩放 Softmax
    logits = scores / temperature
    probs = torch.softmax(logits, dim=0)
    
    # 无放回采样
    # indices = torch.multinomial(probs, top_k, replacement=False)
    
    # 💡 优化：为了保证每个客户端得到的 "随机性" 是独立的，
    # 我们不在这里 fix seed，而是依赖外部调用时的随机状态。
    try:
        indices = torch.multinomial(probs, top_k, replacement=False)
    except RuntimeError:
        # Fallback if probs contain NaNs or sum to 0
        _, indices = torch.topk(scores, top_k)
        
    return indices

# ==============================================================================
# 4. 执行 Soft-Reset (Client Side Operation)
# ==============================================================================

def apply_faar_soft_reset(model, layer_indices_map, scale_factor=1.0):
    """
    model: 客户端本地模型
    layer_indices_map: {layer_name: [idx1, idx2, ...]} 需要重置的索引
    """
    reset_count = 0
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in layer_indices_map and isinstance(module, nn.Conv2d):
                indices = layer_indices_map[name]
                if len(indices) == 0: continue
                
                # 计算统计量 (Soft Reset Logic)
                current_mean = module.weight.data.mean().item()
                current_std = module.weight.data.std().item()
                
                for idx in indices:
                    # ori_normal 初始化: N(mean, std)
                    new_kernel = torch.randn_like(module.weight[idx]) * current_std * scale_factor + current_mean
                    module.weight.data[idx] = new_kernel
                    reset_count += 1
                    
    return reset_count