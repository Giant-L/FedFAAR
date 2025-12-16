import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
import copy

# =============================================================================
# 1. 诊断模块：提取分布 & 计算 Bias
# =============================================================================

def extract_activation_distributions(model, probe_loader, sensitive_attr_idx=1, device='cuda'):
    """
    运行探针数据，提取每一层、每一个卷积核对 Male/Female 的激活分布。
    """
    model.eval()
    activations = {}
    
    # 自动寻找卷积层
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layers.append((name, module))

    # Hook 函数：截获中间输出
    def get_hook(name):
        def hook(model, input, output):
            # Global Average Pooling: [B, C, H, W] -> [B, C]
            avg_act = output.mean(dim=[2, 3]).detach().cpu()
            if name not in activations:
                activations[name] = {'raw': []}
            activations[name]['raw'].append(avg_act)
        return hook

    # 注册 Hooks
    hooks = [m.register_forward_hook(get_hook(n)) for n, m in target_layers]

    # 运行探针
    sensitive_labels = []
    with torch.no_grad():
        for imgs, labels in probe_loader:
            imgs = imgs.to(device)
            # 兼容 CelebA 格式
            if labels.dim() > 1:
                s_targets = labels[:, sensitive_attr_idx].cpu()
            else:
                s_targets = torch.zeros_like(labels).cpu()
            sensitive_labels.append(s_targets)
            _ = model(imgs)

    # 移除 Hooks
    for h in hooks: h.remove()
    
    # 整理数据
    all_sensitive = torch.cat(sensitive_labels)
    mask_male = (all_sensitive == 1)
    mask_female = (all_sensitive == 0)
    
    results = {}
    
    for layer_name, data in activations.items():
        full_act = torch.cat(data['raw'], dim=0)
        
        # 安全检查：防止某类样本为0
        if mask_male.sum() == 0 or mask_female.sum() == 0:
            continue
            
        act_male = full_act[mask_male].numpy()
        act_female = full_act[mask_female].numpy()
        
        num_channels = full_act.shape[1]
        results[layer_name] = []
        for c in range(num_channels):
            # 存储每个 Channel 的分布数据
            results[layer_name].append({
                'male': act_male[:, c],
                'female': act_female[:, c]
            })
            
    return results


def calculate_reset_probs_softmax(dists, tau=0.1):
    """
    【你的创新点】Soft-Hard Guided Reset
    输入：激活分布差异
    输出：每一层每个卷积核的重置概率 P = Softmax(Bias / tau)
    """
    layer_probs = {}
    
    for layer_name, channels in dists.items():
        scores = []
        # 计算每个 Channel 的 Wasserstein 距离
        for c_data in channels:
            m_dist = c_data['male']
            f_dist = c_data['female']
            if len(m_dist) < 2 or len(f_dist) < 2:
                score = 0.0
            else:
                score = wasserstein_distance(m_dist, f_dist)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Softmax 计算
        # 1. 归一化 Score (防止数值不稳定)
        if scores.max() > 0:
            scores_norm = scores / (scores.max() + 1e-6)
        else:
            scores_norm = scores
            
        # 2. 计算概率：Bias 越大，P 越大
        logits = torch.tensor(scores_norm / tau)
        probs = F.softmax(logits, dim=0).numpy()
        
        layer_probs[layer_name] = {
            'scores': scores,
            'probs': probs
        }
        
    return layer_probs

# =============================================================================
# 2. 干预模块：层级衰减 & 重置
# =============================================================================

def get_layer_decay_ratio(layer_name, base_ratio):
    """
    【复刻 FedPhoenix】层级衰减机制 (Layer-wise Decay)
    浅层 (Layer1) 重置力度大，深层 (Layer4) 保护起来。
    """
    # 简单的层深判断逻辑 (适配 ResNet18)
    if 'conv1' in layer_name or 'layer1' in layer_name:
        factor = 1.0  # 100% 力度
    elif 'layer2' in layer_name:
        factor = 0.8  # 80% 力度
    elif 'layer3' in layer_name:
        factor = 0.5  # 50% 力度
    elif 'layer4' in layer_name:
        factor = 0.1  # 10% 力度 (深层几乎不重置)
    else:
        factor = 0.1
        
    real_ratio = base_ratio * factor
    return real_ratio

def apply_faar_reset(model, dists, args, logger=None):
    """
    主入口函数：执行 FAAR 干预
    参数：
      model: 全局模型
      dists: 提取出的分布数据
      args: 包含 args.reset (基础比例) 和 args.faar_tau (温度)
    """
    # 1. 计算 Softmax 概率
    layer_probs = calculate_reset_probs_softmax(dists, tau=args.faar_tau)
    
    total_reset = 0
    stats = {} # 用于保存统计信息
    
    for name, module in model.named_modules():
        if name in layer_probs and isinstance(module, nn.Conv2d):
            probs = layer_probs[name]['probs']
            scores = layer_probs[name]['scores']
            num_filters = len(probs)
            
            # 2. 【复刻 FedPhoenix】获取当前层的衰减比例
            current_ratio = get_layer_decay_ratio(name, args.reset)
            
            # 计算需要重置的数量 K
            k = int(num_filters * current_ratio)
            
            if k > 0:
                # 3. 【Soft 采样】根据 Bias 概率选择 K 个倒霉蛋
                reset_indices = np.random.choice(
                    np.arange(num_filters), 
                    size=k, 
                    replace=False, 
                    p=probs
                )
                
                # 4. 执行 Kaiming 初始化重置
                with torch.no_grad():
                    for idx in reset_indices:
                        nn.init.kaiming_uniform_(module.weight[idx].unsqueeze(0), a=np.sqrt(5))
                        if module.bias is not None:
                            module.bias[idx].fill_(0)
                
                total_reset += k
                stats[name] = {
                    'count': k,
                    'max_bias': float(scores.max()),
                    'indices': reset_indices.tolist()
                }
            else:
                stats[name] = {'count': 0}
                
    print(f"   [FAAR] Reset Completed. Total Filters Reset: {total_reset}")
    return stats