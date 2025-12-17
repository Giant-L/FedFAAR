# 文件: Algorithm/FAAR_Phoenix_util.py

import torch
import torch.nn as nn
import numpy as np

def apply_faar_soft_reset(model, reset_map, blend_alpha=0.8):
    """
    FAAR Soft Reset (治疗函数)
    
    创新点: 
    不直接把神经元砍掉(置0)，而是将其重置为“统计中性状态”。
    这类似于 FedPhoenix 中的重置，但我们是有针对性地重置那些 biased filters。
    
    Args:
        model: 全局模型
        reset_map: 诊断出的坏神经元 {layer_name: [index_list]}
        blend_alpha: 混合比例。
                     1.0 = 完全重置为随机噪声/均值 (消除记忆)
                     0.0 = 不变
                     0.5 = 一半记忆一半噪声 (软着陆)
    Returns:
        total_reset_count: 重置的神经元总数
    """
    total_reset_count = 0
    
    # 获取模型的 state_dict 以便操作
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in reset_map and isinstance(module, (nn.Conv2d, nn.Linear)):
                indices_to_reset = reset_map[name]
                if len(indices_to_reset) == 0:
                    continue
                
                # 获取权重张量
                weight = module.weight # [Out_Channels, In_Channels, K, K]
                
                # --- 核心操作：构造“中性”权重 ---
                # 策略 A: 重置为该层的平均值 (Mean Imputation) - 最温和
                # layer_mean = weight.mean(dim=0, keepdim=True)
                
                # 策略 B (Phoenix Style): 重置为高斯噪声，但保持该层的方差尺度 (Statistical Reset)
                # 这样不会破坏信号传输的能量，但消除了具体的偏见模式
                std = weight.std().item()
                mean = weight.mean().item()
                
                for idx in indices_to_reset:
                    # 生成统计匹配的噪声
                    noise = torch.randn_like(weight[idx]) * std + mean
                    
                    # 执行软重置 (Soft Reset)
                    # New_Weight = (1 - alpha) * Old_Weight + alpha * Noise
                    weight[idx] = (1 - blend_alpha) * weight[idx] + blend_alpha * noise
                    
                    # 如果有 Bias，通常也建议重置 Bias
                    if module.bias is not None:
                         module.bias[idx] = (1 - blend_alpha) * module.bias[idx] # 简单衰减 Bias
                
                total_reset_count += len(indices_to_reset)
                
    return total_reset_count