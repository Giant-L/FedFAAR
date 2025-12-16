import torch
import torch.nn as nn
import numpy as np
import scipy.stats
import copy

def get_target_layers(model):
    """获取所有卷积层和全连接层"""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers

def _get_wasserstein_distance(dist1, dist2):
    """
    计算两个一维分布的 Wasserstein 距离 (Earth Mover's Distance)
    """
    if len(dist1) == 0 or len(dist2) == 0:
        return 0.0
    return scipy.stats.wasserstein_distance(dist1, dist2)

def extract_activation_distributions(model, data_loader, sensitive_attr_idx, device, max_samples=200):
    """
    提取模型在给定数据集上，针对敏感属性（如男/女）的神经元激活分布。
    """
    model.eval()
    activations = {} 
    handles = []

    # 1. Register Hooks
    def get_hook(name):
        def hook(model, input, output):
            # output shape: [Batch, Neurons, H, W] for Conv, [Batch, Neurons] for Linear
            if output.dim() == 4:
                # Global Average Pooling 
                act = output.mean(dim=[2, 3]) 
            else:
                act = output
            
            if name not in activations:
                activations[name] = act.detach().cpu()
            else:
                activations[name] = torch.cat([activations[name], act.detach().cpu()], dim=0)
        return hook

    target_layers = get_target_layers(model)
    for name, layer in target_layers:
        handles.append(layer.register_forward_hook(get_hook(name)))

    # 2. Forward Pass
    collected_samples = 0
    sensitive_labels = [] 
    
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(data_loader):
            if collected_samples >= max_samples: break
            
            imgs = imgs.to(device)
            model(imgs)
            
            # 处理标签，提取敏感属性
            # 假设 labels 已经是处理好的 Tensor，或者是 tuple
            if isinstance(labels, list):
                sens = labels  # 简化处理
            elif isinstance(labels, torch.Tensor):
                # 如果是二维 [Batch, 2]，取第二列
                if labels.dim() > 1 and labels.size(1) > sensitive_attr_idx:
                    sens = labels[:, sensitive_attr_idx]
                else:
                    sens = labels # 兜底
            else:
                sens = labels
                
            sensitive_labels.extend(sens.cpu().numpy().tolist())
            collected_samples += imgs.size(0)

    # 3. Clean Hooks
    for h in handles: h.remove()

    # 4. Organize Data by Neuron and Group
    distributions = {}
    sensitive_labels = np.array(sensitive_labels)
    
    for name, acts in activations.items():
        n_neurons = acts.shape[1]
        distributions[name] = {}
        
        group0_mask = (sensitive_labels == 0)
        group1_mask = (sensitive_labels == 1)
        
        act_np = acts.numpy()
        
        for i in range(n_neurons):
            distributions[name][i] = {
                '0': act_np[group0_mask, i], 
                '1': act_np[group1_mask, i]  
            }
            
    return distributions

def apply_faar_distribution_reset(model, distributions, reset_ratio=0.05, logger=None):
    """
    FedFAAR 核心逻辑：基于 Wasserstein 距离的 Reset
    """
    bias_scores = [] 
    
    # 1. 计算 Bias Score
    for layer_name, neurons in distributions.items():
        for neuron_idx, groups in neurons.items():
            dist0 = groups['0']
            dist1 = groups['1']
            score = _get_wasserstein_distance(dist0, dist1)
            bias_scores.append((layer_name, neuron_idx, score))
            
    # 2. 排序并确定 Reset 阈值
    bias_scores.sort(key=lambda x: x[2], reverse=True)
    
    total_neurons = len(bias_scores)
    num_reset = int(total_neurons * reset_ratio)
    
    # 选出 Top-K 偏见神经元
    targets_to_reset = bias_scores[:num_reset]
    
    # 3. 执行 Reset
    reset_stats = {} 
    layer_indices = {}
    for lname, nidx, score in targets_to_reset:
        if lname not in layer_indices: layer_indices[lname] = []
        layer_indices[lname].append(nidx)
        
    for name, module in model.named_modules():
        if name in layer_indices:
            indices = layer_indices[name]
            w = module.weight.data
            mean = w.mean().item()
            std = w.std().item()
            
            for idx in indices:
                # 使用 Phoenix 风格的高斯重置
                if isinstance(module, nn.Conv2d):
                    new_kernel = torch.randn_like(w[idx]) * std + mean
                    module.weight.data[idx] = new_kernel
                elif isinstance(module, nn.Linear):
                    new_neuron = torch.randn_like(w[idx]) * std + mean
                    module.weight.data[idx] = new_neuron
            
            reset_stats[name] = len(indices)

    if logger:
        top_bias = bias_scores[0][2] if bias_scores else 0
        avg_bias = np.mean([x[2] for x in bias_scores]) if bias_scores else 0
        print(f"[FAAR] Max Wasserstein: {top_bias:.4f}, Avg: {avg_bias:.4f}, Reset: {num_reset}")
        
    return reset_stats