import torch
import torch.nn as nn
import torch.optim as optim

# ================= 1. 概率采样 =================

def select_neurons_by_temperature(scores, temperature=1.0, top_k=None):
    """
    根据怀疑度分数和温度系数，计算每个神经元的被选中概率。
    :param scores: Tensor, 形状 [C], 包含每个神经元的 DoubtDegree
    :param temperature: float, 温度系数 tau (图中公式)
                        tau 较大 -> 接近均匀采样
                        tau 很小 -> 接近 Hard Argmax
    :param top_k: int, 如果指定，则只返回概率最高的 k 个索引；如果不指定，则进行采样
    :return: selected_indices (LongTensor)
    """
    # 归一化 scores 以防数值过大溢出，也可以直接用
    # Softmax 公式: p_j = exp(w_j / tau) / sum(exp(w_k / tau))
    
    # 确保 scores 是 float
    scores = scores.float()
    
    if temperature < 1e-4:
        # 极低温度 -> 直接取 Top-K (Hard Selection)
        k = top_k if top_k else int(len(scores) * 0.05) # 默认 5%
        _, indices = torch.topk(scores, k)
        return indices
    
    # 计算 logits
    logits = scores / temperature
    probs = torch.softmax(logits, dim=0)
    
    if top_k is not None:
        # 方式 A: 即使有概率，也强制取概率最大的 Top-K (Deterministic based on prob)
        # 或者 方式 B: 根据概率无放回采样 Top-K
        # 这里使用方式 B: Weighted Random Sampling
        indices = torch.multinomial(probs, top_k, replacement=False)
        return indices
    else:
        # 如果不指定具体数量，可以通过阈值截断或随机掩码
        # 这里为了配合后续操作，建议始终传入一个期望处理的数量
        raise ValueError("Please specify top_k (number of neurons to treat).")

# ================= 2. 三种治疗手段 =================

# --- A. Mask (直接置零) ---
def treatment_mask(model, layer_name, neuron_indices):
    """
    将指定层、指定索引的卷积核权重置零
    """
    layer = dict(model.named_modules())[layer_name]
    if isinstance(layer, nn.Conv2d):
        with torch.no_grad():
            # weight shape: [Out, In, K, K]
            # 我们置零 Out Channel (Filter)
            layer.weight.data[neuron_indices, :, :, :] = 0.0
            if layer.bias is not None:
                layer.bias.data[neuron_indices] = 0.0
    print(f"✂️ [Mask] Pruned {len(neuron_indices)} neurons in {layer_name}")

# --- B. Reset (统计平滑重置) ---
def treatment_reset(model, layer_name, neuron_indices, mean_val=0.0, std_val=0.01):
    """
    将指定神经元重置为该层的均值/高斯分布，保留流动性但去除偏见信息
    """
    layer = dict(model.named_modules())[layer_name]
    if isinstance(layer, nn.Conv2d):
        with torch.no_grad():
            # 获取当前层的统计量（如果没传入，就现算）
            if mean_val == 0.0 and std_val == 0.01:
                current_mean = layer.weight.data.mean()
                current_std = layer.weight.data.std()
            else:
                current_mean, current_std = mean_val, std_val
            
            # 重新初始化这些 filters
            # 使用正态分布填充，模拟初始化状态或层统计态
            noise = torch.randn_like(layer.weight.data[neuron_indices])
            layer.weight.data[neuron_indices] = noise * current_std + current_mean
            
            if layer.bias is not None:
                layer.bias.data[neuron_indices] = 0.0
    print(f"🔄 [Reset] Soft-reset {len(neuron_indices)} neurons in {layer_name}")

# --- C. Operation (梯度反向微调 / Unlearning) ---
def treatment_operation(model, layer_name, neuron_indices, probe_loader, device, steps=5, lr=0.001):
    """
    对选中神经元进行'反向学习'：最大化男女特征差异的 Loss (即去除差异)
    或者 最小化特定公平性正则项
    """
    # 锁定其他参数，只训练选中神经元
    for param in model.parameters():
        param.requires_grad = False
    
    layer = dict(model.named_modules())[layer_name]
    # 这种精细控制比较难，通常需要 hook 或手动 mask 梯度
    # 简单实现：允许该层更新，但手动把非选中神经元的梯度置零
    layer.weight.requires_grad = True
    if layer.bias is not None: layer.bias.requires_grad = True
    
    optimizer = optim.SGD([layer.weight, layer.bias] if layer.bias is not None else [layer.weight], lr=lr)
    
    model.train()
    print(f"🔧 [Operation] Fine-tuning {len(neuron_indices)} neurons for {steps} steps...")
    
    iterator = iter(probe_loader)
    for i in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(probe_loader)
            batch = next(iterator)
            
        imgs, labels, sensitives = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        
        optimizer.zero_grad()
        output = model(imgs)
        
        # 这里定义 Operation 的目标：
        # 比如：让模型无法区分性别 (Maximizing Entropy of P(Gender|Act))
        # 或者简单的：最小化 (Act_Male - Act_Female)^2 
        # 为了演示，我们使用简单的 'Negative Fairness Loss' (假设我们有公平性 Loss)
        # 这里写一个简单的伪 Loss：让选中神经元对 sensitive 属性的互信息最小化
        # 实际简单实现：Random Noise Injection into Gradient? 
        # 按照用户描述："按照loss增大的方向走" -> Gradient Ascent on Bias?
        
        # 占位：标准 CrossEntropy 但针对 gender 做对抗？
        # 这里简单实现：Standard Loss 但只更新选中的 filter
        preds = output
        loss = nn.CrossEntropyLoss()(preds, labels) # 保持任务能力
        
        # 反向传播
        loss.backward()
        
        # === 关键：手动 Mask 梯度 ===
        # 只保留 neuron_indices 对应的梯度，其他置零
        with torch.no_grad():
            mask = torch.zeros_like(layer.weight.grad)
            mask[neuron_indices, :, :, :] = 1.0
            layer.weight.grad.mul_(mask)
            
            if layer.bias is not None:
                bias_mask = torch.zeros_like(layer.bias.grad)
                bias_mask[neuron_indices] = 1.0
                layer.bias.grad.mul_(bias_mask)
        
        optimizer.step()
    
    # 恢复 requires_grad
    for param in model.parameters():
        param.requires_grad = True
        
    print("   Done.")