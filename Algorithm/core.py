# ~/ljy/AAAFedPhoenix/Algorithm/core.py

import torch
import numpy as np
import math

# =========================================================
# 1. CosineDecay: 余弦衰减学习率调度器
# ---------------------------------------------------------
class CosineDecay:
    """
    实现余弦学习率衰减。
    在训练过程中，学习率从初始值（LR）以余弦曲线的形式衰减到 min_lr。
    """
    def __init__(self, LR, min_lr, T_max):
        """
        Args:
            LR (float): 初始学习率 (Maximum learning rate).
            min_lr (float): 最小学习率 (Minimum learning rate, usually 0).
            T_max (int): 衰减周期内的总迭代次数或总轮次。
        """
        self.LR = LR
        self.min_lr = min_lr
        self.T_max = T_max
        self.delta = LR - min_lr # 学习率的范围

    def get_lr(self, epoch):
        """
        根据当前的 epoch (或迭代次数) 计算学习率。
        Args:
            epoch (int): 当前的训练轮次或迭代次数。
        Returns:
            float: 计算得到的学习率。
        """
        # 确保 epoch 不超过 T_max
        if epoch > self.T_max:
            return self.min_lr
        
        # 余弦衰减公式: lr_t = min_lr + 0.5 * delta * (1 + cos(pi * t / T_max))
        lr = self.min_lr + 0.5 * self.delta * (1 + math.cos(math.pi * epoch / self.T_max))
        return lr

# =========================================================
# 2. Masking: 权重掩码类（用于剪枝或模型稀疏化）
# ---------------------------------------------------------
class Masking:
    """
    实现模型权重掩码的工具类。
    用于在训练过程中对模型权重应用稀疏化或剪枝。
    """
    def __init__(self, model):
        """
        初始化掩码器。
        Args:
            model (torch.nn.Module): 要应用掩码的 PyTorch 模型。
        """
        self.model = model
        # 存储所有参数的掩码
        self.masks = self.init_masks() 
        # 用于存储参数的初始/全局值 (可能用于恢复)
        self.old_params = {} 

    def init_masks(self):
        """
        初始化所有参数的掩码，最初全部设置为 1 (即不进行掩码)。
        """
        masks = {}
        for name, param in self.model.named_parameters():
            # 只有可训练的权重参数才需要掩码
            if param.requires_grad:
                masks[name] = torch.ones_like(param)
        return masks

    def apply_mask(self):
        """
        将掩码应用到模型参数上。
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    # 参数 = 参数 * 掩码 (将掩码为 0 的权重清零)
                    param.data.mul_(self.masks[name].to(param.device))

    def update_mask(self, new_mask):
        """
        更新当前的掩码。
        Args:
            new_mask (dict): 包含新掩码的字典。
        """
        self.masks.update(new_mask)
        # 立即应用新掩码
        self.apply_mask()

    def load_masks(self, masks):
        """
        从外部加载掩码字典。
        """
        self.masks = masks
        self.apply_mask()

    def get_masks(self):
        """
        返回当前的掩码字典。
        """
        return self.masks

    def get_sparsity(self):
        """
        计算模型中稀疏化的比例（即掩码为 0 的比例）。
        """
        total_elements = 0
        masked_elements = 0
        for name, mask in self.masks.items():
            total_elements += mask.numel()
            masked_elements += (mask == 0).sum().item()
        
        if total_elements == 0:
            return 0.0
        return masked_elements / total_elements

    # FedPhoenix 可能需要的其他方法（例如，基于梯度或幅度的剪枝逻辑）
    # 如果运行仍有问题，可能需要补充更复杂的剪枝方法。
    
# --- 结束 ---