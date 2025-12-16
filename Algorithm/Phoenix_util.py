import matplotlib
matplotlib.use('Agg')
import copy
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np

def get_reset_probability(current_round, m, p, initial_mu):
    """计算当前轮次的Reset概率"""
    if current_round <= m:
        return initial_mu
    elif current_round <= m + p:
        # 线性递减
        progress = (current_round - m) / p
        return initial_mu * (1 - progress)
    else:
        return 0.0

def zero_out_model_params(model, percent_set_zero_base):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            percent_set_zero = percent_set_zero_base * 1 
        elif isinstance(layer, nn.Linear):
            # 全连接层设定较高的置零比例
            percent_set_zero = percent_set_zero_base * 1
        else:
            percent_set_zero = percent_set_zero_base * 0
            
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if hasattr(layer, 'weight'):
                param = layer.weight
                total_params = param.numel()
                num_zero_params = int(total_params * percent_set_zero)
                mask = torch.ones_like(param)
                zero_indices = np.random.choice(total_params, num_zero_params, replace=False)
                mask.view(-1)[zero_indices] = 0
                param.data.mul_(mask)
    return model

def get_test_image(dataset, seed=42):
    """
    Get a test image from the dataset
    """
    torch.manual_seed(seed)
    idx = torch.randint(len(dataset), (1,)).item()
    img, label = dataset[idx]
    
    # Denormalize the image for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    orig_img = img * std + mean
    
    return orig_img, img, label

def reset_kernels_and_neurons_stair(model, current_iter, conv_transition_period, conv_reset_ratio=0.03125, fc_reset_ratio=0, scale_factor=1, init_method='ori_normal'):
    """
    Stop resetting convolutional kernels and fully connected layer neurons in a stepwise manner
    """
    if not (0 <= conv_reset_ratio <= 1 and 0 <= fc_reset_ratio <= 1):
        raise ValueError("Reset ratio must be between 0 and 1")

    # Separately get convolutional layers and fully connected layers
    conv_layers = []
    fc_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
        elif isinstance(module, nn.Linear):
            fc_layers.append((name, module))

    reset_count = 0
    stopped_count = 0  # Used to count the number of layers that have stopped resetting

    # Process convolutional layers
    total_conv = len(conv_layers)
    for depth, (name, layer) in enumerate(conv_layers):
        # Calculate the iteration threshold for this layer to stop resetting
        stop_iter = (depth + 1) * (conv_transition_period / total_conv)

        # If the current iteration has exceeded the stop threshold for this layer, skip this layer
        if current_iter >= stop_iter:
            stopped_count += 1  # Count the number of layers that have stopped resetting
            continue

        current_mean = layer.weight.data.mean().item()
        current_std = layer.weight.data.std().item()

        num_kernels = layer.weight.shape[0]
        num_reset = int(num_kernels * conv_reset_ratio)

        if num_reset > 0:
            reset_indices = random.sample(range(num_kernels), num_reset)
            
            for idx in reset_indices:
                if init_method == 'uniform':
                    new_kernel = torch.zeros_like(layer.weight[idx]).uniform_(
                        current_mean - current_std * scale_factor, 
                        current_mean + current_std * scale_factor
                    )
                elif init_method == 'kaiming_uniform':
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = np.sqrt(6.0 / fan_in) * scale_factor
                    new_kernel = torch.zeros_like(layer.weight[idx]).uniform_(-bound, bound)
                elif init_method == 'kaiming_normal':
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    std = np.sqrt(2.0 / fan_in) * scale_factor
                    new_kernel = torch.randn_like(layer.weight[idx]) * std
                elif init_method == 'ori_normal':
                    new_kernel = torch.randn_like(layer.weight[idx]) * current_std * scale_factor + current_mean
                else:
                    raise ValueError("Unsupported initialization method")
                
                layer.weight.data[idx] = new_kernel
            
            reset_count += 1
            
    return reset_count, stopped_count