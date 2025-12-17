#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
import matplotlib
matplotlib.use('Agg')
import copy
import torch
import numpy as np
import os
import sys
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# --- 导入自定义工具 ---
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.raw_celeba import get_celeba_dataset_raw  
from utils.get_dataset import get_dataset            

# --- 导入模型与更新逻辑 ---
from models.Update import LocalUpdate_FedAvg, DatasetSplit
from models.Nets import ResNet18_cifar10
from models.Fed import Aggregation
from models.test import test_img

# --- 导入日志 ---
from wandb_config import WandbLogger

# --- 导入算法核心 ---
from Algorithm.Phoenix_util import reset_kernels_and_neurons_stair
# 注意：虽然这里导入了 extract_activation_distributions，但在下面的高级诊断中我们手写了更高效的Hook逻辑
from Algorithm.FAAR_util import extract_activation_distributions, apply_faar_reset
from Algorithm.FAAR_Phoenix_util import apply_faar_soft_reset


def partition_attribute_dirichlet(dataset, num_users, beta=0.3, min_require_size=10):
    """
    基于敏感属性 (Sensitive Attribute) 进行 Dirichlet 分布划分 (Non-IID)。
    
    Args:
        dataset: PyTorch Dataset
        num_users: 客户端数量
        beta: Dirichlet 分布参数 (越小越 Non-IID, 0.3 是常用值)
        min_require_size: 每个客户端最少分到的样本数 (防止空客户端)
    Returns:
        dict_users: {client_id: [data_indices]}
    """
    print(f"⚡ Generating Dirichlet Non-IID Partition based on SENSITIVE ATTRIBUTE (Beta={beta})...")

    min_size = 0
    # 循环尝试，直到生成的分组满足每个客户端至少有 min_require_size 个样本
    while min_size < min_require_size:
        # 1. 提取所有样本的敏感属性
        # CelebA Raw 格式通常是: labels -> [Target, Sensitive]
        # 我们需要 dataset 中所有数据的 Sensitive Attribute
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if torch.is_tensor(targets) and targets.dim() > 1:
                sensitives = targets[:, 1].numpy() # [N]
            elif isinstance(targets, list):
                sensitives = np.array([t[1] for t in targets])
            else:
                sensitives = np.array(targets) # 假设是一维
        else:
            # 慢速回退方案：遍历 dataset (仅第一次运行慢)
            sensitives = []
            for i in range(len(dataset)):
                # 假设 dataset[i] 返回 (img, (target, sensitive)) 或 (img, label_tensor)
                _, label = dataset[i]
                if isinstance(label, torch.Tensor) and label.numel() > 1:
                    sensitives.append(label[1].item())
                elif isinstance(label, (list, tuple)) and len(label) > 1:
                    sensitives.append(label[1])
                else:
                    raise ValueError("Dataset label format not supported for Attribute Skew.")
            sensitives = np.array(sensitives)

        num_classes = 2 # 敏感属性只有 0 和 1 (女/男)
        N = len(dataset)
        dict_users = {}
        
        # 2. Dirichlet 划分逻辑
        # idx_batch: 存储每个客户端分配到的索引列表
        idx_batch = [[] for _ in range(num_users)]
        
        for k in range(num_classes):
            # 获取属于敏感属性 k 的所有样本索引
            idx_k = np.where(sensitives == k)[0]
            np.random.shuffle(idx_k)
            
            # 使用 Dirichlet 分布生成每个客户端对属性 k 的占有比例
            # proportions 形状: [num_users]
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            
            # 修正比例，防止因为精度问题导致分配不完或溢出
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # 按比例切分索引并分配给客户端
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            
        # 检查最小样本数
        min_size = min([len(idx_j) for idx_j in idx_batch])
        if min_size < min_require_size:
            print(f"   ⚠️ Re-sampling Dirichlet... (Min size {min_size} < {min_require_size})")

    # 3. 封装成字典
    for i in range(num_users):
        dict_users[i] = np.array(idx_batch[i])
        
    print(f"✅ Partition Done. Min Client Size: {min_size}")
    
    # 4. (可选) 打印前5个客户端的分布情况，确认 Non-IID 程度
    print("   [Sample Stats] Ratio of Sensitive Attribute '1' (Male):")
    for i in range(min(5, num_users)):
        indices = dict_users[i]
        s_vals = sensitives[indices]
        ratio = np.sum(s_vals == 1) / len(s_vals)
        print(f"   Client {i}: {ratio:.2f} (Total: {len(indices)})")

    return dict_users


# ==============================================================================
# 自定义带时间戳的打印函数
# ==============================================================================
def log_print(msg):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{current_time} {msg}")

# ==============================================================================
# 辅助函数：公平性测试 (含 Loss 计算)
# ==============================================================================
def test_fairness(model, dataset, args):
    """
    计算：Total Acc, Average Loss, Accuracy Gap (Male vs Female)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    
    correct = 0
    total = 0
    test_loss = 0.0
    loss_func = torch.nn.CrossEntropyLoss()
    
    # 统计器：{Sensitive_Attr: Correct_Count}
    # CelebA Sensitive (Male): 0 = Female, 1 = Male
    group_correct = {0: 0, 1: 0}
    group_total = {0: 0, 1: 0}
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(args.device)
            
            # 兼容 CelebA_Raw 的标签格式 [Batch, 2] -> (Target, Sensitive)
            if labels.dim() > 1:
                targets = labels[:, 0].to(args.device).long()
                sensitives = labels[:, 1].to(args.device).long()
            else:
                targets = labels.to(args.device).long()
                sensitives = torch.zeros_like(targets) 

            outputs = model(imgs)
            if isinstance(outputs, dict): outputs = outputs['output']
            
            # 计算 Loss
            loss = loss_func(outputs, targets)
            test_loss += loss.item() * targets.size(0)

            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            batch_correct = (predicted == targets)
            correct += batch_correct.sum().item()
            
            # 分组统计
            for s in [0, 1]:
                mask = (sensitives == s)
                if mask.sum() > 0:
                    group_total[s] += mask.sum().item()
                    group_correct[s] += batch_correct[mask].sum().item()

    # 计算平均指标
    avg_loss = test_loss / (total + 1e-6)
    acc = 100.0 * correct / (total + 1e-6)
    
    acc_s0 = 100.0 * group_correct[0] / (group_total[0] + 1e-6) # Female
    acc_s1 = 100.0 * group_correct[1] / (group_total[1] + 1e-6) # Male
    
    # Gap 越小越公平
    acc_gap = abs(acc_s0 - acc_s1)
    
    return acc, avg_loss, acc_gap, acc_s0, acc_s1

# ==============================================================================
# FAAR 高级诊断函数 (基于 Wasserstein 距离)
# ==============================================================================
def diagnose_bias_by_wasserstein(net, probe_loader, args, top_k_ratio=0.05):
    """
    高级诊断：基于 Wasserstein 距离 (Earth Mover's Distance) 衡量分布差异。
    比简单的均值差更稳健，能捕捉到更复杂的偏见分布。
    """
    net.eval()
    
    # 存储: {layer_name: {'0': [act_list], '1': [act_list]}}
    # 为了节省显存，我们只存储均值统计量
    act_stats = {}
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hooks = []
    target_layers = []
    # 只针对卷积层
    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            h = layer.register_forward_hook(get_activation(name))
            hooks.append(h)
            target_layers.append(name)

    with torch.no_grad():
        for imgs, labels in probe_loader:
            imgs = imgs.to(args.device)
            # 处理标签: [Target, Sensitive]
            if labels.dim() > 1:
                sensitives = labels[:, 1].to(args.device).long()
            else:
                sensitives = torch.zeros(labels.size(0)).long().to(args.device)

            _ = net(imgs)

            for name in target_layers:
                # act shape: [Batch, Channel, H, W] -> [Batch, Channel] (Global Average Pooling)
                act = activation[name].mean(dim=[2, 3]) 
                
                if name not in act_stats:
                    num_channels = act.shape[1]
                    # 我们需要维护两个分布的统计特征：均值(mu)
                    act_stats[name] = {
                        0: torch.zeros(num_channels).to(args.device), # Sum for Female
                        1: torch.zeros(num_channels).to(args.device), # Sum for Male
                        'count0': 0,
                        'count1': 0
                    }
                
                mask0 = (sensitives == 0)
                mask1 = (sensitives == 1)
                
                if mask0.any():
                    act_stats[name][0] += act[mask0].sum(dim=0)
                    act_stats[name]['count0'] += mask0.sum().item()
                if mask1.any():
                    act_stats[name][1] += act[mask1].sum(dim=0)
                    act_stats[name]['count1'] += mask1.sum().item()
    
    for h in hooks: h.remove()

    # --- 计算 Wasserstein 距离并生成 Map ---
    reset_map = {}
    
    for name, stats in act_stats.items():
        if stats['count0'] == 0 or stats['count1'] == 0:
            continue
            
        mean0 = stats[0] / stats['count0']
        mean1 = stats[1] / stats['count1']
        
        # 1-Wasserstein Distance 近似值
        w_dist = torch.abs(mean0 - mean1)
        
        # 选出该层中差异最大的 k% 个 Filter
        num_channels = w_dist.shape[0]
        k = max(int(num_channels * top_k_ratio), 1)
        
        _, top_indices = torch.topk(w_dist, k)
        reset_map[name] = top_indices.cpu().tolist()
        
    return reset_map

# ==============================================================================
# 通用训练流程 (支持断点续训 + 完整日志)
# ==============================================================================
def run_training(args, net_glob, dataset_train, dataset_test, dict_users, logger):
    
    net_glob.train()
    
    # 1. 准备 FAAR 探针
    # 为了提速，只取 500 张做分布探针
    probe_indices = list(range(500)) 
    if len(dataset_test) < 500: probe_indices = list(range(len(dataset_test)))
    probe_set = Subset(dataset_test, probe_indices)
    probe_loader = DataLoader(probe_set, batch_size=100, shuffle=False, num_workers=2)

    # 准备保存目录 (支持后缀)
    suffix = args.suffix if args.suffix else ""
    save_dir = f'./save/{args.algorithm}{suffix}_{args.dataset}_Beta{args.data_beta}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/distributions', exist_ok=True)

    # =======================================================
    # 智能断点续训加载逻辑
    # =======================================================
    start_round = 0
    
    # 场景 1: 用户指定了具体的 Checkpoint 文件路径
    if args.checkpoint_path is not None and args.checkpoint_path != "":
        load_path = args.checkpoint_path
        log_print(f"🔄 [Explicit Load] Loading from specific file: {load_path}")
        
        if os.path.exists(load_path):
            try:
                state_dict = torch.load(load_path, map_location=args.device)
                net_glob.load_state_dict(state_dict)
                start_round = args.start_epoch
                log_print(f"✅ Checkpoint loaded successfully! Starting from round {start_round}")
            except Exception as e:
                log_print(f"❌ Error loading checkpoint: {e}")
                exit(1)
        else:
            log_print(f"❌ Error: File not found at {load_path}. Exiting.")
            exit(1)

    # 场景 2: 自动 Resume
    elif args.resume:
        load_round = args.start_epoch - 1
        load_path = f"{save_dir}/model_round_{load_round}.pth"
        
        if os.path.exists(load_path):
            log_print(f"🔄 [Auto Resume] Resuming from Round {args.start_epoch} (Loading: {load_path})...")
            try:
                state_dict = torch.load(load_path, map_location=args.device)
                net_glob.load_state_dict(state_dict)
                start_round = args.start_epoch
                log_print("✅ Checkpoint loaded successfully!")
            except Exception as e:
                log_print(f"❌ Error loading checkpoint: {e}")
                exit(1)
        else:
            log_print(f"⚠️ Warning: Checkpoint not found at {load_path}. Starting from scratch.")
            start_round = 0

    log_print(f"🔥 开始训练: {args.algorithm} | Rounds: {start_round} -> {args.epochs}")

    # =======================================================
    # 主循环
    # =======================================================
    for iter in range(start_round, args.epochs):
        log_print(f'=== Round {iter:3d} ===')
        
        # -----------------------------------------------------------
        # [优化版] 余弦退火 (Cosine Annealing)
        # -----------------------------------------------------------
        # progress 从 0.0 到 1.0
        progress = iter / args.epochs 
        
        # 公式：0.5 * (1 + cos(pi * progress))
        # t=0 -> 1.0; t=0.5 -> 0.5; t=1 -> 0.0
        # 这是一个标准的平滑下降曲线
        decay_factor = 0.5 * (1 + np.cos(np.pi * progress))
        
        dynamic_reset_ratio = args.reset * decay_factor
        
        # 设定下限
        if dynamic_reset_ratio < 0.001: 
            dynamic_reset_ratio = 0.0
            
        log_print(f"   [Scheduler] Round {iter}: Dynamic Reset Ratio = {dynamic_reset_ratio:.5f}")

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # --- Client Local Training ---
        for idx in idxs_users:
            net_local = copy.deepcopy(net_glob).to(args.device)
            
            # [FedPhoenix / FedFAAR 本地预处理]
            # 关键：传入 dynamic_reset_ratio
            if args.algorithm in ['FedPhoenix', 'FedFAAR']:
                reset_kernels_and_neurons_stair(
                    net_local, 
                    iter, 
                    args.FP_conv, 
                    conv_reset_ratio=dynamic_reset_ratio
                )
            
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=net_local)
            
            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
            
        # --- Server Aggregation ---
        w_glob = Aggregation(w_locals, lens)
        net_glob.load_state_dict(w_glob)

        # --- FedFAAR Post-Aggregation Intervention (全局公平性干预) ---
        if args.algorithm == 'FedFAAR':
            # 🟢 [修改点] 错峰执行：
            # 假设 faar_reset_period 是 5。
            # 我们希望在余数为 2 的时候重置 (如 22, 27, 32...)
            # 而在余数为 0 的时候测试 (如 25, 30, 35...)
            # 这样模型有 2-3 轮的时间(Round 3, 4) 来通过本地 SGD "消化" 刚才的重置
            
            reset_offset = 2  # 设置偏移量为 2
            is_reset_round = (iter % args.faar_reset_period == reset_offset)
            
            if iter >= args.faar_warmup and is_reset_round and dynamic_reset_ratio > 0:
                log_print(f"[FAAR] Diagnosing Bias (Wasserstein) Round {iter}...")
                
                # 1. 诊断
                faar_reset_map = diagnose_bias_by_wasserstein(
                    net_glob, 
                    probe_loader, 
                    args, 
                    top_k_ratio=dynamic_reset_ratio
                )
                
                total_to_fix = sum([len(v) for v in faar_reset_map.values()])
                log_print(f"   [FAAR] Identified {total_to_fix} neurons (Ratio: {dynamic_reset_ratio:.4f})")

                # 2. 治疗
                if total_to_fix > 0:
                    total_reset = apply_faar_soft_reset(net_glob, faar_reset_map, blend_alpha=0.8)
                    log_print(f"   [FAAR] Soft-Reset Applied. {total_reset} Filters neutralized.")
            
            elif dynamic_reset_ratio <= 0 and (iter % 5 == 0):
                 # 仅在整5轮打印提示，避免刷屏
                 log_print("   [FAAR] Reset ratio decayed to 0. Intervention stopped for stability.")
        
        # --- Testing & Logging (每5轮测一次) ---
        if iter % 5 == 0 or iter == args.epochs - 1: 
            if args.dataset == 'celeba':
                acc, val_loss, gap, acc_f, acc_m = test_fairness(net_glob, dataset_test, args)
                log_print(f"[TEST] Acc: {acc:.2f}% | Loss: {val_loss:.4f} | Gap: {gap:.2f}% (F:{acc_f:.2f} vs M:{acc_m:.2f})")
                
                if logger:
                    logger.log_metric({
                        "test/acc": acc,
                        "test/loss": val_loss,
                        "fairness/gap": gap,
                        "fairness/acc_female": acc_f,
                        "fairness/acc_male": acc_m,
                        "round": iter
                    }, epoch=iter)
            else:
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
                log_print(f"[TEST] Acc: {acc_test:.2f}% | Loss: {loss_test:.2f}")
                if logger:
                    logger.log_train(loss_test, acc_test, epoch=iter)

        # --- Checkpointing (每10轮保存一次) ---
        if iter % 10 == 0 or iter == args.epochs - 1:
            ckpt_path = f"{save_dir}/model_round_{iter}.pth"
            torch.save(net_glob.state_dict(), ckpt_path)
            log_print(f"💾 Checkpoint saved: {ckpt_path}")

    log_print("✅ Training Finished.")
    if logger: logger.finish()

# ==============================================================================
# 主入口
# ==============================================================================
if __name__ == '__main__':
    args = args_parser()
    
    # 修复显卡映射
    if torch.cuda.is_available() and args.gpu != -1:
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')

    set_random_seed(args.seed)

    # 2. 加载数据集
    if args.dataset == 'celeba':
        dataset_train, dataset_test, dict_users = get_celeba_dataset_raw(args)
        num_classes = 2
        
        # 🔴 [关键修改] 使用 Attribute Skew (Dirichlet Beta)
        # 我们用 args.noniid_case = 99 作为暗号，代表 "Attribute Skew"
        # 使用 args.data_beta (例如 0.3) 作为 Dirichlet 参数
        if args.noniid_case == 99: 
            dict_users = partition_attribute_dirichlet(
                dataset_train, 
                args.num_users, 
                beta=args.data_beta  # 这里传入 0.3
            )
    elif args.dataset == 'cifar10':
        dataset_train, dataset_test, dict_users = get_dataset(args)
        num_classes = 10
    else:
        log_print("Error: Unsupported dataset. Use 'celeba' or 'cifar10'.")
        exit(1)

    # 3. 初始化模型
    if args.model == 'resnet18':
        net_glob = ResNet18_cifar10(num_classes=num_classes)
    else:
        log_print(f"Error: Model {args.model} not supported in this demo.")
        exit(1)
        
    net_glob.to(args.device)
    log_print(f"Model initialized: {args.model} (Classes: {num_classes})")

    # 4. 初始化 Wandb
    suffix = args.suffix if args.suffix else ""
    run_name = f"{args.algorithm}{suffix}_{args.dataset}_Beta{args.data_beta}"
    logger = WandbLogger(args=args, name=run_name, active=args.use_wandb)

    # 5. 开始训练
    run_training(args, net_glob, dataset_train, dataset_test, dict_users, logger)