#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import copy
import torch
import numpy as np
import os
import sys
from datetime import datetime  # <--- 新增
from torch.utils.data import DataLoader

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
from Algorithm.FAAR_util import extract_activation_distributions, apply_faar_distribution_reset

# ==============================================================================
# 自定义带时间戳的打印函数
# ==============================================================================
def log_print(msg):
    """
    格式: 14:34:55 [消息内容]
    """
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{current_time} {msg}")

# ==============================================================================
# 辅助函数：公平性测试
# ==============================================================================
def test_fairness(model, dataset, args):
    """
    专门针对 CelebA 的公平性测试函数。
    计算：Total Acc, Accuracy Gap (Male vs Female)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    
    correct = 0
    total = 0
    
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

    acc = 100.0 * correct / (total + 1e-6)
    acc_s0 = 100.0 * group_correct[0] / (group_total[0] + 1e-6)
    acc_s1 = 100.0 * group_correct[1] / (group_total[1] + 1e-6)
    
    # Gap 越小越公平
    acc_gap = abs(acc_s0 - acc_s1)
    
    return acc, acc_gap, acc_s0, acc_s1

# ==============================================================================
# 通用训练流程 (支持 FedAvg, FedPhoenix, FedFAAR)
# ==============================================================================
def run_training(args, net_glob, dataset_train, dataset_test, dict_users, logger):
    
    net_glob.train()
    
    # 准备 Probe Loader (用于 FAAR 提取分布)
    probe_loader = DataLoader(dataset_test, batch_size=100, shuffle=True, num_workers=0)
    
    # 准备保存目录
    save_dir = f'./save/{args.algorithm}_{args.dataset}_Beta{args.data_beta}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/distributions', exist_ok=True)

    log_print(f"🔥 开始训练: {args.algorithm} | Epochs: {args.epochs} | Clients: {args.num_users}")

    for iter in range(args.epochs):
        # 注意：这里使用 log_print 替代 print
        log_print(f'=== Round {iter:3d} ===')
        
        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # 1. 客户端本地训练
        for idx in idxs_users:
            net_local = copy.deepcopy(net_glob).to(args.device)
            
            # -----------------------------------------------------
            # 算法分支逻辑
            # -----------------------------------------------------
            
            # A. FedPhoenix (Baseline)
            if args.algorithm == 'FedPhoenix':
                reset_kernels_and_neurons_stair(
                    net_local, iter, args.FP_conv, conv_reset_ratio=args.reset
                )
            
            # B. FedFAAR (Ours)
            elif args.algorithm == 'FedFAAR':
                if iter > 0 and iter % args.faar_reset_period == 0:
                    log_print(f"[FAAR] Diagnosing Distribution Bias...")
                    
                    dists = extract_activation_distributions(
                        net_local, probe_loader, sensitive_attr_idx=1, device=args.device
                    )
                    stats = apply_faar_distribution_reset(
                        net_local, dists, reset_ratio=0.05, logger=logger
                    )

            # -----------------------------------------------------
            
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=net_local)
            
            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
            
        # 2. 聚合 (Aggregation)
        w_glob = Aggregation(w_locals, lens)
        net_glob.load_state_dict(w_glob)

        # 3. 测试与日志 (Testing & Logging)
        # 建议每 5 轮测一次以提速，这里您可以按需修改
        if iter % 5 == 0:
            if args.dataset == 'celeba':
                acc, gap, acc_f, acc_m = test_fairness(net_glob, dataset_test, args)
                # === 这里的输出就是您想要的格式 ===
                log_print(f"[TEST] Acc: {acc:.2f}% | Gap: {gap:.2f}% (F:{acc_f:.2f} vs M:{acc_m:.2f})")
                
                if logger:
                    logger.log_metric({
                        "test/acc": acc,
                        "fairness/gap": gap,
                        "fairness/acc_female": acc_f,
                        "fairness/acc_male": acc_m
                    }, epoch=iter)
            else:
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
                log_print(f"[TEST] Acc: {acc_test:.2f}% | Loss: {loss_test:.2f}")
                if logger:
                    logger.log_train(loss_test, acc_test, epoch=iter)

        # 4. 保存检查点 (Checkpointing)
        if iter % 20 == 0:
            ckpt_path = f"{save_dir}/model_round_{iter}.pth"
            torch.save(net_glob.state_dict(), ckpt_path)
            
            if args.dataset == 'celeba':
                log_print(f"[SAVE] Extracting & Saving distributions...")
                dists = extract_activation_distributions(
                    net_glob, probe_loader, sensitive_attr_idx=1, device=args.device
                )
                dist_path = f"{save_dir}/distributions/dist_round_{iter}.pt"
                torch.save(dists, dist_path)

    log_print("✅ Training Finished.")
    if logger: logger.finish()

# ==============================================================================
# 主入口
# ==============================================================================
if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_random_seed(args.seed)

    # 2. 加载数据集
    if args.dataset == 'celeba':
        dataset_train, dataset_test, dict_users = get_celeba_dataset_raw(args)
        num_classes = 2 
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
    run_name = f"{args.algorithm}_{args.dataset}_Beta{args.data_beta}"
    logger = WandbLogger(args=args, name=run_name, active=True)

    # 5. 开始训练
    run_training(args, net_glob, dataset_train, dataset_test, dict_users, logger)
