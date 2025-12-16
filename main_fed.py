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
from datetime import datetime
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
from Algorithm.FAAR_util import extract_activation_distributions, apply_faar_reset

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
# 通用训练流程 (支持断点续训 + 完整日志)
# ==============================================================================
def run_training(args, net_glob, dataset_train, dataset_test, dict_users, logger):
    
    net_glob.train()
    
    # 1. 准备 FAAR 探针
    # 为了提速，只取 500 张做分布探针
    probe_indices = list(range(500)) 
    if len(dataset_test) < 500: probe_indices = list(range(len(dataset_test)))
    from torch.utils.data import Subset
    probe_set = Subset(dataset_test, probe_indices)
    probe_loader = DataLoader(probe_set, batch_size=100, shuffle=False, num_workers=2)

    # 准备保存目录 (支持后缀)
    suffix = args.suffix if args.suffix else ""
    save_dir = f'./save/{args.algorithm}{suffix}_{args.dataset}_Beta{args.data_beta}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/distributions', exist_ok=True)

    # =======================================================
    # 断点续训加载逻辑
    # =======================================================
    start_round = 0
    if args.resume:
        # 逻辑：AutoRun 传 start_epoch=21，我们这里 -1 读 round_20.pth
        load_round = args.start_epoch - 1
        load_path = f"{save_dir}/model_round_{load_round}.pth"
        
        if os.path.exists(load_path):
            log_print(f"🔄 Resuming training from Round {args.start_epoch} (Loading: {load_path})...")
            try:
                state_dict = torch.load(load_path, map_location=args.device)
                net_glob.load_state_dict(state_dict)
                start_round = args.start_epoch
                log_print("✅ Checkpoint loaded successfully!")
            except Exception as e:
                log_print(f"❌ Error loading checkpoint: {e}")
                exit(1)
        else:
            log_print(f"❌ Error: Checkpoint not found at {load_path}. Starting from scratch.")
            start_round = 0

    log_print(f"🔥 开始训练: {args.algorithm} | Rounds: {start_round} -> {args.epochs}")

    # =======================================================
    # 主循环
    # =======================================================
    for iter in range(start_round, args.epochs):
        log_print(f'=== Round {iter:3d} ===')
        
        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # --- Client Local Training ---
        for idx in idxs_users:
            net_local = copy.deepcopy(net_glob).to(args.device)
            
            # [FedPhoenix / FedFAAR 本地预处理]
            if args.algorithm in ['FedPhoenix', 'FedFAAR']:
                reset_kernels_and_neurons_stair(
                    net_local, iter, args.FP_conv, conv_reset_ratio=args.reset
                )
            
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=net_local)
            
            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
            
        # --- Server Aggregation ---
        w_glob = Aggregation(w_locals, lens)
        net_glob.load_state_dict(w_glob)

        # --- FedFAAR Post-Aggregation Intervention ---
        if args.algorithm == 'FedFAAR':
            if iter >= args.faar_warmup and (iter % args.faar_reset_period == 0):
                log_print(f"[FAAR] Diagnosing Bias (Round {iter})...")
                dists = extract_activation_distributions(net_glob, probe_loader, sensitive_attr_idx=1, device=args.device)
                
                # 执行 Soft-Reset
                reset_stats = apply_faar_reset(net_glob, dists, args, logger)
                
                # 保存统计 (可选，占空间可注释)
                # torch.save(reset_stats, f"{save_dir}/distributions/reset_stats_round_{iter}.pt")

        # --- Testing & Logging (★修正：每5轮测一次) ---
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

        # --- Checkpointing (每10轮保存一次，方便续训) ---
        if iter % 10 == 0 or iter == args.epochs - 1:
            ckpt_path = f"{save_dir}/model_round_{iter}.pth"
            torch.save(net_glob.state_dict(), ckpt_path)

    log_print("✅ Training Finished.")
    if logger: logger.finish()

# ==============================================================================
# 主入口
# ==============================================================================
if __name__ == '__main__':
    args = args_parser()
    
    # [OOM 修复] 确保 device 严格跟随 args.gpu (虽然 AutoRun 做了物理隔离，这里双保险)
    if torch.cuda.is_available() and args.gpu != -1:
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')

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
    suffix = args.suffix if args.suffix else ""
    run_name = f"{args.algorithm}{suffix}_{args.dataset}_Beta{args.data_beta}"
    logger = WandbLogger(args=args, name=run_name, active=args.use_wandb)

    # 5. 开始训练
    run_training(args, net_glob, dataset_train, dataset_test, dict_users, logger)