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
from collections import deque
from torch.utils.data import DataLoader, Subset, TensorDataset

# --- 导入自定义工具 ---
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.raw_celeba import get_celeba_dataset_raw  
from utils.get_dataset import get_dataset            

# --- 导入模型与更新逻辑 ---
from models.Update import LocalUpdate_FedAvg, DatasetSplit
from models.Nets import ResNet18_cifar10, SimpleCNN 
from models.Fed import Aggregation
from models.test import test_img

# --- 导入日志 ---
from wandb_config import WandbLogger

# --- 导入算法核心 ---
from Algorithm.Phoenix_util import reset_kernels_and_neurons_stair
from Algorithm.FAAR_util import extract_activation_distributions, apply_faar_reset
from Algorithm.FAAR_Phoenix_util import apply_faar_soft_reset


class MovingAverageLogger:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
    def update(self, metrics):
        """
        metrics: dict {'acc': 90, 'loss': 0.5, ...}
        """
        self.history.append(metrics)
        
    def get_averages(self):
        if not self.history:
            return {}
            
        avg_metrics = {}
        # 假设 metrics 里的所有 value 都是数字
        keys = self.history[0].keys()
        for k in keys:
            # 计算平均值
            avg_metrics[k] = sum(d[k] for d in self.history) / len(self.history)
        return avg_metrics

def partition_attribute_dirichlet(dataset, num_users, beta=0.3, min_require_size=10):
    """
    基于敏感属性 (Sensitive Attribute) 进行 Dirichlet 分布划分 (Non-IID)。
    """
    print(f"⚡ Generating Dirichlet Non-IID Partition based on SENSITIVE ATTRIBUTE (Beta={beta})...")

    min_size = 0
    while min_size < min_require_size:
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if torch.is_tensor(targets) and targets.dim() > 1:
                sensitives = targets[:, 1].numpy() 
            elif isinstance(targets, list):
                sensitives = np.array([t[1] for t in targets])
            else:
                sensitives = np.array(targets)
        else:
            sensitives = []
            for i in range(len(dataset)):
                _, label = dataset[i]
                if isinstance(label, torch.Tensor) and label.numel() > 1:
                    sensitives.append(label[1].item())
                elif isinstance(label, (list, tuple)) and len(label) > 1:
                    sensitives.append(label[1])
                else:
                    raise ValueError("Dataset label format not supported for Attribute Skew.")
            sensitives = np.array(sensitives)

        num_classes = 2 
        N = len(dataset)
        dict_users = {}
        
        idx_batch = [[] for _ in range(num_users)]
        
        for k in range(num_classes):
            idx_k = np.where(sensitives == k)[0]
            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            
        min_size = min([len(idx_j) for idx_j in idx_batch])
        if min_size < min_require_size:
            print(f"   ⚠️ Re-sampling Dirichlet... (Min size {min_size} < {min_require_size})")

    for i in range(num_users):
        dict_users[i] = np.array(idx_batch[i])
        
    print(f"✅ Partition Done. Min Client Size: {min_size}")
    
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

def inspect_data_distribution(client_datasets):
    """
    统计并打印所有 Client 的数据分布
    """
    print("\n📊 =========================================================")
    print("🕵️  [Data Inspection] Inspecting Biased Dataset Structure...")
    print("    Format: Total | Male (Smile%) | Female (Smile%)")
    print("---------------------------------------------------------")
    
    total_samples = 0
    total_male = 0
    total_female = 0
    
    sorted_ids = sorted(client_datasets.keys())
    
    for cid in sorted_ids:
        dataset = client_datasets[cid]
        _, targets, sensitives = dataset.tensors
        
        n_total = len(targets)
        total_samples += n_total
        
        mask_male = (sensitives == 1)
        mask_female = (sensitives == 0)
        n_male = mask_male.sum().item()
        n_female = mask_female.sum().item()
        
        total_male += n_male
        total_female += n_female
        
        if n_male > 0:
            n_male_smile = (targets[mask_male] == 1).sum().item()
            ratio_male_smile = (n_male_smile / n_male) * 100
        else:
            n_male_smile = 0
            ratio_male_smile = 0.0
            
        if n_female > 0:
            n_female_smile = (targets[mask_female] == 1).sum().item()
            ratio_female_smile = (n_female_smile / n_female) * 100
        else:
            n_female_smile = 0
            ratio_female_smile = 0.0
            
        tag = "🔴 Min" if cid < 4 else "🔵 Maj"
        
        print(f"Client {cid:2d} {tag} : {n_total:4d} | "
              f"Male: {n_male:3d} (Smile: {ratio_male_smile:5.1f}%) | "
              f"Female: {n_female:3d} (Smile: {ratio_female_smile:5.1f}%)")

    print("---------------------------------------------------------")
    print(f"✅ Global Stats:")
    print(f"   Total Images used for Training: {total_samples}")
    print(f"   Total Clients: {len(client_datasets)}")
    print(f"   Male Ratio: {total_male/total_samples*100:.1f}% | Female Ratio: {total_female/total_samples*100:.1f}%")
    print("=========================================================\n")

# ==============================================================================
# 辅助函数：公平性测试 (升级版：含子群分析)
# ==============================================================================
def test_fairness(model, dataset, args):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    
    total = 0
    test_loss = 0.0
    correct = 0
    loss_func = torch.nn.CrossEntropyLoss()
    
    counts = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    corrects = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(args.device)
            
            if labels.dim() > 1:
                targets = labels[:, 0].to(args.device).long()
                sensitives = labels[:, 1].to(args.device).long()
            else:
                targets = labels.to(args.device).long()
                sensitives = torch.zeros_like(targets) 

            outputs = model(imgs)
            if isinstance(outputs, dict): outputs = outputs['output']
            
            loss = loss_func(outputs, targets)
            test_loss += loss.item() * targets.size(0)

            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            batch_correct = (predicted == targets)
            correct += batch_correct.sum().item()
            
            for s in [0, 1]:
                for t in [0, 1]:
                    mask = (sensitives == s) & (targets == t)
                    if mask.sum() > 0:
                        counts[s][t] += mask.sum().item()
                        corrects[s][t] += batch_correct[mask].sum().item()

    avg_loss = test_loss / (total + 1e-6)
    acc_global = 100.0 * correct / (total + 1e-6)
    
    total_f = counts[0][0] + counts[0][1]
    total_m = counts[1][0] + counts[1][1]
    acc_female = 100.0 * (corrects[0][0] + corrects[0][1]) / (total_f + 1e-6)
    acc_male   = 100.0 * (corrects[1][0] + corrects[1][1]) / (total_m + 1e-6)
    
    gap = abs(acc_female - acc_male)
    
    acc_male_smile = 100.0 * corrects[1][1] / (counts[1][1] + 1e-6)
    acc_female_nosmile = 100.0 * corrects[0][0] / (counts[0][0] + 1e-6)
    
    acc_worst = min(acc_female, acc_male)

    return {
        'loss': avg_loss,
        'acc': acc_global,
        'gap': gap,
        'acc_f': acc_female,
        'acc_m': acc_male,
        'acc_male_smile': acc_male_smile,
        'acc_female_nosmile': acc_female_nosmile,
        'acc_worst': acc_worst
    }


# ==============================================================================
# FAAR 高级诊断函数 (基于 Wasserstein 距离)
# ==============================================================================
def diagnose_bias_by_wasserstein(net, probe_loader, args, top_k_ratio=0.05):
    net.eval()
    act_stats = {}
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hooks = []
    target_layers = []
    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            h = layer.register_forward_hook(get_activation(name))
            hooks.append(h)
            target_layers.append(name)

    with torch.no_grad():
        for imgs, labels in probe_loader:
            imgs = imgs.to(args.device)
            if labels.dim() > 1:
                sensitives = labels[:, 1].to(args.device).long()
            else:
                sensitives = torch.zeros(labels.size(0)).long().to(args.device)

            _ = net(imgs)

            for name in target_layers:
                act = activation[name].mean(dim=[2, 3]) 
                
                if name not in act_stats:
                    num_channels = act.shape[1]
                    act_stats[name] = {
                        0: torch.zeros(num_channels).to(args.device), 
                        1: torch.zeros(num_channels).to(args.device), 
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

    reset_map = {}
    for name, stats in act_stats.items():
        if stats['count0'] == 0 or stats['count1'] == 0:
            continue
            
        mean0 = stats[0] / stats['count0']
        mean1 = stats[1] / stats['count1']
        
        w_dist = torch.abs(mean0 - mean1)
        
        num_channels = w_dist.shape[0]
        k = max(int(num_channels * top_k_ratio), 1)
        
        _, top_indices = torch.topk(w_dist, k)
        reset_map[name] = top_indices.cpu().tolist()
        
    return reset_map

# ==============================================================================
# 通用训练流程 (支持断点续训 + 完整日志)
# ==============================================================================
def run_training(args, net_glob, train_data_source, dataset_test, dict_users, logger, use_pre_partition=False):
    
    net_glob.train()
    
    probe_indices = list(range(500)) 
    if len(dataset_test) < 500: probe_indices = list(range(len(dataset_test)))
    probe_set = Subset(dataset_test, probe_indices)
    probe_loader = DataLoader(probe_set, batch_size=100, shuffle=False, num_workers=2)

    suffix = args.suffix if args.suffix else ""
    save_dir = f'./save/{args.algorithm}{suffix}_{args.dataset}_Beta{args.data_beta}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/distributions', exist_ok=True)

    start_round = 0
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
    elif args.resume:
        load_round = args.start_epoch - 1
        load_path = f"{save_dir}/model_round_{load_round}.pth"
        if os.path.exists(load_path):
            log_print(f"🔄 [Auto Resume] Resuming from Round {args.start_epoch}...")
            try:
                state_dict = torch.load(load_path, map_location=args.device)
                net_glob.load_state_dict(state_dict)
                start_round = args.start_epoch
                log_print("✅ Checkpoint loaded successfully!")
            except Exception as e:
                log_print(f"❌ Error loading checkpoint: {e}")
                exit(1)
        else:
            log_print(f"⚠️ Warning: Checkpoint not found. Starting from scratch.")
            start_round = 0

    log_print(f"🔥 开始训练: {args.algorithm} | Rounds: {start_round} -> {args.epochs}")

    # 🟢🟢🟢 【关键修复】初始化 Logger 🟢🟢🟢
    ma_logger = MovingAverageLogger(window_size=5)
    # 🟢🟢🟢 ------------------------ 🟢🟢🟢

    for iter in range(start_round, args.epochs):
        log_print(f'=== Round {iter:3d} ===')
        
        progress = iter / args.epochs 
        decay_factor = 0.5 * (1 + np.cos(np.pi * progress))
        dynamic_reset_ratio = args.reset * decay_factor
        if dynamic_reset_ratio < 0.001: dynamic_reset_ratio = 0.0
        log_print(f"   [Scheduler] Round {iter}: Dynamic Reset Ratio = {dynamic_reset_ratio:.5f}")

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            net_local = copy.deepcopy(net_glob).to(args.device)
            
            if args.algorithm in ['FedPhoenix', 'FedFAAR']:
                reset_kernels_and_neurons_stair(
                    net_local, 
                    iter, 
                    args.FP_conv, 
                    conv_reset_ratio=dynamic_reset_ratio
                )
            
            if use_pre_partition:
                local = LocalUpdate_FedAvg(args=args, dataset=None, idxs=None, pre_dataset=train_data_source[idx])
                data_len = len(train_data_source[idx])
            else:
                local = LocalUpdate_FedAvg(args=args, dataset=train_data_source, idxs=dict_users[idx], pre_dataset=None)
                data_len = len(dict_users[idx])
            
            w, loss = local.train(net=net_local)
            w_locals.append(copy.deepcopy(w))
            lens.append(data_len)
            
        w_glob = Aggregation(w_locals, lens)
        net_glob.load_state_dict(w_glob)

        if args.algorithm == 'FedFAAR':
            reset_offset = 2  
            is_reset_round = (iter % args.faar_reset_period == reset_offset)
            
            if iter >= args.faar_warmup and is_reset_round and dynamic_reset_ratio > 0:
                log_print(f"[FAAR] Diagnosing Bias (Wasserstein) Round {iter}...")
                faar_reset_map = diagnose_bias_by_wasserstein(net_glob, probe_loader, args, top_k_ratio=dynamic_reset_ratio)
                total_to_fix = sum([len(v) for v in faar_reset_map.values()])
                log_print(f"   [FAAR] Identified {total_to_fix} neurons (Ratio: {dynamic_reset_ratio:.4f})")
                if total_to_fix > 0:
                    total_reset = apply_faar_soft_reset(net_glob, faar_reset_map, blend_alpha=0.8)
                    log_print(f"   [FAAR] Soft-Reset Applied. {total_reset} Filters neutralized.")
            elif dynamic_reset_ratio <= 0 and (iter % 5 == 0):
                 log_print("   [FAAR] Reset ratio decayed to 0. Intervention stopped for stability.")
        
        if iter % 5 == 0 or iter == args.epochs - 1: 
            if args.dataset == 'celeba':
                res = test_fairness(net_glob, dataset_test, args)
                
                # 🟢🟢🟢 【修复】现在 ma_logger 已经定义了 🟢🟢🟢
                ma_logger.update(res)
                avg_res = ma_logger.get_averages()
                
                log_print(f"[TEST] Round {iter}")
                log_print(f"   Acc     : {res['acc']:.2f}% ({avg_res['acc']:.2f}%)")
                log_print(f"   Loss    : {res['loss']:.4f} ({avg_res['loss']:.4f})")
                log_print(f"   Gap     : {res['gap']:.2f}% ({avg_res['gap']:.2f}%)")
                log_print(f"   Group   : F={res['acc_f']:.1f}% M={res['acc_m']:.1f}% (Worst: {res['acc_worst']:.1f}%)")
                log_print(f"   🔥Hard  : Male+Smile={res['acc_male_smile']:.1f}% | Female+NoSmile={res['acc_female_nosmile']:.1f}%")
                
                if logger:
                    logger.log_metric({
                        "test/acc": res['acc'],
                        "test/loss": res['loss'],
                        "test/acc_avg5": avg_res['acc'], 
                        "fairness/gap": res['gap'],
                        "fairness/acc_female": res['acc_f'],
                        "fairness/acc_male": res['acc_m'],
                        "fairness/acc_male_smile": res['acc_male_smile'],
                        "fairness/acc_worst": res['acc_worst'],
                        "round": iter
                    }, epoch=iter)
            else:
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
                log_print(f"[TEST] Acc: {acc_test:.2f}% | Loss: {loss_test:.2f}")
                if logger:
                    logger.log_train(loss_test, acc_test, epoch=iter)

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
    
    if torch.cuda.is_available() and args.gpu != -1:
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')

    set_random_seed(args.seed)

    # 1. 加载测试集 (保持与 Raw CelebA 一致以公平对比)
    _, dataset_test, _ = get_celeba_dataset_raw(args)
    num_classes = 2 

    # 2. 加载训练数据 (判断模式)
    use_pre_partition = False
    client_datasets = {} 
    dict_users = None    
    dataset_train = None 
    
    if args.pre_partitioned_path != "":
        log_print(f"📂 Loading pre-partitioned data from: {args.pre_partitioned_path}")
        
        try:
            raw_data = torch.load(args.pre_partitioned_path)
            
            count_users = 0
            for cid_str, c_data in raw_data.items():
                cid = int(cid_str)
                c_dataset = TensorDataset(c_data['x'], c_data['y'], c_data['s'])
                client_datasets[cid] = c_dataset
                count_users += 1
            
            args.num_users = count_users 
            log_print(f"✅ Loaded {count_users} clients from .pt file.")

            inspect_data_distribution(client_datasets)

            use_pre_partition = True
            
        except Exception as e:
            log_print(f"❌ Error loading .pt file: {e}")
            sys.exit(1)
            
    else:
        log_print("⚡ Using Standard Raw Dataset + Dirichlet Partition")
        use_pre_partition = False
        dataset_train, _, dict_users = get_celeba_dataset_raw(args)
        
        if args.noniid_case == 99: 
            dict_users = partition_attribute_dirichlet(
                dataset_train, 
                args.num_users, 
                beta=args.data_beta
            )

    # 3. 初始化模型
    if args.model == 'resnet18':
        net_glob = ResNet18_cifar10(num_classes=num_classes)
    elif args.model == 'simplecnn':
        net_glob = SimpleCNN(num_classes=num_classes)
    else:
        net_glob = ResNet18_cifar10(num_classes=num_classes)
        
    net_glob.to(args.device)
    log_print(f"Model initialized: {args.model} (Classes: {num_classes})")

    # 4. 初始化 Wandb
    suffix = args.suffix if args.suffix else ""
    run_name = f"{args.algorithm}{suffix}_{args.dataset}_Beta{args.data_beta}"
    logger = WandbLogger(args=args, name=run_name, active=args.use_wandb)

    # 5. 开始训练
    run_training(
        args, 
        net_glob, 
        client_datasets if use_pre_partition else dataset_train, 
        dataset_test, 
        None if use_pre_partition else dict_users, 
        logger,
        use_pre_partition=use_pre_partition
    )