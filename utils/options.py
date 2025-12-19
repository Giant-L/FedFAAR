#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # ============================================================
    # 1. Federated Learning Settings
    # ============================================================
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0000, help="weight decay")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # ============================================================
    # 2. Model & Dataset
    # ============================================================
    parser.add_argument('--algorithm', type=str, default='FedAvg', help="algorithm name")
    parser.add_argument('--limit_time', type=float, default=240, help="limit_time")
    parser.add_argument('--model', type=str, default='mlp', help="model name")
    parser.add_argument('--kernel_num', type=int, default=9, help="number of each kind of kernel")
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help="comma-separated kernel size to use for convolution")
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--num_workers', type=int, default=4, help="dataloader workers")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--use_project_head', type=int, default=0, help="whether use projection head")
    parser.add_argument('--out_dim', type=int, default=256, help="the output dimension for the projection head")

    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--generate_data', type=int, default=1, help="whether generate new data")
    parser.add_argument('--iid', type=int, default=1, help="whether i.i.d or not")
    parser.add_argument('--noniid_case', type=int, default=0, help="non-iid case index")
    parser.add_argument('--data_beta', type=float, default=0.5, help="beta for non-iid dirichlet distribution")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help="rounds of early stopping")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # ============================================================
    # 3. Algorithm Specific
    # ============================================================
    parser.add_argument('--prox_alpha', type=float, default=0.01, help='The hyperparameters for FedProx')
    parser.add_argument('--ensemble_alpha', type=float, default=0.9, help='The hyperparameters for FedMix')
    parser.add_argument('--temperature', type=float, default=0.9, help='the temperature parameter for contrastive loss')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sim_type', type=str, default='L1', help='Cluster Sampling: cosine or L1 or L2')
    parser.add_argument('--p', type=float, default=0.0, help='power for propotional selection in clustered sampling')
    parser.add_argument('--trans_beta', type=float, default=0.0, help='beta for FedMut')
    parser.add_argument('--l2', type=float, default=0.0, help='l2 for FedMut')
    parser.add_argument('--generative_alpha', type=float, default=10.0, help='gamma for FedGen')
    parser.add_argument('--generative_beta', type=float, default=10.0, help='beta for FedGen')
    parser.add_argument('--radius', type=float, default=0.95, help='Radius for FedMut')
    parser.add_argument('--min_radius', type=float, default=0.01, help='Min Radius for FedMut')
    parser.add_argument('--mut_acc_rate', type=float, default=0.5, help='Acc Rate for FedMut')
    parser.add_argument('--mut_bound', type=int, default=0, help='Bound for FedMut')
    parser.add_argument('--finetuning', action='store_true', help='finetuning')
    parser.add_argument('--helf_sparse', action='store_true', help='helf_sparse')
    parser.add_argument('--aggmut', action='store_true', help='aggmut')

    # ============================================================
    # 4. FedPhoenix Parameters
    # ============================================================
    parser.add_argument('--FP_conv', type=int, default=1000, help="FedPhoenix: conv layer param")
    parser.add_argument('--FP_fc', type=int, default=0, help="FedPhoenix: fc layer param")
    parser.add_argument('--reset', type=float, default=0.1, help="FedPhoenix/FAAR: Reset ratio (e.g. 0.05)")
    parser.add_argument('--remethod', type=str, default='stair', help="FedPhoenix reset method")

    # ============================================================
    # 5. FedFAAR Parameters
    # ============================================================
    parser.add_argument('--faar_reset_period', type=int, default=20, help="Frequency of FAAR diagnosis and reset")
    parser.add_argument('--faar_tau', type=float, default=0.1, help="Softmax temperature for FAAR")
    parser.add_argument('--faar_warmup', type=int, default=20, help="Warmup rounds before first intervention")

    # ============================================================
    # 6. WandB & Logging & Save
    # ============================================================
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB for logging')
    parser.add_argument('--wandb_project', type=str, default='FedFairness', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--suffix', type=str, default='', help="Suffix for save directory (e.g. '_v2')")

    # ============================================================
    # 7. Checkpoint & Resume (Updated)
    # ============================================================
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
    parser.add_argument('--start_epoch', type=int, default=0, help="Start epoch for resuming (e.g., 21)")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Specific path to load checkpoint")
    
    # 🟢 [新增] 预划分数据路径
    parser.add_argument('--pre_partitioned_path', type=str, default="", help='path to the .pt file with pre-partitioned data')

    parser.add_argument('--wandb_tags', type=str, default="", help="Comma-separated list of WandB tags")
    
    args = parser.parse_args()
    return args