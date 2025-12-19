#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from optimizer.Adabelief import AdaBelief
import numpy as np
import random
import torch.nn.functional as F

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self): 
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate_FedAvg(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False, dataset_test=None, pre_dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.verbose = verbose
        
        loader_args = {
            'batch_size': self.args.local_bs,
            'shuffle': True,
            'num_workers': self.args.num_workers
        }
        # Windows/某些环境可能不支持 pin_memory，可视情况关闭
        if self.args.num_workers > 0:
            loader_args['pin_memory'] = True

        # 🟢 [修改] 支持直接传入预划分的 dataset (TensorDataset)
        if pre_dataset is not None:
            self.ldr_train = DataLoader(pre_dataset, **loader_args)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), **loader_args)

    def train(self, net):
        net.train()
        # optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            
            # 🟢 [修改] 训练循环适配不同的返回值
            for batch_idx, batch_data in enumerate(self.ldr_train):
                # 如果是 .pt 数据，batch_data 会包含 (images, labels, sensitives)
                if len(batch_data) == 3:
                    images, labels, sensitives = batch_data
                    # 对于训练，我们暂时只需要 images 和 labels
                elif len(batch_data) == 2:
                    images, labels = batch_data
                else:
                    raise ValueError(f"Unexpected data format: len={len(batch_data)}")

                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # ============================================================
                # 【核心修复】 处理 CelebA 的多列标签 [Batch, 2]
                # ============================================================
                # 只有当 labels 维度大于 1 且不是 1D Tensor 时才切片
                # .pt 文件传进来的 labels 通常已经是 1D 的 Smiling 标签
                if labels.dim() > 1 and labels.size(1) >= 2:
                    target_labels = labels[:, 0] # 取第一列：Smiling
                else:
                    target_labels = labels

                # 确保是 1D LongTensor [N]
                target_labels = target_labels.view(-1).long()
                # ============================================================

                net.zero_grad()
                output = net(images)
                
                # 兼容字典输出
                if isinstance(output, dict):
                    log_probs = output['output']
                else:
                    log_probs = output
                
                loss = self.loss_func(log_probs, target_labels)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

# FedProx逻辑类似，也加上切片处理
class LocalUpdate_FedProx(object):
    def __init__(self, args, glob_model, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.glob_model = glob_model
        self.prox_alpha = args.prox_alpha
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=self.args.num_workers)

    def train(self, net):
        net.train()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        global_weight_collector = list(self.glob_model.parameters())

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # 【核心修复】
                if labels.dim() > 1 and labels.size(1) >= 2:
                    target_labels = labels[:, 0]
                else:
                    target_labels = target_labels = labels
                target_labels = target_labels.view(-1).long()

                net.zero_grad()
                output = net(images)
                if isinstance(output, dict): output = output['output']

                # Proximal term
                proximal_term = 0.0
                for param, global_param in zip(net.parameters(), global_weight_collector):
                    proximal_term += ((self.prox_alpha / 2) * torch.norm((param - global_param)) ** 2)

                loss = self.loss_func(output, target_labels) + proximal_term
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)