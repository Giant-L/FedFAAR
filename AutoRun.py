import subprocess
import itertools
import os
import time
import sys

"""
================================================================
🛠️ FedFAAR 启动脚本 (针对 20 Client Biased 数据集)
================================================================
"""

# 🟢 【指定算法】
algorithms = [
              'FedAvg',   # 建议先跑 FedAvg 作为 Baseline，看看 Gap 有多大
            #   'FedFAAR',  
            # 'FedPhoenix'

            ]

# 🟢 【指定 GPU】
gpus = [4] 

# --------------------------------------------------------------
# 实验参数
# --------------------------------------------------------------
datasets = ['celeba']
dataset_classes = {'celeba': 2}

# 🟢 1. 修改模型名称 (对应 main_fed.py 里的判定字符串)
models = [
    'simplecnn'
    # 'resnet18'
] 

# 训练总轮数 (SimpleCNN 收敛快，200轮足够)
EPOCHS = 200       

# 核心超参
lrs = [0.01]
data_betas = [0.5]         # 这个参数在 pre_partitioned 模式下其实没用了，但保留防止报错
noniid_cases = [100]       # 🟢 使用 100 作为标记，虽然 .pt 模式下主要靠路径
local_bs_list = [64]

# 🟢 2. 这里的 num_users 必须是 20，对应你的 .pt 文件结构
num_users = [20]
# 🟢 3. 参与率：0.2 表示每轮选 4 个 client (20 * 0.2)
frac = [0.2] 

weight_decays = [0.0000]
num_workers_list = [2]

# 算法参数
# 🟢 4. SimpleCNN 参数量小，FP_conv 不要设太大 (比如 200 或 500)
fp_convs = [200]  
resets = [0.05]
fp_fcs = [0]

# FAAR 参数
faar_tau_list = [0.1]
faar_warmups = [20]       
faar_reset_periods = [5]

# 🟢 5. 指定你的 .pt 文件路径
# 这是200 800 * 1-4； 800 200 * 5-20
PT_PATH = "/home/ubuntu/ljy/AAAFAAR/data/celeba/celeba_biased_20clients_1000img.pt"

# 这是废物数据集，训练出来的model只会瞎猜
# 这是20 80 * 1-4； 80 20 * 5-20
# PT_PATH = "/home/ubuntu/ljy/AAAFAAR/data/celeba/celeba_biased_20clients.pt"


# ==============================================================================
#  👇 启动逻辑
# ==============================================================================

def run_task():
    combinations = list(itertools.product(
        datasets, algorithms, lrs, models, data_betas, 
        fp_convs, fp_fcs, resets, num_users, frac, 
        weight_decays, faar_tau_list, faar_warmups, faar_reset_periods, 
        local_bs_list, num_workers_list, noniid_cases
    ))

    for params in combinations:
        (
            dataset, algorithm, lr, model, data_beta, 
            fp_conv, fp_fc, reset, num_user, frac_val, 
            weight_decay, faar_tau, faar_warmup, faar_period, 
            local_bs, num_workers, noniid_case
        ) = params

        num_classes = dataset_classes[dataset]
        gpu_id = gpus[0]
        
        # 命名后缀
        suffix_name = f"_SimpleCNN_Biased20"
        
        cmd = [
            'python', '-u', 'main_fed.py',
            '--gpu', '0',
            '--dataset', dataset,
            '--algorithm', algorithm,
            '--model', model,
            '--lr', str(lr),
            '--epochs', str(EPOCHS),
            '--num_classes', str(num_classes),
            '--local_bs', str(local_bs),
            '--num_workers', str(num_workers),
            '--weight_decay', str(weight_decay),
            
            # 🟢 关键参数修改
            '--num_users', '20',       # 强制 20
            '--frac', str(frac_val),   # 0.2
            '--pre_partitioned_path', PT_PATH, # 传入 .pt 路径
            
            # 核心参数
            '--FP_conv', str(fp_conv),
            '--FP_fc', str(fp_fc),
            '--reset', str(reset),
            '--faar_tau', str(faar_tau),
            '--faar_warmup', str(faar_warmup),
            '--faar_reset_period', str(faar_period),
            
            '--suffix', suffix_name,
            
            '--use_wandb',
            '--wandb_project', 'FedFairness_SimpleCNN',
            '--wandb_tags', f"BiasedData,{algorithm},{model}",
        ]

        # 执行命令
        cmd_str = " ".join(cmd)
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        log_dir = "logs_simplecnn"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{algorithm}_{model}_gpu{gpu_id}_{int(time.time())}.log")
        
        print(f"🚀 Starting {algorithm} with {model} on 20 Biased Clients")
        print(f"   Data: {PT_PATH}")
        print(f"   Log: {log_file}")
        
        subprocess.run(f"{cmd_str} > {log_file} 2>&1", shell=True, env=env)

if __name__ == "__main__":
    run_task()