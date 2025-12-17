import subprocess
import itertools
import os
import time
import sys

"""
================================================================
🛠️ FedFAAR 从零启动脚本 (Fresh Start)
================================================================
"""

# 🟢 【指定算法】只跑 FedFAAR
algorithms = ['FedPhoenix']

# 🟢 【指定 GPU】
gpus = [0] 

# --------------------------------------------------------------
# 实验参数 (Attribute Dirichlet V3)
# --------------------------------------------------------------
datasets = ['celeba']
dataset_classes = {'celeba': 2}
models = ['resnet18']

# 🟢 【训练总轮数】 从 0 跑到 300 (或800，你自己定)
EPOCHS = 300       

# 核心超参
lrs = [0.01]
data_betas = [0.3]         # Attribute Dirichlet 参数
noniid_cases = [99]        # 99 = 触发属性偏斜
local_bs_list = [64]

# 算法参数
fp_convs = [1000]
resets = [0.05]
fp_fcs = [0]

# 🟢 【FAAR 参数】
faar_tau_list = [0.1]
# 如果从零跑，建议预热不要设太大，比如 20 或 50 轮后开始介入
# 如果你设为 100，那前 100 轮它就是纯 FedPhoenix，101轮才开始 FAAR
faar_warmups = [20]       
faar_reset_periods = [5]   # 5n+2 策略

# 联邦参数
num_users = [100]
frac = [0.1]
weight_decays = [0.0000]
num_workers_list = [2]

# ==============================================================================
#  👇 启动逻辑 (无 Checkpoint，无等待)
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
        suffix_name = f"_AttrDirichlet_Beta{data_beta}_FreshStart"
        
        cmd = [
            'python', '-u', 'main_fed.py',
            '--gpu', '0',
            '--dataset', dataset,
            '--algorithm', algorithm,
            '--model', model,
            '--lr', str(lr),
            '--epochs', str(EPOCHS),  # 直接跑满
            '--iid', '0',
            '--noniid_case', str(noniid_case),
            '--data_beta', str(data_beta),
            '--num_classes', str(num_classes),
            '--num_users', str(num_user),
            '--frac', str(frac_val),
            '--local_bs', str(local_bs),
            '--num_workers', str(num_workers),
            '--weight_decay', str(weight_decay),
            
            # 核心参数
            '--FP_conv', str(fp_conv),
            '--FP_fc', str(fp_fc),
            '--reset', str(reset),
            '--faar_tau', str(faar_tau),
            '--faar_warmup', str(faar_warmup),
            '--faar_reset_period', str(faar_period),
            
            '--suffix', suffix_name,
            
            '--use_wandb',
            '--wandb_project', 'FedFairness_Fresh_Run',
            '--wandb_tags', f"Fresh,{algorithm},AttrDirichlet",
        ]

        # 执行命令
        cmd_str = " ".join(cmd)
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        log_dir = "logs_fresh"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{algorithm}_gpu{gpu_id}_{int(time.time())}.log")
        
        print(f"🚀 Starting {algorithm} from SCRATCH (Round 0 -> {EPOCHS})")
        print(f"   GPU: {gpu_id}")
        print(f"   Log: {log_file}")
        
        subprocess.run(f"{cmd_str} > {log_file} 2>&1", shell=True, env=env)

if __name__ == "__main__":
    run_task()