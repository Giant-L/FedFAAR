import subprocess
import itertools
import os
import time
import sys
import atexit
import signal
import numpy as np
from typing import Dict, List, Optional
import pynvml

# 初始化 NVML
try:
    pynvml.nvmlInit()
except Exception as e:
    print(f"Error initializing NVML: {e}")
    # 如果没有显卡驱动，可以选择退出或继续（取决于是否有CPU fallback）
    # sys.exit(1) 

"""
Experiment Configuration Parameters
Modify this part of the parameters to configure different experiments
"""

#----------------------------------------
# 1. Dataset Related Parameters
#----------------------------------------
datasets = [
    'celeba',   # 修改：使用 CelebA 数据集
    # 'cifar10',  
    #'cifar100',
    #'timage'
]

# Number of classes for each dataset
dataset_classes = {
    'cifar10': 10,
    'cifar100': 20,
    'timage': 200,
    'femnist':62,
    'celeba': 2 # 新增：CelebA 为二分类 (Smiling/Not Smiling)
}

#----------------------------------------
# 2. Algorithm Related Parameters
#----------------------------------------
algorithms = [
    #'FedGen',         
    #'FedMut',          
    # 'FedPhoenix', # 启用
    'FedFAAR',    # 启用
    # 'FedAvg',     # 启用
    #'FedProx',
    #'ClusteredSampling',   
]

#----------------------------------------
# 3. Model Related Parameters
#----------------------------------------
models = [
    'resnet18',   
    #'resnet_drop', 
    #'mobnet'   ,      
    #'vgg'
    #'vggdrop',
]

#----------------------------------------
# 4. Training Related Parameters
#----------------------------------------
lrs = [0.01]           #Learning Rate
epochs = [100]         # 修改：CelebA 较大，演示可设为 100
data_betas = [0.3]     #If you need to set it as IID, please use 0.5. (注意：这里0.5会被处理为IID)
weight_decays = [0.0000] 
local_bs_list = [64] 
num_workers_list = [2]

#----------------------------------------
# 5. Federated Learning Related Parameters
#---------------------------------------- 
num_users = [100]       # Total number of clients
frac = [0.1]          # Fraction of clients participating in each round

#----------------------------------------
# 6. Specific Algorithm Parameters
#----------------------------------------
# FedPhoenix algorithm related parameters
fp_convs = [1000]      # r_s
resets = [0.05]       # Reset rate   θ (建议 0.05)
fp_fcs = [0]         

# === FAAR-FL params ===
# 您可以在这里进行网格搜索
faar_tau_list = [0.1]       # 温度系数 (对应 args.faar_tau)
faar_warmups = [20]         # 预热轮数 (对应 args.faar_warmup)
faar_reset_periods = [10]   # 重置周期 (对应 args.faar_reset_period)
# ======================



#----------------------------------------
# 7. Hardware Related Parameters
#----------------------------------------
# 您之前指定使用 GPU 6，我保留了这个设置，您也可以恢复为列表 [0,1,2...]
gpus = [7]            # List of GPU IDs to use
# If you have multiple GPUs, you can set it to:
# gpus = [0,1,2,3]    # Use GPU 0, 1, 2, 3

#----------------------------------------
# 8. Resource Management Parameters
#----------------------------------------
# GPU usage threshold settings
GPU_MEMORY_THRESHOLD = 0.85  # GPU memory usage threshold (85%)
GPU_UTIL_THRESHOLD = 90   # GPU utilization threshold (90%)

# Task scheduling parameters
TASK_INIT_SLEEP = 30   # Task initialization wait time (seconds) -> 增加等待时间以防冲突
SCHEDULER_SLEEP_BUSY = 10    # Scheduler busy polling interval (seconds)
SCHEDULER_SLEEP_IDLE = 10    # Scheduler idle polling interval (seconds)


# ==============================================================================
#  👇 下面是核心逻辑的修复版本 (请勿随意修改，包含了防崩溃和进程清理机制) 👇
# ==============================================================================

# 全局变量用于存储运行中的进程，以便清理
global_running_procs = []

def cleanup_at_exit():
    """程序退出时的清理函数，杀死所有子进程"""
    if global_running_procs:
        print("\n[AutoRun] Exiting... Cleaning up child processes.")
        for p in global_running_procs:
            if p.poll() is None:  # 如果进程还在运行
                try:
                    # 使用进程组杀手，确保 Shell 模式下的子进程也被清理
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception as e:
                    print(f"Error killing process: {e}")
        print("[AutoRun] Cleanup complete.")

# 注册清理函数
atexit.register(cleanup_at_exit)

def signal_handler(sig, frame):
    """处理 Ctrl+C"""
    print("\n[AutoRun] Ctrl+C detected.")
    sys.exit(0) # 这会触发 atexit

signal.signal(signal.SIGINT, signal_handler)

#----------------------------------------
# Parameter Validation
#----------------------------------------
def validate_params():
    """Validate the legality of the parameter configuration"""
    assert all(d in dataset_classes for d in datasets), "Invalid dataset name"
    # assert all(m in ['resnet18', 'mobnet'] for m in models), "Invalid model name"
    assert all(0 < lr <= 1 for lr in lrs), "Learning rate must be in the range (0,1]"
    assert all(e > 0 for e in epochs), "Training epochs must be positive"
    assert all(0 < f <= 1 for f in frac), "Client sampling fraction must be in the range (0,1]"
    
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        # assert all(g < device_count for g in gpus), f"GPU ID out of range, the system has {device_count} GPUs"
    except Exception as e:
        print(f"GPU check failed or skipped: {e}")
        return True # Skip strict check if nvml fails partially
    
    return True

if not validate_params():
    raise ValueError("Parameter validation failed")

#----------------------------------------
# Generate Parameter Combinations
#----------------------------------------
variable_combinations = list(itertools.product(
    datasets, algorithms, lrs, epochs, models, 
    data_betas, fp_convs, fp_fcs, resets, 
    num_users, frac, weight_decays,
    
    # === FAAR 新参数 (这里改了！) ===
    faar_tau_list,      # 对应 faar_tau
    faar_warmups,       # 对应 faar_warmup
    faar_reset_periods, # 对应 faar_reset_period

    local_bs_list,      # Batch Size
    num_workers_list    # Worker 数量
))

# Dynamically generate fixed_params
fixed_params = []
for params in variable_combinations:
    data_beta = params[5]
    # === 关键修正：确保 beta=0.5 时逻辑一致，避免文件名错乱 ===
    if abs(data_beta - 0.5) > 1e-6:
        fixed_params.append({'iid': 0, 'noniid_case': 5})
    else:
        fixed_params.append({'iid': 1, 'noniid_case': 0})

print(f"A total of {len(variable_combinations)} parameter combinations were generated")

class GPUManager:
    """GPU Manager: Responsible for monitoring and managing the status and task allocation of multiple GPUs"""
    
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.gpu_handles = {gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in gpu_ids}
        
    def get_gpu_stats(self, gpu_id: int) -> dict:
        handle = self.gpu_handles[gpu_id]
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            'memory_used': info.used / info.total,
            'gpu_utilization': utilization.gpu,
            'memory_total': info.total
        }

    def is_gpu_available(self, gpu_id: int, memory_threshold=GPU_MEMORY_THRESHOLD, util_threshold=GPU_UTIL_THRESHOLD) -> bool:
        try:
            stats = self.get_gpu_stats(gpu_id)
            return (stats['memory_used'] < memory_threshold and 
                   stats['gpu_utilization'] < util_threshold)
        except:
            return False

    def get_best_gpu(self) -> Optional[int]:
        gpu_loads = {}
        for gpu_id in self.gpu_ids:
            if self.is_gpu_available(gpu_id):
                stats = self.get_gpu_stats(gpu_id)
                # Calculate load score: 0.7 weight for memory, 0.3 for utilization
                load_score = stats['memory_used'] * 0.7 + stats['gpu_utilization'] / 100 * 0.3
                gpu_loads[gpu_id] = load_score
        
        return min(gpu_loads.items(), key=lambda x: x[1])[0] if gpu_loads else None

def run_training(params, gpu_id, iid, noniid_case):
    (
        dataset, algorithm, lr, epoch, model, data_beta, 
        fp_conv, fp_fc, reset, num_user, frac_val, weight_decay,
        faar_tau, faar_warmup, faar_period, 
        local_bs, num_workers
    ) = params

    num_classes = dataset_classes[dataset]

    # [新增] 强制设置环境变量，确保物理隔离
    # 这样即使 main_fed.py 默认用 cuda:0，它实际上用的也是我们分配的 gpu_id
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. 构造命令
    cmd_list = [
        'python', '-u', 'main_fed.py',
        '--dataset', dataset,
        '--algorithm', algorithm,
        '--lr', str(lr),
        '--epochs', str(epoch), 
        '--model', model,
        '--iid', str(iid),
        '--noniid_case', str(noniid_case),
        '--data_beta', str(data_beta),
        '--generate_data', '1',
        
        # [修改] 欺骗子进程，告诉它用 0 号卡
        # 因为我们已经用 env 隔离了显卡，子进程看到的 0 号卡就是物理上的 gpu_id
        '--gpu', '0', 
        
        '--num_classes', str(num_classes),
        '--FP_conv', str(fp_conv),
        '--FP_fc', str(fp_fc),
        '--reset', str(reset),
        '--num_users', str(num_user),
        '--frac', str(frac_val),
        '--weight_decay', str(weight_decay),

        # === 核心优化参数 ===
        '--local_bs', str(local_bs),
        '--num_workers', str(num_workers),
        '--use_wandb',           # <--- [FIX] 开启 WandB
        '--wandb_project', 'FedFairness_CelebA', 
        
        # === FAAR 参数 ===
        '--reset', str(reset),
        '--faar_tau', str(faar_tau),
        '--faar_warmup', str(faar_warmup),
        '--faar_reset_period', str(faar_period),

        # === 【核心修改：指定新名字】 ===
        '--suffix', '_PhoenixStart', 

        # ==========================================
        # ★★★ 核心：断点续训参数 ★★★
        # ==========================================
        '--resume',             
        '--start_epoch', '21',  # <--- [FIX] 必须是 21！才能读到 model_round_20.pth
    ]
    
    full_cmd_str = " ".join(cmd_list)
    
    # 日志文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    run_info = f"{algorithm}{'_PhoenixStart'}_{dataset}"
    log_file = os.path.join(log_dir, f"{run_info}_gpu{gpu_id}_{timestamp}.log")

    shell_cmd = f"{full_cmd_str} > {log_file} 2>&1"

    print(f"🚀 Starting task on GPU {gpu_id} (Physical ID)")
    print(f"   Params: Alg={algorithm}, Suffix=_PhoenixStart, Resume=True, Start=21")
    print(f"   Logs -> {log_file}")
    
    try:
        # [修改] 传入 env 参数
        process = subprocess.Popen(shell_cmd, shell=True, executable='/bin/bash', preexec_fn=os.setsid, env=env)
        return process
    except Exception as e:
        print(f"❌ Failed to start process: {e}")
        return None

def main():
    gpu_manager = GPUManager(gpus)
    tasks = list(zip(variable_combinations, fixed_params))
    local_running_tasks = []
    
    while tasks or local_running_tasks:
        # 1. 清理
        active_tasks = []
        for gpu, p in local_running_tasks:
            if p.poll() is None:
                active_tasks.append((gpu, p))
            else:
                if p in global_running_procs:
                    global_running_procs.remove(p)
                print(f"✅ Task on GPU {gpu} finished (Exit code: {p.returncode})")
                if p.returncode != 0:
                    print(f"⚠️ Task failed/exited on GPU {gpu}. Check logs!")
        local_running_tasks = active_tasks
        
        # 2. 调度
        while tasks:
            best_gpu = gpu_manager.get_best_gpu()
            gpu_busy = any(t_gpu == best_gpu for t_gpu, _ in local_running_tasks)

            if best_gpu is not None and not gpu_busy:
                params, fixed_param = tasks.pop(0)
                process = run_training(params, best_gpu, fixed_param['iid'], fixed_param['noniid_case'])
                
                if process is not None:
                    local_running_tasks.append((best_gpu, process))
                    global_running_procs.append(process)
                    time.sleep(TASK_INIT_SLEEP)
                else:
                    print(f"⚠️ Task failed to launch, skipping.")
            else:
                if not local_running_tasks and not tasks:
                    pass
                elif best_gpu is None:
                    print(f"   [Scheduler] No GPU available (High load). Waiting...")
                elif gpu_busy:
                    print(f"   [Scheduler] Best GPU {best_gpu} is busy. Waiting...")
                break 
        
        sleep_time = SCHEDULER_SLEEP_BUSY if local_running_tasks else SCHEDULER_SLEEP_IDLE
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()