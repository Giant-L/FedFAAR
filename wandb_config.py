import time
import os
import sys

class WandbLogger:
    """
    Wandb日志记录器封装类
    """
    def __init__(self, args=None, project=None, entity=None, name=None, config=None, active=True):
        self.active = active
        self.run = None
        self.start_time = None
        
        # 优先级：函数参数 active > args.use_wandb
        if args is not None and not active:
             # 如果外部没强制开启，则检查 args
            self.active = getattr(args, 'use_wandb', False)

        if not self.active:
            print("[WandbLogger] Wandb 未启用 (Active=False)")
            return

        try:
            import wandb
            # 如果已经有 run 在运行（防止多进程冲突），先 finish
            if wandb.run is not None:
                wandb.finish()

            # 确保 config 是字典
            if config is None and args is not None:
                config = vars(args)
            elif config is None:
                config = {}

            # 初始化 wandb
            self.run = wandb.init(
                project=project if project else "FedFAAR_Experiment",
                entity=entity,
                name=name,
                config=config,
                reinit=True, # 允许重新初始化
                settings=wandb.Settings(start_method="fork") # 修复多进程下的某些问题
            )
            self.start_time = time.time()
            
            # 定义常用指标
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch") 
            wandb.define_metric("valid/*", step_metric="epoch")
            wandb.define_metric("fairness/*", step_metric="epoch")

            print(f"🚀 Wandb 初始化成功！Run: {self.run.name}")
            print(f"🔗 链接: {self.run.get_url()}")
            
        except ImportError:
            print("❌ 错误: 未安装 wandb，请运行 pip install wandb")
            self.active = False
        except Exception as e:
            print(f"❌ Wandb 初始化失败: {e}")
            self.active = False

    def log_metric(self, metrics, epoch=None):
        """通用的 log 函数"""
        if not self.active or self.run is None: return
        
        log_dict = metrics.copy()
        if epoch is not None:
            log_dict['epoch'] = epoch
        
        # 自动计算运行时间
        if self.start_time:
            log_dict['runtime_hours'] = (time.time() - self.start_time) / 3600
            
        import wandb
        wandb.log(log_dict)

    def finish(self):
        if self.active and self.run:
            import wandb
            wandb.finish()
            print("[WandbLogger] 运行结束。")