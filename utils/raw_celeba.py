import os
import torch
import time
from torch.utils.data import Dataset
from torchvision import transforms

class CelebA_Fast(Dataset):
    """
    【极速版】直接读取预处理好的 .pt 文件
    无需 CSV 解析，无需 JPG 解码，无需磁盘寻道。
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pt_path = os.path.join(root_dir, 'celeba_fast.pt')
        
        # 1. 加载二进制大文件
        # 为了避免 Train 和 Test 加载两次导致内存爆炸 (2.4GB * 2)，
        # 我们使用一个简单的单例模式或者依赖操作系统的 Page Cache。
        # 这里为了逻辑简单，我们假设内存足够大 (你有100GB内存，随便用)
        
        if not hasattr(CelebA_Fast, 'shared_data'):
            print(f" -> [Fast Loader] 正在从 {self.pt_path} 加载数据...")
            t0 = time.time()
            if not os.path.exists(self.pt_path):
                print("❌ 错误：没找到 celeba_fast.pt！")
                print("   请先运行 'python make_celeba_bin.py' 生成它！")
                exit(1)
                
            CelebA_Fast.shared_data = torch.load(self.pt_path)
            t1 = time.time()
            print(f"✅ 加载完成！耗时: {t1 - t0:.2f} 秒")
        
        # 2. 从共享数据中切分
        data = CelebA_Fast.shared_data
        
        # partition: 0=train, 1=val, 2=test
        target_part = 0 if split == 'train' else 2
        
        # 创建掩码
        mask = (data['partitions'] == target_part)
        
        # 提取当前 split 的数据
        # 注意：这里我们只保存引用，不要 clone()，否则内存翻倍
        self.images = data['images'][mask]
        self.targets = data['targets'][mask]
        
        print(f" -> [{split}] 分配图片: {len(self.images)} 张")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # images 是 uint8 的 [3, 64, 64] Tensor
        img = self.images[idx]
        
        # 转为 float 并归一化 (0-255 -> 0.0-1.0)
        # 这一步非常快，因为是在 Tensor 上操作，没有 PIL 转换开销
        img = img.float() / 255.0
        
        # 手动实现 Normalize ((0.5,0.5,0.5), (0.5,0.5,0.5))
        # (x - mean) / std  => (x - 0.5) / 0.5 => x*2 - 1
        img = (img - 0.5) / 0.5
        
        attr = self.targets[idx]
        return img, attr

def get_celeba_dataset_raw(args):
    print(f"=== Loading CelebA (Binary .pt Mode) ===")
    data_dir = './data/celeba'
    
    # 我们的 Loader 内部已经处理了 ToTensor 和 Normalize
    # 所以这里的 transform 参数其实用不到了，但为了兼容代码接口保留
    transform = None 
    
    train_set = CelebA_Fast(root_dir=data_dir, split='train', transform=transform)
    test_set = CelebA_Fast(root_dir=data_dir, split='test', transform=transform)
    
    num_users = args.num_users
    # 随机分配数据给用户
    import numpy as np
    idxs = np.random.permutation(len(train_set))
    dict_users = np.array_split(idxs, num_users)
    dict_users = {i: dict_users[i] for i in range(num_users)}

    return train_set, test_set, dict_users