import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# 配置路径
ROOT_DIR = './data/celeba'
IMG_DIR = os.path.join(ROOT_DIR, 'img_align_celeba')
ATTR_CSV = os.path.join(ROOT_DIR, 'list_attr_celeba.csv')
EVAL_CSV = os.path.join(ROOT_DIR, 'list_eval_partition.csv')
SAVE_PATH = os.path.join(ROOT_DIR, 'celeba_fast.pt')

def make_binary():
    print("🚀 开始制作 CelebA 极速二进制文件...")
    
    # 1. 读取 CSV 确保顺序一致
    if not os.path.exists(ATTR_CSV) or not os.path.exists(EVAL_CSV):
        print("❌ 找不到 CSV 文件，请检查路径")
        exit(1)
        
    df_attr = pd.read_csv(ATTR_CSV)
    df_eval = pd.read_csv(EVAL_CSV)
    df = pd.merge(df_attr, df_eval, on='image_id')
    
    # 按照 image_id 排序，确保索引绝对对应
    # (Kaggle数据集通常已经是排好的，但保险起见)
    # df = df.sort_values('image_id') 
    
    img_names = df['image_id'].values
    partitions = torch.tensor(df['partition'].values, dtype=torch.long) # 0:train, 1:val, 2:test
    
    # 提取标签: Smiling (31), Male (20)
    # Kaggle 数据通常是 -1 / 1，转为 0 / 1
    labels_raw = df[['Smiling', 'Male']].values
    targets = torch.tensor((labels_raw + 1) // 2, dtype=torch.long)
    
    print(f" -> 待处理图片: {len(img_names)} 张")
    
    # 2. 预分配大 Tensor (N, C, H, W) 
    # 使用 uint8 (0-255) 节省空间，训练时再转 float
    # 202599 * 3 * 64 * 64 bytes ≈ 2.35 GB
    all_images = torch.zeros((len(img_names), 3, 64, 64), dtype=torch.uint8)
    
    print(" -> 开始读取并转换图片 (这是一次性工作，耐心等待)...")
    
    for i, fname in enumerate(tqdm(img_names)):
        fpath = os.path.join(IMG_DIR, fname)
        try:
            with Image.open(fpath) as img:
                img = img.convert('RGB').resize((64, 64))
                # HWC -> CHW
                # np.array 是 (64, 64, 3)，PyTorch 需要 (3, 64, 64)
                np_img = np.array(img).transpose(2, 0, 1)
                all_images[i] = torch.from_numpy(np_img)
        except Exception as e:
            print(f"⚠️ 图片损坏: {fname}, 使用黑图填充")
            # 默认为0，不需要额外操作
            
    # 3. 保存为一个大字典
    data_pack = {
        'images': all_images,    # [N, 3, 64, 64] uint8
        'targets': targets,      # [N, 2] long
        'partitions': partitions # [N] long
    }
    
    print(f" -> 正在保存到硬盘: {SAVE_PATH} ...")
    torch.save(data_pack, SAVE_PATH)
    print("✅ 制作完成！以后每次启动只需 1-2 秒！")

if __name__ == '__main__':
    make_binary()