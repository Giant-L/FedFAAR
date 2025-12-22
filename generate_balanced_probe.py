import torch
import pandas as pd
import os
import random
from torchvision import transforms
from PIL import Image
from torch.utils.data import TensorDataset

# ================= 配置区域 =================
# 请确保这里指向你的 CelebA 图片目录和属性文件
IMAGE_DIR = '/home/ubuntu/ljy/AAAFAAR/data/celeba/img_align_celeba'
ATTR_FILE = '/home/ubuntu/ljy/AAAFAAR/data/celeba/list_attr_celeba.csv'
OUTPUT_PROBE_PATH = '/home/ubuntu/ljy/AAAFAAR/probe_balanced_2000.pt'

# 目标数量
TARGET_COUNT = 500  # 每个子组 500 张，共 2000 张

def generate_probe():
    print(f"📂 Loading attributes from {ATTR_FILE}...")
    df = pd.read_csv(ATTR_FILE)
    
    # 将 -1 替换为 0，方便处理
    df.replace(-1, 0, inplace=True)
    
    # 定义四个子组的筛选条件
    # Male=1, Smiling=1
    mask_m_s = (df['Male'] == 1) & (df['Smiling'] == 1)
    # Male=1, Smiling=0
    mask_m_ns = (df['Male'] == 1) & (df['Smiling'] == 0)
    # Male=0, Smiling=1
    mask_f_s = (df['Male'] == 0) & (df['Smiling'] == 1)
    # Male=0, Smiling=0
    mask_f_ns = (df['Male'] == 0) & (df['Smiling'] == 0)
    
    # 获取文件名列表
    pool_m_s = df[mask_m_s]['image_id'].tolist()
    pool_m_ns = df[mask_m_ns]['image_id'].tolist()
    pool_f_s = df[mask_f_s]['image_id'].tolist()
    pool_f_ns = df[mask_f_ns]['image_id'].tolist()
    
    print(f"📊 Available samples: M_S={len(pool_m_s)}, M_NS={len(pool_m_ns)}, F_S={len(pool_f_s)}, F_NS={len(pool_f_ns)}")
    
    # 随机采样
    random.seed(42) # 固定种子保证可复现
    selected_m_s = random.sample(pool_m_s, TARGET_COUNT)
    selected_m_ns = random.sample(pool_m_ns, TARGET_COUNT)
    selected_f_s = random.sample(pool_f_s, TARGET_COUNT)
    selected_f_ns = random.sample(pool_f_ns, TARGET_COUNT)
    
    all_files = selected_m_s + selected_m_ns + selected_f_s + selected_f_ns
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    data_x = []
    data_y = [] # Target: Smiling
    data_s = [] # Sensitive: Male
    
    print("🚀 Processing images...")
    for fname in all_files:
        img_path = os.path.join(IMAGE_DIR, fname)
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            data_x.append(tensor)
            
            # 获取标签
            row = df[df['image_id'] == fname].iloc[0]
            data_y.append(int(row['Smiling']))
            data_s.append(int(row['Male']))
        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")

    # 转换为 Tensor
    x_tensor = torch.stack(data_x)
    y_tensor = torch.tensor(data_y, dtype=torch.long)
    s_tensor = torch.tensor(data_s, dtype=torch.long)
    
    # 保存
    torch.save({'x': x_tensor, 'y': y_tensor, 's': s_tensor}, OUTPUT_PROBE_PATH)
    print(f"✅ Probe dataset saved to {OUTPUT_PROBE_PATH}")
    print(f"   Shape: {x_tensor.shape}")
    print("   Balance Check:")
    print(f"   Male: {(s_tensor==1).sum().item()}, Female: {(s_tensor==0).sum().item()}")
    print(f"   Smile: {(y_tensor==1).sum().item()}, NoSmile: {(y_tensor==0).sum().item()}")

if __name__ == '__main__':
    generate_probe()