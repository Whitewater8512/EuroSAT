import time
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import rasterio
import numpy as np
from config import *

class EuroSATDataset(Dataset):
    def __init__(self, root_dir=DATA_DIR, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root_dir, target_class)
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    if fname.endswith('.tif'):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[target_class])
                        samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with rasterio.open(path) as src:
            img = src.read()  # 读取所有13个波段
            img = np.transpose(img, (1, 2, 0))  # 转换为HWC格式
            img = img.astype(np.float32)  # 转换为float32类型
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

# def cal_dataset_stat(dataset):
#     all_means = np.zeros(13)
#     all_stds = np.zeros(13)
    
#     # 修改：处理dataset返回的(img, target)元组
#     for img, _ in tqdm(dataset, desc="计算统计值"):
#         # 将 torch.Tensor 转换为 numpy 数组
#         img = img.permute(1, 2, 0).cpu().numpy()
#         for i in range(13):
#             all_means[i] += np.mean(img[:, :, i])
#             all_stds[i] += np.std(img[:, :, i])
    
#     # 计算平均值
#     all_means /= len(dataset)
#     all_stds /= len(dataset)
    
#     return all_means.tolist(), all_stds.tolist()

# # 创建不包含标准化的临时transform
# temp_transform = transforms.Compose([
#     # 将HWC格式的numpy数组转换为CHW格式的Tensor
#     transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32).permute(2, 0, 1))
# ])

# # 创建数据集实例用于计算统计值
# dataset_for_stats = EuroSATDataset(
#     root_dir='e:/遥感/EuroSAT/EuroSAT_MS', 
#     transform=temp_transform
# )

# # 计算统计值
# print("正在计算数据集统计值...")
# means, stds = cal_dataset_stat(dataset_for_stats)
# print(f"计算得到的均值: {means}")
# print(f"计算得到的标准差: {stds}")

# 创建包含标准化的最终transform
transform = transforms.Compose([
    # 将HWC格式的numpy数组转换为CHW格式的Tensor
    # transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[1353.7269257269966, 1117.2022923538773, 1041.8847248444733, 946.5542548737702, 1199.1886644965277, 2003.0067999222367, 2374.008444724754, 2301.2204385669847, 732.1819500777633, 12.099527624059606, 1820.6963775318286, 1118.2027229275175, 2599.782937237775],  # 使用计算得到的均值
        std=[65.28860615528623, 153.75498622232013, 187.67639913129807, 278.09068379446313, 227.89627158266967, 355.88970550351894, 455.0773387908847, 530.7147651184047, 98.91790513907318, 1.1872385047365117, 378.1152245224979, 303.0695108921616, 502.1024616939845]     # 使用计算得到的标准差
    )
])

# 创建最终数据集
st = time.time()
print("开始加载数据...")
dataset = EuroSATDataset(root_dir=DATA_DIR, transform=transform)
print(f"数据加载完成, 耗时: {time.time() - st} 秒")

st = time.time()
print("开始准备数据...")
# 7:3的比例划分训练集和测试集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(f"数据准备完成, 耗时: {time.time() - st} 秒")