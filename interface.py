import torch
import numpy as np
import rasterio
import os
from model import ResNet18
from torchvision import transforms
from config import *

# 加载训练好的模型
model = ResNet18()
model.load_state_dict(torch.load(os.path.join(WEIGHT_DIR, 'model.pth')))
model.eval()

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*13, std=[0.5]*13)
])

# 类别名称映射 (根据EuroSAT数据集结构)
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    'River', 'SeaLake'
]

def predict_tif_image(image_path):
    """预测单个tif图片"""
    with rasterio.open(image_path) as src:
        img = src.read()  # 读取所有13个波段
        img = np.transpose(img, (1, 2, 0))  # 转换为HWC格式
        img = img.astype(np.float32)  # 转换为float32类型
    
    # 应用预处理
    img = transform(img)
    img = img.unsqueeze(0)  # 添加batch维度
    
    # 预测
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        class_idx = predicted.item()
        class_name = class_names[class_idx]
    
    return class_idx, class_name

def process_interface_folder(folder_path=INTERFACE_DIR):
    """处理Interface文件夹中的所有tif图片"""
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在")
        return
    
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            filepath = os.path.join(folder_path, filename)
            class_idx, class_name = predict_tif_image(filepath)
            results.append({
                'filename': filename,
                'class_idx': class_idx,
                'class_name': class_name
            })
            print(f"图片: {filename} -> 预测类别: {class_name} (ID: {class_idx})")
    
    return results

if __name__ == '__main__':
    print("开始处理Interface文件夹中的图片...")
    results = process_interface_folder()
    print("\n处理完成！预测结果汇总:")
    for result in results:
        print(f"{result['filename']}: {result['class_name']}")