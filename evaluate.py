import os
import time
import torch
from tqdm import tqdm
from data_loader import test_loader
from model import ResNet18
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import seaborn as sns
from config import *

# 加载模型
model = ResNet18()
model.load_state_dict(torch.load(os.path.join(WEIGHT_DIR, 'model.pth'), weights_only=True))
model.eval()

correct = 0
total = 0
predictions = []
true_labels = []

st = time.time()
print("开始测试模型...")
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="测试进度"):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
print(f"测试完成，耗时：{time.time() - st}秒")

# 计算评估指标
oa = correct / total
kappa = cohen_kappa_score(true_labels, predictions)
cm = confusion_matrix(true_labels, predictions)
ua = np.diag(cm) / cm.sum(axis=0)

# 打印并保存结果到txt文件
with open(os.path.join(RESULT_DIR, 'evaluation_results.txt'), 'w') as f:
    f.write(f'Overall Accuracy (OA): {100 * oa:.4F}%\n')
    f.write(f'Kappa Index: {kappa:.4F}\n')
    f.write('User Accuracy (UA) per class:\n')
    for i, acc in enumerate(ua):
        f.write(f'Class {i}: {100 * acc:.4F}%\n')

print(f'Overall Accuracy (OA): {100 * oa:.4F}%')
print(f'Kappa Index: {kappa:.4F}')
print('User Accuracy (UA) per class:')
for i, acc in enumerate(ua):
    print(f'Class {i}: {100 * acc:.4F}%')

# 生成分类结果图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(true_labels, bins=len(set(true_labels)), alpha=0.7, label='True Labels')
plt.title('True Labels Distribution')
plt.xlabel('Class')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(predictions, bins=len(set(predictions)), alpha=0.7, label='Predicted Labels', color='orange')
plt.title('Predicted Labels Distribution')
plt.xlabel('Class')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'classification_results.png'))
plt.show()

# 生成混淆矩阵图
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'))
plt.show()