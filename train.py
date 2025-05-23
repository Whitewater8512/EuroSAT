import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from data_loader import train_loader
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from model import ResNet18
from config import *

# 初始化模型、损失函数和优化器
model = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

st = time.time()
losses = []
print("开始训练模型...")
# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(tqdm(train_loader, desc="训练进度")):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')
print(f"训练完成，耗时：{time.time() - st}秒")

plt.figure()
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.title('训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join(WEIGHT_DIR, 'train_loss.png'))
plt.show()

# 保存模型
torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, 'model.pth'))