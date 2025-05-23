import time
import torch
from config import *
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_loader import EuroSATDataset, transform


print("开始训练")
print("加载数据...")
# 创建数据集
dataset = EuroSATDataset(root_dir=DATA_DIR, transform=transform)
print("数据加载完成")

# 7:3的比例划分训练集和测试集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

st = time.time()
print("准备训练数据中...")
# 准备数据
X_train = []
y_train = []
X_test = []
y_test = []

# 训练数据准备部分
train_pbar = tqdm(train_dataset, desc="准备训练数据")
for i, (img, label) in enumerate(train_pbar):
    X_train.append(img.flatten().numpy())
    y_train.append(label)
    if i % 100 == 0:  # 每100个样本更新一次描述
        mem_usage = torch.cuda.memory_allocated() / 1024**2
        train_pbar.set_description(f"准备训练数据, 当前内存使用: {mem_usage:.2f}MB")
    if i % 1000 == 0:
        torch.cuda.empty_cache()

# 测试数据准备部分
test_pbar = tqdm(test_dataset, desc="准备测试数据")
for i, (img, label) in enumerate(test_pbar):
    X_test.append(img.flatten().numpy())
    y_test.append(label)
    if i % 100 == 0:  # 每100个样本更新一次描述
        mem_usage = torch.cuda.memory_allocated() / 1024**2
        test_pbar.set_description(f"准备测试数据, 当前内存使用: {mem_usage:.2f}MB")
    if i % 1000 == 0:
        torch.cuda.empty_cache()

print(f"训练数据准备完成: {len(X_train)} 个训练样本, {len(X_test)} 个测试样本, 耗时: {time.time() - st} 秒")

st = time.time()
print("开始训练随机森林分类器...")
# 初始化并训练随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2)
rf.fit(X_train, y_train)
print(f"随机森林分类器训练完成, 耗时: {time.time() - st} 秒")

# 进行预测
st = time.time()
print("开始进行预测...")
y_pred = rf.predict(X_test)
print(f"预测完成, 耗时: {time.time() - st} 秒")

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 保存结果到文件
result_path = os.path.join(RESULT_DIR, 'RandomForest_Results.txt')
with open(result_path, 'w') as f:
    f.write(f'随机森林分类器准确率: {accuracy * 100}%\n')

print(f'随机森林分类器准确率: {accuracy * 100}%')
print(f'结果已保存到 {result_path}')