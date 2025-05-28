import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA  # 导入PCA模块

from config import *
from data_loader import EuroSATDataset, transform

from skimage.feature import graycomatrix, graycoprops  # 计算图像的 GLCM 特征
from skimage.measure import regionprops  # 计算图像的形状特征

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_score
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''定义空间特征和形状特征的计算函数'''
# 定义计算 GLCM 特征的函数
def calculate_glcm_features(image):
    # 转换为灰度图像
    gray_image = np.mean(image, axis=-1).astype(np.uint8)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    # 计算对比度和熵
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm * np.log(glcm + 1e-10))
    return contrast, entropy

# 定义计算形状特征的函数
def calculate_shape_features(image):
    binary_image = np.any(image > 0, axis=-1).astype(np.uint8)
    props = regionprops(binary_image)
    if len(props) > 0:
        bbox = props[0].bbox
        aspect_ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
    else:
        aspect_ratio = 0
    return aspect_ratio

'''定义模型训练函数'''
def train_model(X_train, y_train, model_name):
    st = time.time()
    print(f"开始训练 {model_name}...")
    if model_name == 'SVM':
        model = SVC(kernel='linear', random_state=42, verbose=True, max_iter=7500000)
    elif model_name == 'Naive Bayes':
        model = GaussianNB()
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=42)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=250, max_depth=25, random_state=42)
    model.fit(X_train, y_train)
    print(f"{model_name} 训练完成, 耗时: {time.time() - st} 秒")
    return model

'''模型评估函数'''
def evaluate_model(model, X_test, y_test, model_name):
    st = time.time()
    print(f"开始进行 {model_name} 的预测...")
    y_pred = model.predict(X_test)
    print(f"{model_name} 预测完成, 耗时: {time.time() - st} 秒")

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 计算 Kappa 系数
    kappa = cohen_kappa_score(y_test, y_pred)

    # 计算总体准确率（OA），这里和准确率相同
    oa = accuracy

    # 计算用户准确率（UA）
    ua = precision_score(y_test, y_pred, average=None)

    # 生成混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 绘制混淆矩阵图
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot()
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

    # 保存结果到文件
    result_path = os.path.join(RESULT_DIR, f'{model_name}_Results.txt')
    with open(result_path, 'w') as f:
        f.write(f'{model_name}分类器准确率: {accuracy * 100}%\n')
        f.write(f'Kappa 系数: {kappa}\n')
        f.write(f'总体准确率 (OA): {oa * 100}%\n')
        f.write('用户准确率 (UA) 按类别:\n')
        for i, score in enumerate(ua):
            f.write(f'类别 {i}: {score * 100}%\n')
        f.write('混淆矩阵:\n')
        f.write(f'{cm}\n')

    print(f'{model_name}分类器准确率: {accuracy * 100}%')
    print(f'Kappa 系数: {kappa}')
    print(f'总体准确率 (OA): {oa * 100}%')
    print('用户准确率 (UA) 按类别:')
    for i, score in enumerate(ua):
        print(f'类别 {i}: {score * 100}%')
    print('混淆矩阵:')
    print(cm)
    print(f'结果已保存到 {result_path}')

# 新增：计算PCA特征的函数
def calculate_pca_features(images, n_components=5):
    # 展平图像数据用于PCA
    flattened_images = []
    for img in images:
        # 将图像展平为二维数组 (H*W, C)
        h, w, c = img.shape
        flattened = img.reshape(h*w, c)
        flattened_images.append(flattened)
    
    # 合并所有图像的像素
    all_pixels = np.vstack(flattened_images)
    
    # 应用PCA
    print(f"应用PCA降维，保留{n_components}个主成分...")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(all_pixels)
    
    print(f"PCA解释方差比例: {pca.explained_variance_ratio_}")
    print(f"累计解释方差比例: {np.sum(pca.explained_variance_ratio_)}")
    
    return pca

if __name__ == '__main__':
    '''数据集准备部分'''
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
    
    # 提取训练图像用于PCA
    print("提取训练图像用于PCA分析...")
    train_images = []
    for i, (img, label) in enumerate(tqdm(train_dataset, desc="提取训练图像")):
        img_np = img.numpy().transpose(1, 2, 0)  # 转换为(H, W, C)格式
        train_images.append(img_np)
        if i >= 1000:  # 限制用于PCA的样本数量，避免内存问题
            break
    
    # 计算PCA特征
    pca = calculate_pca_features(train_images, n_components=0.95)
    
    # 准备数据
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # 训练数据准备部分
    train_pbar = tqdm(train_dataset, desc="准备训练数据")
    for i, (img, label) in enumerate(train_pbar):
        img_np = img.numpy().transpose(1, 2, 0)  # 转换为(H, W, C)格式
        # 计算空间特征
        contrast, entropy = calculate_glcm_features(img_np)
        aspect_ratio = calculate_shape_features(img_np)
        
        # 使用PCA提取特征
        h, w, c = img_np.shape
        flattened_img = img_np.reshape(h*w, c)
        pca_features = pca.transform(flattened_img)
        # 将PCA特征重新整形并计算统计量
        pca_features_reshaped = pca_features.reshape(h, w, -1)
        # 使用均值作为最终特征
        spectral_features = np.mean(pca_features_reshaped, axis=(0, 1))
        
        # 合并特征
        features = np.concatenate([spectral_features, [contrast, entropy, aspect_ratio]])
        X_train.append(features)
        y_train.append(label)
        if i % 1000 == 0:
            torch.cuda.empty_cache()

    # 测试数据准备部分
    test_pbar = tqdm(test_dataset, desc="准备测试数据")
    for i, (img, label) in enumerate(test_pbar):
        img_np = img.numpy().transpose(1, 2, 0)  # 转换为(H, W, C)格式
        # 计算空间特征
        contrast, entropy = calculate_glcm_features(img_np)
        aspect_ratio = calculate_shape_features(img_np)
        
        # 使用PCA提取特征
        h, w, c = img_np.shape
        flattened_img = img_np.reshape(h*w, c)
        pca_features = pca.transform(flattened_img)
        # 将PCA特征重新整形并计算统计量
        pca_features_reshaped = pca_features.reshape(h, w, -1)
        # 使用均值作为最终特征
        spectral_features = np.mean(pca_features_reshaped, axis=(0, 1))
        
        # 合并特征
        features = np.concatenate([spectral_features, [contrast, entropy, aspect_ratio]])
        X_test.append(features)
        y_test.append(label)
        if i % 1000 == 0:
            torch.cuda.empty_cache()

    print(f"训练数据准备完成: {len(X_train)} 个训练样本, {len(X_test)} 个测试样本, 耗时: {time.time() - st} 秒")

    '''模型训练部分'''
    # svm_model = train_model(X_train, y_train, 'SVM')
    rf_model = train_model(X_train, y_train, 'Random Forest')
    # nb_model = train_model(X_train, y_train, 'Naive Bayes')
    # lr_model = train_model(X_train, y_train, 'Logistic Regression')


    '''模型评估部分'''
    # evaluate_model(svm_model, X_test, y_test, 'SVM')
    evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    # evaluate_model(nb_model, X_test, y_test, 'Naive Bayes')
    # evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')