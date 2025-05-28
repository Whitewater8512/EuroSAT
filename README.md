# EuroSAT
EuroSAT 是用于土地利用和土地覆盖分类的数据集和深度学习基准。  
该数据集基于Sentinel-2卫星图像，涵盖13个光谱带，由10个类别组成，总共由27,000个标记和地理参考图像。  
[数据集入口](https://github.com/phelber/EuroSAT)

# 关于本仓库
本仓库主要是利用这个数据集做一些对比实验, 本人只是个AI小白qwq  
目前还在对图像类数据进行一些探索。

# 如何食用？
有关文件路径设置可以在config.py文件内进行修改噢~  

首先下载EuroSAT_MS数据集(存放.tif文件的那个), 将其放在代码文件夹同一目录下, 即可, 具体数据集目录结构如下:  
├─Eurosat_MS  
│  ├─AnnualCrop  
│  ├─Forest  
│  ├─HerbaceousVegetation  
│  ├─Highway  
│  ├─Industrial  
│  ├─Pasture  
│  ├─PermanentCrop  
│  ├─Residential  
│  ├─River  
│  └─SeaLake  

对于ResNet18神经网络模型, 首先运行train.py文件, 训练完毕后就会在Weight目录下生成模型权重以及loss图像。  
对于神经网络评估部分, 运行evaluate.py文件, 即可生成模型各项指标的报告, 包括准确率、Kappa系数以及混淆矩阵图等。  

对于传统分类器模型, 这些传统分类器都是基于sklearn标准库上调用的, 具体代码部分可以在oth_train.py文件内按需进行修改。  
