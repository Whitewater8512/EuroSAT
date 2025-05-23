import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
DATA_DIR = os.path.join(BASE_DIR, 'EuroSAT_MS')

# 权重保存路径
WEIGHT_DIR = os.path.join(BASE_DIR, 'Weight')

# 结果保存路径
RESULT_DIR = os.path.join(BASE_DIR, 'Result')

# 接口路径
INTERFACE_DIR = os.path.join(BASE_DIR, 'Interface')