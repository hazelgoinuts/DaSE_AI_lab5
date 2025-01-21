import os
import torch
import random
import numpy as np

def set_seed(seed=42):
    """
    设置所有可能的随机种子，确保实验可复现
    """
    # Python随机数生成器
    random.seed(seed)
    # Numpy随机数生成器
    np.random.seed(seed)
    # PyTorch随机数生成器
    torch.manual_seed(seed)
    # CUDA随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 设置CUDA的确定性选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据路径
DATA_PATH = 'lab5_data/data'  # 所有文件都在这个目录下
TRAIN_FILE = 'lab5_data/train.txt'
TEST_FILE = 'lab5_data/test_without_label.txt'

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型路径
MODEL_SAVE_PATH = 'models/'

# 超参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 10
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.3  # 验证集比例
EARLY_STOPPING_PATIENCE = 8  # 早停耐心值

# 设置随机种子
SEED = 42
set_seed(SEED)
