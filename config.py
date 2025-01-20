import os
import torch

# 数据路径
DATA_PATH = 'lab5_data/data'  # 所有文件都在这个目录下
TRAIN_FILE = 'lab5_data/train.txt'
TEST_FILE = 'lab5_data/test_without_label.txt'

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型路径
MODEL_SAVE_PATH = 'models/'

# 超参数
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
WEIGHT_DECAY = 1e-4
