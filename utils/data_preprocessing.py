import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from transformers import BertTokenizer
import config

class MultimodalDataset(Dataset):
    def __init__(self, text_data, image_data, labels, tokenizer, max_len=100):
        self.text_data = text_data
        self.image_data = image_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        image = self.image_data[idx]
        label = self.labels[idx]
        
        # 文本处理
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_len)
        
        # 图像处理
        image = torch.tensor(image, dtype=torch.float32)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_text_data(text_path):
    texts = {}
    for filename in os.listdir(text_path):
        if filename.endswith('.txt'):
            file_id = filename.split('.')[0]  # 获取文件名（不含后缀）
            try:
                # 首先尝试 UTF-8
                with open(os.path.join(text_path, filename), 'r', encoding='utf-8') as f:
                    texts[file_id] = f.read().strip()
            except UnicodeDecodeError:
                try:
                    # 如果 UTF-8 失败，尝试 GBK
                    with open(os.path.join(text_path, filename), 'r', encoding='gbk') as f:
                        texts[file_id] = f.read().strip()
                except UnicodeDecodeError:
                    try:
                        # 如果 GBK 也失败，尝试 latin-1
                        with open(os.path.join(text_path, filename), 'r', encoding='latin-1') as f:
                            texts[file_id] = f.read().strip()
                    except Exception as e:
                        print(f"无法读取文件 {filename}: {str(e)}")
                        texts[file_id] = ""  # 如果所有编码都失败，使用空字符串
    return texts

def load_image_data(data_path, image_size=(224, 224)):
    images = {}
    for filename in os.listdir(data_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_id = filename.split('.')[0]
            img_path = os.path.join(data_path, filename)
            img = Image.open(img_path).resize(image_size)
            # 转换为 numpy 数组并调整通道顺序
            img_array = np.array(img) / 255.0
            # 从 [height, width, channels] 转换为 [channels, height, width]
            img_array = np.transpose(img_array, (2, 0, 1))
            images[file_id] = img_array
    return images

def load_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            if line.strip():  # 确保不是空行
                guid, label = line.strip().split(',')
                # 将标签转换为数字
                if label == 'positive':
                    labels.append(0)
                elif label == 'neutral':
                    labels.append(1)
                elif label == 'negative':
                    labels.append(2)
    return np.array(labels)

def prepare_data(train_file, test_file):
    # 加载所有数据
    texts = load_text_data(config.DATA_PATH)
    images = load_image_data(config.DATA_PATH)
    labels = load_labels(train_file)
    
    # 找到所有数据集中共有的 ID
    text_ids = set(texts.keys())
    image_ids = set(images.keys())
    label_ids = set([str(i) for i in range(len(labels))])
    
    # 取交集
    common_ids = text_ids.intersection(image_ids).intersection(label_ids)
    
    # 只保留共有的数据
    filtered_texts = [texts[id] for id in common_ids]
    filtered_images = [images[id] for id in common_ids]
    filtered_labels = [labels[int(id)] for id in common_ids]
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        list(zip(filtered_texts, filtered_images)), 
        filtered_labels, 
        test_size=0.1, 
        random_state=42
    )
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    
    # 创建训练集和验证集的Dataset对象
    train_dataset = MultimodalDataset(
        [x[0] for x in X_train],  # 文本
        [x[1] for x in X_train],  # 图像
        y_train,                  # 标签
        tokenizer
    )
    
    val_dataset = MultimodalDataset(
        [x[0] for x in X_val],    # 文本
        [x[1] for x in X_val],    # 图像
        y_val,                    # 标签
        tokenizer
    )
    
    return train_dataset, val_dataset
