import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
from torchvision import transforms
import config

class MultimodalDataset(Dataset):
    def __init__(self, text_data, image_data, labels, tokenizer, max_len=128, transform=None):
        self.text_data = text_data
        self.image_data = image_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            # 以下为数据增强部分，若不需要则注释
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(15),      # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        image_path = self.image_data[idx]
        label = self.labels[idx]
        
        # 文本处理
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len
        )
        
        # 图像处理
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(text_path):
    texts = {}
    for filename in os.listdir(text_path):
        if filename.endswith('.txt'):
            file_id = filename.split('.')[0]
            try:
                with open(os.path.join(text_path, filename), 'r', encoding='ISO-8859-1') as f:
                    texts[file_id] = f.read().strip()
            except Exception as e:
                print(f"无法读取文件 {filename}: {str(e)}")
                texts[file_id] = ""
    return texts

def prepare_data(train_file, test_file, text_backbone='bert'):
    # 加载所有数据
    texts = load_data(config.DATA_PATH)
    
    # 处理标签
    train_data = []
    with open(train_file, 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            if line.strip():
                guid, label = line.strip().split(',')
                if guid in texts:
                    label_idx = {'positive': 0, 'neutral': 1, 'negative': 2}[label]
                    train_data.append((
                        texts[guid],
                        os.path.join(config.DATA_PATH, f"{guid}.jpg"),
                        label_idx
                    ))
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(
        train_data,
        test_size=config.VAL_RATIO,
        random_state=config.SEED
    )
    
    # 根据backbone类型初始化tokenizer
    if text_backbone == 'bert':
        tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    elif text_backbone == 'distilbert':
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('./distilbert-base-uncased')
    elif text_backbone == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('./roberta-base')
    else:
        raise ValueError(f"不支持的text_backbone类型: {text_backbone}")
    
    # 创建数据集
    train_dataset = MultimodalDataset(
        [x[0] for x in train_data],  # 文本
        [x[1] for x in train_data],  # 图像路径
        [x[2] for x in train_data],  # 标签
        tokenizer
    )
    
    val_dataset = MultimodalDataset(
        [x[0] for x in val_data],
        [x[1] for x in val_data],
        [x[2] for x in val_data],
        tokenizer
    )
    
    return train_dataset, val_dataset