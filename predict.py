import torch
import argparse
from torch.utils.data import DataLoader
from models.multimodal_model import MultimodalModel
from utils.data_preprocessing import MultimodalDataset, load_data
from transformers import BertTokenizer, DistilBertTokenizer
import config
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import os

def create_test_dataset(text_data, image_paths, tokenizer, transform=None):
    """创建测试数据集"""
    return MultimodalDataset(
        text_data,
        image_paths,
        [0] * len(text_data),  # 填充标签
        tokenizer,
        transform=transform
    )

def predict(model, test_loader, device):
    """进行预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            
            outputs = model(input_ids, attention_mask, images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return predictions

def save_predictions(predictions, test_file, output_file):
    """保存预测结果"""
    # 读取原始测试文件
    test_df = pd.read_csv(test_file)
    
    # 将数字标签转换为文本标签
    label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    test_df['tag'] = [label_map[pred] for pred in predictions]
    
    # 保存预测结果
    test_df.to_csv(output_file, index=False)
    print(f"预测结果已保存至: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='多模态情感分析预测脚本')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--fusion_type', type=str, default='concat',
                       choices=['concat', 'attention'],
                       help='选择融合策略类型: concat(简单拼接) 或 attention(注意力机制)')
    parser.add_argument('--image_backbone', type=str, default='resnet50',
                       choices=['resnet50', 'resnet18', 'vit'],
                       help='选择图像特征提取器的backbone')
    parser.add_argument('--text_backbone', type=str, default='bert',
                       choices=['bert', 'distilbert'],
                       help='选择文本特征提取器的backbone')
    
    # 数据相关参数
    parser.add_argument('--test_file', type=str, default=config.TEST_FILE,
                       help='测试数据文件路径')
    parser.add_argument('--output_file', type=str, default='predictions.csv',
                       help='预测结果输出文件路径')
    parser.add_argument('--batch_size', type=str, default=32,
                       help='批次大小')
    
    args = parser.parse_args()
    
    # 设置设备
    device = config.DEVICE
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载模型...")
    fusion_params = {
        'attention_heads': 4,  # 默认值
        'fusion_dropout': 0.5  # 默认值
    }
    model = MultimodalModel(
        fusion_type=args.fusion_type,
        fusion_params=fusion_params,
        image_backbone=args.image_backbone,
        text_backbone=args.text_backbone
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path))
    print(f"模型已加载: {args.model_path}")
    
    # 准备数据
    print("正在准备数据...")
    # 初始化tokenizer
    if args.text_backbone == 'bert':
        tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    elif args.text_backbone == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('./distilbert-base-uncased')
        
    # 加载测试数据
    test_df = pd.read_csv(args.test_file)
    texts = load_data(config.DATA_PATH)
    
    test_texts = []
    test_image_paths = []
    for guid in test_df['guid']:
        test_texts.append(texts[str(guid)])
        test_image_paths.append(os.path.join(config.DATA_PATH, f"{guid}.jpg"))
    
    # 创建测试数据集和数据加载器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = create_test_dataset(test_texts, test_image_paths, tokenizer, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 进行预测
    print("开始预测...")
    predictions = predict(model, test_loader, device)
    
    # 保存预测结果
    save_predictions(predictions, args.test_file, args.output_file)

if __name__ == "__main__":
    main()