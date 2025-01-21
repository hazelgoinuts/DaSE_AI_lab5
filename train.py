import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.multimodal_model import MultimodalModel
from utils.data_preprocessing import prepare_data
from tqdm import tqdm
import config
from utils.metrics_plotter import MetricsPlotter


# 添加命令行参数
parser = argparse.ArgumentParser(description='多模态情感分析训练脚本')

# 融合策略相关参数
parser.add_argument('--fusion_type', type=str, default='concat',
                   choices=['concat', 'attention'],
                   help='选择融合策略类型: concat(简单拼接) 或 attention(注意力机制)')
parser.add_argument('--attention_heads', type=int, default=4,
                   help='注意力机制的head数量(仅在fusion_type=attention时有效)')
parser.add_argument('--fusion_dropout', type=float, default=0.5,
                   help='融合层的dropout率')

# 其他训练参数
parser.add_argument('--batch_size', type=int, default=32,
                   help='批次大小')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                   help='学习率')
parser.add_argument('--epochs', type=int, default=10,
                   help='训练轮数')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                   help='权重衰减')
parser.add_argument('--val_ratio', type=float, default=0.3,
                   help='验证集比例')
parser.add_argument('--early_stopping_patience', type=int, default=5,
                   help='早停耐心值')

# 在现有的参数基础上添加backbone选择
parser.add_argument('--image_backbone', type=str, default='resnet50',
                   choices=['resnet50', 'resnet18', 'vit'],
                   help='选择图像特征提取器的backbone')
parser.add_argument('--text_backbone', type=str, default='bert',
                   choices=['bert', 'distilbert'],
                   help='选择文本特征提取器的backbone')
parser.add_argument('--vit_model_name', type=str, default='vit_base_patch16_224',
                   help='ViT模型名称 (仅在image_backbone=vit时有效)')

args = parser.parse_args()

# 更新config中的参数
config.BATCH_SIZE = args.batch_size
config.LEARNING_RATE = args.learning_rate
config.EPOCHS = args.epochs
config.WEIGHT_DECAY = args.weight_decay
config.VAL_RATIO = args.val_ratio
config.EARLY_STOPPING_PATIENCE = args.early_stopping_patience

# 准备数据
train_dataset, val_dataset = prepare_data(
    config.TRAIN_FILE, 
    config.TEST_FILE,
    text_backbone=args.text_backbone  # 传递text_backbone参数
)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# 创建模型
fusion_params = {
    'attention_heads': args.attention_heads,
    'fusion_dropout': args.fusion_dropout
}

model = MultimodalModel(
    fusion_type=args.fusion_type,
    fusion_params=fusion_params,
    image_backbone=args.image_backbone,
    text_backbone=args.text_backbone
).to(config.DEVICE)


# 编译模型
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# 在训练开始前，准备配置字典
config_dict = {
    'model_config': {
        'text_feature_dim': 256,
        'image_feature_dim': 256,
        'fusion_type': args.fusion_type,
        'fusion_params': fusion_params,
        'image_backbone': args.image_backbone,
        'text_backbone': args.text_backbone,
        'num_classes': 3,
    },
    'training_config': {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'val_ratio': args.val_ratio,
        'early_stopping_patience': args.early_stopping_patience
    },
    'data_config': {
        'max_text_length': 128,
        'image_size': 224,
        'data_augmentation': {
            'random_flip': True,
            'random_rotation': 15,  # 随机旋转角度
            'color_jitter': {
                'brightness': 0.2,  # 亮度变化范围
                'contrast': 0.2,    # 对比度变化范围
                'saturation': 0.2    # 饱和度变化范围
            }
        }
    },
    'best_metrics': None  # 将在训练结束后更新
}

# 初始化绘图器
plotter = MetricsPlotter('output')
# 保存初始配置
plotter.save_config(config_dict)

# 训练模型
best_val_loss = float('inf')
patience_counter = 0

print(f"使用的图像backbone: {args.image_backbone}")
print(f"使用的文本backbone: {args.text_backbone}")
print(f"使用的融合策略: {args.fusion_type}")

for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    
    # 添加进度条
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]')
    for batch in train_pbar:
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        images = batch['image'].to(config.DEVICE)
        labels = batch['label'].to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct_preds += (torch.argmax(outputs, dim=1) == labels).sum().item()
        total_preds += labels.size(0)
        
        # 更新进度条信息
        current_loss = loss.item()
        current_acc = 100 * (torch.argmax(outputs, dim=1) == labels).sum().item() / labels.size(0)
        train_pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    train_loss = total_loss / len(train_loader)
    train_acc = correct_preds / total_preds
    
    # 验证
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    # 验证集进度条
    val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Valid]')
    with torch.no_grad():
        for batch in val_pbar:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            images = batch['image'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)

            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            val_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            val_total += labels.size(0)
            
            # 更新验证进度条信息
            current_val_loss = loss.item()
            current_val_acc = 100 * (torch.argmax(outputs, dim=1) == labels).sum().item() / labels.size(0)
            val_pbar.set_postfix({
                'loss': f'{current_val_loss:.4f}',
                'acc': f'{current_val_acc:.2f}%'
            })

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # 更新并绘制曲线
    plotter.update(train_loss, val_loss, train_acc, val_acc)
    plotter.plot()

    # 打印每个epoch的详细信息
    print(f'\nEpoch [{epoch+1}/{config.EPOCHS}] Summary:')
    print(f'Training    - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
    print(f'Validation  - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

    # 早停逻辑
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print(f'✓ New best validation loss! Saving model...')
        torch.save(model.state_dict(), f"{config.MODEL_SAVE_PATH}/best_model.pth")
    else:
        patience_counter += 1
        print(f'✗ No improvement in validation loss. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}')
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("\n早停触发！训练结束。")
            break
    print('-' * 60)

# 在训练结束后，更新并保存最终配置
config_dict['best_metrics'] = plotter.best_metrics
plotter.save_config(config_dict)
