import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.multimodal_model import MultimodalModel
from utils.data_preprocessing import prepare_data
from tqdm import tqdm
import config
from utils.metrics_plotter import MetricsPlotter

# 准备数据
train_dataset, val_dataset = prepare_data(config.TRAIN_FILE, config.TEST_FILE)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# 创建模型
model = MultimodalModel().to(config.DEVICE)

# 编译模型
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# 在训练开始前，准备配置字典
config_dict = {
    'model_config': {
        'text_feature_dim': 256,    # bert输出特征降维后的维度，于models/feature_extractors.py中定义
        'image_feature_dim': 256,   # resnet输出特征降维后的维度，于models/feature_extractors.py中定义
        'fusion_hidden_dim': 256,   # 融合网络隐藏层维度，于models/multimodal_model.py中定义
        'num_classes': 3,          # 分类类别数
        'dropout_rate': 0.5         # dropout率
    },
    'training_config': {
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'epochs': config.EPOCHS,
        'weight_decay': config.WEIGHT_DECAY,    # L2正则化系数
        'val_ratio': config.VAL_RATIO,
        'early_stopping_patience': config.EARLY_STOPPING_PATIENCE
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
plotter = MetricsPlotter('results')
# 保存初始配置
plotter.save_config(config_dict)

# 训练模型
best_val_loss = float('inf')
patience_counter = 0

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
