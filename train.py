import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.multimodal_model import MultimodalModel
from utils.data_preprocessing import prepare_data
import config

# 准备数据
train_dataset, val_dataset = prepare_data(config.TRAIN_FILE, config.TEST_FILE)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# 创建模型
model = MultimodalModel().to(config.DEVICE)

# 编译模型
optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1, 
    patience=2,
    verbose=True
)

# 训练模型
for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        image = batch['image'].to(config.DEVICE)
        labels = batch['label'].to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            image = batch['image'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)

            outputs = model(input_ids, attention_mask, image)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 添加 ()
    
    # 打印训练信息
    print(f'Epoch [{epoch+1}/{config.EPOCHS}]')
    print(f'Training Loss: {total_loss/len(train_loader):.4f}')
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')
    
    # 在训练循环中更新学习率
    scheduler.step(val_loss)