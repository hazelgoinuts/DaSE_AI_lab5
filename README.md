# Multimodal Sentiment Analysis

基于多模态融合的情感分析系统，支持文本和图像的联合分析。该项目实现了一个灵活的多模态情感分析框架，支持多种backbone和融合策略的切换，可用于处理包含文本和图像的情感分析任务。

## 特点

- 支持多种图像backbone（ResNet18/ResNet50/ViT）
- 支持多种文本backbone（BERT/DistilBERT/RoBERTa）
- 提供多种融合策略（Concat/Attention）
- 完整的训练和推理流程
- 详细的训练过程可视化
- 灵活的命令行参数配置

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (GPU版本)

## 安装说明

1. 克隆项目
```bash
git clone https://github.com/yourusername/multimodal-sentiment-analysis.git
cd multimodal-sentiment-analysis
```

2. 创建虚拟环境
```bash
conda create -n multimodal python=3.8
conda activate multimodal
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 下载预训练模型
```bash
# 下载BERT模型
wget https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin -P ./bert-base-uncased/
wget https://huggingface.co/bert-base-uncased/resolve/main/config.json -P ./bert-base-uncased/
wget https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt -P ./bert-base-uncased/

# 下载ViT模型
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth -P ./timm_models/
```

## 项目结构
```
multimodal-sentiment-analysis/
├── models/                    # 模型相关代码
│   ├── feature_extractors.py  # 特征提取器
│   ├── multimodal_model.py   # 多模态模型
│   └── fusion/              # 融合策略
│       ├── concat.py        # 拼接融合
│       └── attention.py     # 注意力融合
├── utils/                    # 工具函数
│   ├── data_preprocessing.py # 数据预处理
│   └── metrics_plotter.py   # 指标绘制
├── config.py                 # 配置文件
├── train.py                 # 训练脚本
└── requirements.txt         # 依赖库
```

## 数据准备

1. 准备数据集
```
lab5_data/
├── data/           # 包含所有文本和图像文件
├── train.txt      # 训练数据标注
└── test_without_label.txt  # 测试数据
```

2. 数据格式
- train.txt 格式：
```
guid,tag
1,positive
2,neutral
3,negative
```

## 使用说明


本项目使用argparse进行命令行参数管理，所有可配置的参数如下：

### 1. 融合策略参数

```bash
--fusion_type     # 选择融合策略类型
                  # 默认值: concat
                  # 可选值: [concat, attention]
                  
--attention_heads # 注意力机制的head数量(仅在fusion_type=attention时有效)
                  # 默认值: 4
                  
--fusion_dropout  # 融合层的dropout率
                  # 默认值: 0.5
```

### 2. 模型backbone参数

```bash
--image_backbone  # 选择图像特征提取器的backbone
                  # 默认值: resnet50
                  # 可选值: [resnet50, resnet18, vit]
                  
--text_backbone   # 选择文本特征提取器的backbone
                  # 默认值: bert
                  # 可选值: [bert, distilbert, roberta]
                  
--vit_model_name  # ViT模型名称(仅在image_backbone=vit时有效)
                  # 默认值: vit_base_patch16_224
```

### 3. 训练相关参数

```bash
--batch_size     # 批次大小
                 # 默认值: 32

--learning_rate  # 学习率
                 # 默认值: 1e-5

--epochs         # 训练轮数
                 # 默认值: 10

--weight_decay   # 权重衰减
                 # 默认值: 1e-3

--val_ratio     # 验证集比例
                # 默认值: 0.3

--early_stopping_patience  # 早停耐心值
                          # 默认值: 5
```

### 4. 输出相关参数

```bash
--output_file    # 预测结果输出文件路径
                 # 默认值: predictions.txt
```

### 使用示例

1. 使用默认配置：
```bash
python train.py
```

2. 使用concat融合和ResNet50：
```bash
python train.py \
    --fusion_type concat \
    --image_backbone resnet50 \
    --text_backbone bert
```

3. 使用attention融合和ViT：
```bash
python train.py \
    --fusion_type attention \
    --image_backbone vit \
    --attention_heads 8 \
    --fusion_dropout 0.3
```

4. 自定义训练参数：
```bash
python train.py \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --epochs 20 \
    --early_stopping_patience 8
```

5. 完整参数示例：
```bash
python train.py \
    --fusion_type attention \
    --attention_heads 4 \
    --fusion_dropout 0.5 \
    --image_backbone vit \
    --text_backbone roberta \
    --vit_model_name vit_base_patch16_224 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --epochs 10 \
    --weight_decay 1e-3 \
    --val_ratio 0.3 \
    --early_stopping_patience 5 \
    --output_file predictions.txt
```

### 注意事项

1. 当使用attention融合策略时，需要同时指定attention_heads参数
2. 当使用vit作为image_backbone时，可以通过vit_model_name指定具体的ViT模型
3. 所有参数都提供了默认值，可以根据需要选择性地修改部分参数
4. early_stopping_patience参数控制早停机制，当验证集损失在指定轮数内没有改善时，训练将提前终止

### 主要参数说明

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| fusion_type | 融合策略 | concat | concat/attention |
| image_backbone | 图像特征提取器 | resnet50 | resnet50/resnet18/vit |
| text_backbone | 文本特征提取器 | bert | bert/distilbert/roberta |
| batch_size | 批次大小 | 32 | - |
| learning_rate | 学习率 | 1e-5 | - |
| epochs | 训练轮数 | 10 | - |
| output_file | 预测结果保存路径 | predictions.txt | - |

### 训练结果

训练过程中会自动生成：
- 训练损失和准确率曲线
- 最佳模型保存
- 预测结果文件

预测结果格式：
```
guid,tag
1,positive
2,neutral
3,negative
```

## 实验结果

1. 模型性能
- 验证集准确率：xx%
- 测试集准确率：xx%

2. 可视化示例
- 训练过程曲线
- 注意力可视化
