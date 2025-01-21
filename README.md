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

### 训练模型

基本训练命令：
```bash
python train.py \
    --fusion_type concat \
    --image_backbone resnet50 \
    --text_backbone bert \
    --batch_size 32 \
    --learning_rate 1e-5
```

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
