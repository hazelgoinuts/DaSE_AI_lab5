# 多模态情感分析任务

本项目实现了一个多模态情感分析模型，可以同时处理文本和图像数据，对社交媒体内容进行情感分类（积极、中性、消极）。

## 环境要求

- Python 3.7+
- CUDA 11.0+ (如果使用GPU)

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

将数据集按以下结构放置：
```
data/
├── images/
│   ├── [guid1].jpg
│   ├── [guid2].jpg
│   └── ...
├── train.csv
└── test.csv
```

## 模型训练

### 基础训练命令

```bash
python train.py --modality multimodal --text_backbone bert --image_backbone resnet50 --fusion_type concat
```

### 参数说明

- `--modality`: 选择输入模态 ['text', 'image', 'multimodal']
- `--text_backbone`: 文本特征提取器 ['bert', 'distilbert', 'roberta']
- `--image_backbone`: 图像特征提取器 ['resnet50', 'resnet18', 'vit']
- `--fusion_type`: 特征融合方式 ['concat', 'attention']

### 消融实验

运行单模态实验：

```bash
# 仅文本
python train.py --modality text --text_backbone bert

# 仅图像
python train.py --modality image --image_backbone resnet50
```

## 项目结构

```
.
├── models/
│   ├── feature_extractors.py  # 特征提取器实现
│   └── multimodal_model.py    # 多模态模型实现
├── utils/
│   └── data_preprocessing.py  # 数据预处理
├── train.py                   # 训练脚本
├── requirements.txt           # 依赖包列表
└── README.md                  # 说明文档
```

## 注意事项

1. 首次运行时会自动下载预训练模型，请确保网络连接正常
2. 建议使用GPU进行训练，可以显著提升训练速度
3. 如果显存不足，可以尝试减小batch_size
4. 对于不同的backbone，建议使用不同的学习率

## 结果

模型在验证集上的表现：

- 多模态：[待填写]
- 仅文本：[待填写]
- 仅图像：[待填写]

## 引用


这两个文件涵盖了：
1. 所有必要的依赖包及其版本要求
2. 详细的项目说明，包括安装、运行和参数配置
3. 完整的项目结构说明
4. 注意事项和使用建议

可以根据实际情况修改其中的占位符（如[待填写]部分）和具体的性能数据。
