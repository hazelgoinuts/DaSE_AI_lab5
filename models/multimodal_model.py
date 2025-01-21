import torch
import torch.nn as nn
from models.feature_extractors import ImageFeatureExtractor, TextFeatureExtractor
from models.fusion.concat import ConcatFusion
from models.fusion.attention import AttentionFusion

class MultimodalModel(nn.Module):
    def __init__(self, fusion_type='concat', fusion_params=None, 
                 image_backbone='resnet50', text_backbone='bert'):
        super(MultimodalModel, self).__init__()
        
        # 保存backbone类型，用于数据预处理
        self.text_backbone = text_backbone
        self.image_backbone = image_backbone
        
        # 使用指定的backbone初始化特征提取器
        self.text_feature_extractor = TextFeatureExtractor(backbone_type=text_backbone)
        self.image_feature_extractor = ImageFeatureExtractor(backbone_type=image_backbone)
        
        # 融合层配置
        text_dim = 256  # 来自TextFeatureExtractor的输出维度
        image_dim = 256  # 来自ImageFeatureExtractor的输出维度
        hidden_dim = 256  # 融合层的隐藏维度
        
        fusion_params = fusion_params or {}
        
        # 根据fusion_type选择相应的融合策略
        if fusion_type == 'concat':
            self.fusion_layer = ConcatFusion(
                text_dim, 
                image_dim, 
                hidden_dim,
                dropout=fusion_params.get('fusion_dropout', 0.5)
            )
        elif fusion_type == 'attention':
            self.fusion_layer = AttentionFusion(
                text_dim,
                image_dim,
                hidden_dim,
                num_heads=fusion_params.get('attention_heads', 4),
                dropout=fusion_params.get('fusion_dropout', 0.5)
            )
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
            
        # 分类层
        self.classifier = nn.Linear(hidden_dim, 3)  # 3个类别

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_feature_extractor(input_ids, attention_mask)
        image_features = self.image_feature_extractor(image)
        
        # 使用选定的融合策略
        fused_features = self.fusion_layer(text_features, image_features)
        output = self.classifier(fused_features)
        return output