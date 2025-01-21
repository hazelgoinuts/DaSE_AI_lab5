import torch
import torch.nn as nn
from .base import BaseFusion

class AttentionFusion(BaseFusion):
    """注意力融合"""
    def __init__(self, text_dim, image_dim, output_dim, num_heads=4, dropout=0.5):
        super(AttentionFusion, self).__init__(text_dim, image_dim, output_dim)
        
        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 投影层
        self.fc = nn.Sequential(
            nn.Linear(text_dim + image_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    # def forward(self, text_features, image_features):
    #     # 重塑特征以适应多头注意力层
    #     text_features = text_features.unsqueeze(1)  # [batch_size, 1, text_dim]
    #     image_features = image_features.unsqueeze(1)  # [batch_size, 1, image_dim]
        
    #     # 计算注意力
    #     attn_output, _ = self.multihead_attn(
    #         query=text_features,
    #         key=image_features,
    #         value=image_features
    #     )
        
    #     # 合并特征
    #     attn_output = attn_output.squeeze(1)
    #     combined = torch.cat((attn_output, image_features.squeeze(1)), dim=1)
    #     return self.fc(combined)

    def forward(self, text_features, image_features):
    # 确保输入是3D张量 [batch_size, seq_len, dim]
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
    
        # 计算注意力
        attn_output, _ = self.multihead_attn(
            query=text_features,
            key=image_features,
            value=image_features
        )
    
        # 合并特征
        attn_output = attn_output.squeeze(1)
        image_features = image_features.squeeze(1)
        combined = torch.cat((attn_output, image_features), dim=1)
        return self.fc(combined)