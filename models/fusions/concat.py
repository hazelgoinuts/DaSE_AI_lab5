import torch
import torch.nn as nn
from .base import BaseFusion

class ConcatFusion(BaseFusion):
    """简单拼接融合"""
    def __init__(self, text_dim, image_dim, output_dim, dropout=0.5):
        super(ConcatFusion, self).__init__(text_dim, image_dim, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(text_dim + image_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, text_features, image_features):
        combined = torch.cat((text_features, image_features), dim=1)
        return self.fc(combined)