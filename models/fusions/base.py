import torch.nn as nn

class BaseFusion(nn.Module):
    """融合策略的基类"""
    def __init__(self, text_dim, image_dim, output_dim):
        super(BaseFusion, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = output_dim

    def forward(self, text_features, image_features):
        raise NotImplementedError