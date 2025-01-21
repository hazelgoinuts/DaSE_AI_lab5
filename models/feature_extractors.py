import torch
import torch.nn as nn
import timm
from torchvision import models
from transformers import BertModel, DistilBertModel, ViTModel, RobertaModel
from efficientnet_pytorch import EfficientNet

class ImageBackboneFactory:
    @staticmethod
    def create_backbone(backbone_type: str):
        if backbone_type == 'resnet50':
            model = models.resnet50(pretrained=True)
            out_features = model.fc.in_features
            return model, out_features
            
        elif backbone_type == 'resnet18':
            model = models.resnet18(pretrained=True)
            out_features = model.fc.in_features
            return model, out_features

        elif backbone_type == 'vit':
            model_name = 'vit_base_patch16_224'
            weights_path = './timm_models/vit_base_patch16_224.pth'
            
            model = timm.create_model(model_name, pretrained=False)

            checkpoint = torch.load(weights_path)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
            out_features = model.head.in_features
            return model, out_features
            

        else:
            raise ValueError(f"不支持的backbone类型: {backbone_type}")

class TextBackboneFactory:
    @staticmethod
    def create_backbone(backbone_type: str):
        if backbone_type == 'bert':
            return BertModel.from_pretrained('./bert-base-uncased')
        elif backbone_type == 'distilbert':
            return DistilBertModel.from_pretrained('./distilbert-base-uncased')
        elif backbone_type == 'roberta':
            return RobertaModel.from_pretrained('./roberta-base')
        else:
            raise ValueError(f"不支持的backbone类型: {backbone_type}")

class ImageFeatureExtractor(nn.Module):
    def __init__(self, backbone_type='resnet50'):
        super(ImageFeatureExtractor, self).__init__()
        self.backbone, out_features = ImageBackboneFactory.create_backbone(backbone_type)
        self.backbone_type = backbone_type  # 保存backbone类型
        
        # 移除最后的分类层
        if backbone_type.startswith('resnet'):
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        if self.backbone_type.startswith('resnet'):
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
        elif self.backbone_type == 'vit':
            x = self.backbone.forward_features(x)

            if x.dim() == 3:
                x = x[:, 0]

        x = self.fc(x)
        return x

class TextFeatureExtractor(nn.Module):
    def __init__(self, backbone_type='bert'):
        super(TextFeatureExtractor, self).__init__()
        self.backbone = TextBackboneFactory.create_backbone(backbone_type)
        hidden_size = self.backbone.config.hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # 取[CLS]标记的输出
        x = self.fc(x)
        return x