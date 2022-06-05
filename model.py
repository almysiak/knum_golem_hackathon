from turtle import forward
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

from ipynb.fs.full.fb_model_extractor import DeiTForImageClassificationWithTeacher
from transformers import AutoFeatureExtractor

class ImageEmbeddingModel(nn.Module):
    def __init__(self, out_dim=128, efficientnet_model_name="efficientnet-b0") -> None:
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(efficientnet_model_name)
        self.last_layer = nn.Linear(7*7*1280, out_dim)
    def forward(self, x):
        out = self.backbone.extract_endpoints(x)['reduction_6']
        out = out.reshape((-1, 7*7*1280))
        out = self.last_layer(out)
        # out = torch.nn.functional.normalize(out)
        return out
    def get_last_params(self):
        return self.last_layer.parameters()
    def get_backbone_params(self):
        return self.backbone.parameters()
    
    
class ImageEmbeddingWithTransofmers(nn.Module):
    def __init__(self, out_dim=128, freeze_backbone=True, device='cpu'):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-small-distilled-patch16-224') # TODO to nie jest dotrenowywane
        self.model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-small-distilled-patch16-224')
        self.last_layer = nn.Linear(384, out_dim)
        self.freeze_backbone = freeze_backbone
        self.device = device
        self.model.to(device)
        self.last_layer.to(device)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        for param in self.last_layer.parameters():
                param.requires_grad = True
            
            # for param in self.feature_extractor.parameters():
            #     param.requires_grad = False
    def forward(self, x):
        out = self.model(**self.feature_extractor(x, return_tensors='pt').to(self.device))
        out = self.last_layer(out)
        out = torch.nn.functional.normalize(out)
        return out
    def get_last_params(self):
        return self.last_layer.parameters()
    def to_device(self, device):
        self.device = device
        self.model.to(device)
        self.last_layer.to(device)

        
        