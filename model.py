from turtle import forward
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

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
        