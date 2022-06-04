from turtle import forward
import torch
from torchvision.transforms import transforms as T, functional as F

class ResizeToSquare(torch.nn.Module):
    def __init__(self, scale):
        super.__init__()
        self.scale = scale
    def forward(self, x):
        

train_transforms = T.Compose([
    T.RandomRotation(90, expand=True),
    T.RandomPerspective(distortion_scale=0.2),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
    T.ToTensor()
])


