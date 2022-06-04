from turtle import forward
import torch
from torchvision.transforms import transforms as T, functional as F

class ResizeToSquare(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        
    def forward(self, x):
        x.thumbnail([self.scale, self.scale])
        x = T.ToTensor()(x)

        # pad with zeros to shape (3, scale, scale)
        pad = torch.zeros((3, self.scale - x.shape[1], x.shape[2]))
        # print(x.shape, pad.shape)

        x = torch.cat([x, pad], dim=1)
        pad = torch.zeros((3, self.scale, self.scale - x.shape[2]))
        # print(x.shape, pad.shape)

        x = torch.cat([x, pad], dim=2)
        x = (255 * x).to(torch.uint8)
        # print(x.shape)
        return x
        
        
        
        

train_transforms = T.Compose([
    T.RandomRotation(90, expand=True),
    T.RandomPerspective(distortion_scale=0.2),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
    ResizeToSquare(224)
])


