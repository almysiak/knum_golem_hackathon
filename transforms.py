import torch
from torch import Tensor
from torchvision.transforms import transforms as T, functional as F
import torchvision

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
    
class ConvertImageDtype(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, image):
        image = F.convert_image_dtype(image, self.dtype)
        return image
    
class Normalize(torch.nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.normalizer = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, image):
        return self.normalizer(image)
        

train_transforms = T.Compose([
    T.RandomRotation(90, expand=True),
    T.RandomPerspective(distortion_scale=0.2),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
    ResizeToSquare(224),
    ConvertImageDtype(torch.float32)
])

val_transforms = T.Compose([
    ResizeToSquare(224),
    ConvertImageDtype(torch.float32)
])

zoo_transforms = [
    T.Compose([aug, ResizeToSquare(224), ConvertImageDtype(torch.float32)]) 
        for aug in [T.RandomRotation(90, expand=True),
            T.RandomPerspective(distortion_scale=0.2),
            # T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),]
]


