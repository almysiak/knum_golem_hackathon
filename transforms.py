from torchvision.transforms import transforms as T

train_transforms = T.Compose([
    T.RandomRotation(90),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomPerspective(distortion_scale=0.2),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0)
])

