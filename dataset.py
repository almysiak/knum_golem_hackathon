import torch

class ContrastiveLearningDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        transforms,
    ):
        self.transforms = transforms
        self.images = images
        self.labels = labels

        self.prior_len = len(self.images)
        self.prod_len = self.prior_len ** 2
        
        self.classes_val_count = labels.value_counts()
        
        self.pairs = []
        self.weights = []
        

    def __getitem__(self, idx):
        id_1 = idx // self.prior_len
        id_2 = idx % self.prior_len
        
        image_1 = self.transforms(
            self.images[id_1]
        )
        label_1 = self.labels[id_1]
        
        image_2 = self.transforms(
            self.images[id_2]
        )
        label_2 = self.labels[id_2]
        
        sign = 1 if label_1 == label_2 else -1
        weight = self.prior_len / (self.classes_val_count[label_1] * self.classes_val_count[label_2])
        
        return (
            image_1,
            image_2,
            sign,
            weight
        )

    def __len__(self):
        return self.prod_len

