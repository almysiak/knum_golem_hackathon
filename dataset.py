import torch
import numpy as np

def list_resize(l, s):
    list_len = len(l)
    reps = s // list_len
    rest = s - reps * list_len
    res = l * reps + l[:rest]
    return res
     
    

# class ContrastiveLearningDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         images,
#         labels,
#         transforms,
#     ):
#         self.transforms = transforms
#         self.images = images
#         self.labels = labels

#         self.prior_len = len(self.images)
#         self.prod_len = self.prior_len ** 2
        
#         self.classes_val_count = labels.value_counts()
        
#         self.pairs = []
#         self.weights = []
        

#     def __getitem__(self, idx):
#         id_1 = idx // self.prior_len
#         id_2 = idx % self.prior_len
        
#         image_1 = self.transforms(
#             self.images[id_1]
#         )
#         label_1 = self.labels[id_1]
        
#         image_2 = self.transforms(
#             self.images[id_2]
#         )
#         label_2 = self.labels[id_2]
        
#         sign = 1 if label_1 == label_2 else -1
#         weight = self.prior_len / (self.classes_val_count[label_1] * self.classes_val_count[label_2])
        
#         return (
#             image_1,
#             image_2,
#             sign,
#             weight
#         )

#     def __len__(self):
#         return self.prod_len
    
class ContrastiveLearningDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        transforms,
    ):
        self.transforms = transforms

        classes_val_count = labels.value_counts()
        images_by_class = {}
        
        for i, l in zip(images, labels):
            if l not in images_by_class.keys():
                images_by_class[l] = [i]
            else:
                images_by_class[l].append(i)
        
        max_count = classes_val_count.max()
        for l in images_by_class.keys():
            images_by_class[l] = list_resize(images_by_class[l], max_count)
            
        self.images = []
        self.labels = []
        
        for l in images_by_class.keys():
            self.images = self.images + images_by_class[l]
            self.labels = self.labels + [l] * max_count
        
        assert len(self.images) == len(self.labels)

    def __getitem__(self, idx):
        image = self.transforms(self.images[idx])
        label = self.labels[idx]
        return (
            image,
            label
        )
    def __len__(self):
        return len(self.labels)
    
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        transforms,
    ):
        self.transforms = transforms
        self.images = images
        self.labels = labels   

    def __getitem__(self, idx): 
        image = self.transforms(
            self.images[idx]
        )
        label = self.labels[idx]
        
        return (
            image,
            label
        )

    def __len__(self):
        return len(self.labels)

