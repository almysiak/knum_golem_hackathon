from ipynb.fs.full.read_in_data import read_in_data
import json
import torch
from torchvision.transforms import transforms as T
from tqdm.notebook import tqdm
from transforms import train_transforms, val_transforms
from dataset import SimpleDataset

BATCH_SIZE = 8

NO_PASSES = 3

def predict(model, ref_data, valid_data, dist_func, feature_extractor=None):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    to_tensor = T.ToTensor()

    ref_data = ref_data.copy()
    id_to_vect = {}
    valid_data = valid_data.copy()
    for id_, row in ref_data.iterrows():
        if feature_extractor is not None:
             vect = model(**feature_extractor(to_tensor(row["img"]), return_tensors='pt'))
        else:
            vect = model(to_tensor(row["img"]))
                        
        id_to_vect[id_] = vect
    
    for id_, row in tqdm(valid_data.iterrows()):
        if feature_extractor is not None:
             vect = model(**feature_extractor(to_tensor(row["img"]), return_tensors='pt'))
        else:
            vect = model(to_tensor(row["img"]))
            
        best_row = next(iter(id_to_vect))
        min_dist = dist_func(id_to_vect[best_row], vect)
        
        for id_ref, vect_ref in id_to_vect.items():
            dist = dist_func(vect_ref, vect)
            if dist < min_dist:
                min_dist = dist
                best_row = id_ref
                                
            
        valid_data.loc[id_, "pred_category"] = ref_data.category_id[best_row]
        
    return valid_data

def predict_knn_zoo(model, ref_data, valid_data, k=3, feature_extractor=None, final=False):
    train_images = ref_data['img']
    train_labels = ref_data['category_id']
    train_dataset = SimpleDataset(train_images, train_labels, train_transforms)
    
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    BATCH_SIZE, 
    shuffle=True, 
    num_workers=0,
    pin_memory=False,
    )
    
    all_labels = []
    all_ys = []
    
    for _ in range(NO_PASSES):
        for i, data in enumerate(train_loader, 0):
            torch.cuda.empty_cache()
            images, labels = data
            images = images.to(device=DEVICE)
            images = [np.array(image.cpu(), dtype=np.uint8) for image in images]

            ys = model(**feature_extractor(images, return_tensors='pt'))

            all_ys.append(ys)
            all_labels.append(labels)

            
    ref_vects = all_ys
    
    for id_, row in tqdm(valid_data.iterrows()):
        if feature_extractor is not None:
            vect = model(**feature_extractor(to_tensor(row["img"]), return_tensors='pt'))
        else:
            vect = model(to_tensor(row["img"]))
            
        dists = ((ref_vects - vect)**2).sum(dim=1)
        k_nearest = dists.argsort()[:k]
        classes = [all_labels[idx] for idx in k_nearest]
        
        def most_common(lst):
            return max(set(lst), key=lst.count)
        
        class_ = most_common(classes)

                                
        if not final:
            valid_data.loc[id_, "pred_category"] = class_
        else:
            valid_data.loc[id_, "category_id"] = class_
        
    return valid_data
            
    
def predict_knn(model, ref_data, valid_data, k=3, feature_extractor=None, final=False):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    to_tensor = T.ToTensor()

    ref_data = ref_data.copy()
    id_to_vect = {}
    valid_data = valid_data.copy()
    for id_, row in ref_data.iterrows():
        if feature_extractor is not None:
            vect = model(**feature_extractor(to_tensor(row["img"]), return_tensors='pt'))
        else:
            vect = model(to_tensor(row["img"]))
                        
        id_to_vect[id_] = vect
        
    ref_idx_to_id = {}
    ref_vects = []
    idx = 0

    for key, vect in id_to_vect.items():
        ref_idx_to_id[idx] = key
        idx += 1
        ref_vects.append(vect)
        
    ref_vects = torch.cat(ref_vects, dim=0)
    # print(ref_vects.shape) # 144, 384
    
    for id_, row in tqdm(valid_data.iterrows()):
        if feature_extractor is not None:
            vect = model(**feature_extractor(to_tensor(row["img"]), return_tensors='pt'))
        else:
            vect = model(to_tensor(row["img"]))
            
        # print(vect.shape) # 1, 384
        dists = ((ref_vects - vect)**2).sum(dim=1)
        # print(dists.shape) # 144
        k_nearest = dists.argsort()[:k]
        classes = [ref_data.category_id[ref_idx_to_id[idx.item()]] for idx in k_nearest]
        
        def most_common(lst):
            return max(set(lst), key=lst.count)
        
        class_ = most_common(classes)

                                
        if not final:
            valid_data.loc[id_, "pred_category"] = class_
        else:
            valid_data.loc[id_, "category_id"] = class_
        
    return valid_data