from ipynb.fs.full.read_in_data import read_in_data
import json
import torch
from torchvision.transforms import transforms as T

def predict(model, ref_data, valid_data, feature_extractor=None):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    to_tensor = T.ToTensor()

    ref_data = ref_data.copy()
    valid_data = valid_data.copy()
    for id_, row in ref_data.iterrows():
        if feature_extractor is not None:
             vect = model(**feature_extractor(to_tensor(row["img"]), return_tensors='pt'))
        else:
            vect = model(to_tensor(row["img"]))
            
        print(type(vect))
            
        ref_data.loc[id_, "vect"] = vect.detach().numpy()
    
    for id_, row in valid_data.iterrows():
        if feature_extractor is not None:
             vect = model(**feature_extractor(to_tensor(row["img"]), return_tensors='pt'))
        else:
            vect = model(to_tensor(row["img"]))
            
        best_row = ref_data.id.loc[0]
        min_dist = cos(ref_data.vect.loc[0], vect)
        
        for id_ref, row_ref in ref_data.iterrows():
            dist = cos(ref_data.vect.loc[id_ref], vect)
            if dist < min_dist:
                min_dist = dist
                best_row = id_ref
                
            
        valid_data.loc[id_, "pred_category"] = ref_data.category_id[best_row]
        
    return valid_data
            