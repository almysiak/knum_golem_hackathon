import torch
from itertools import combinations
from pytorch_metric_learning import losses



 

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, eps, reduction="mean", pos_weight=1.):
        super().__init__()
        self.eps = eps
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, ys, labels):
        loss = torch.zeros(1).to(ys)
        for (y1, l1), (y2, l2) in combinations(zip(ys, labels), 2):
            sqr_dst = ((y1 - y2)**2).sum(dim=-1)
            loss += self.pos_weight * sqr_dst if l1 == l2 else torch.clip(self.eps - torch.sqrt(sqr_dst) , min=0) ** 2
        if self.reduction == "mean":
            return loss / (len(labels) * (len(labels) - 1) / 2)
        else:
            return loss
        
        
class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", pos_weight=1.):
        super().__init__()
        self.pos_weight = pos_weight
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=eps)
        self.reduction = reduction
        
    def forward(self, ys, labels):
        loss = torch.zeros(1).to(ys)
        for (y1, l1), (y2, l2) in combinations(zip(ys, labels), 2):
            cs = self.cos_sim(y1, y2)
            loss += cs * (-self.pos_weight if l1 == l2 else 1)
        if self.reduction == "mean":
            return loss / (len(labels) * (len(labels) - 1) / 2)
        else:
            return loss


# class ContrastiveLoss(torch.nn.Module):
#     def __init__(self, eps, reduction="mean"):
#         super().__init__()
#         self.eps = eps
#         self.reduction = reduction
        
#     def forward(self, y_1, y_2, sign, weight=None):
#         same = (sign + 1) / 2
#         sqr_dst = ((y_1 - y_2)**2).sum(dim=-1)
#         loss = same * sqr_dst + (1 - same) * torch.clip(self.eps - torch.sqrt(sqr_dst) , min=0) ** 2
#         if weight is not None:
#             loss = loss * weight
#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         return loss
    
# class CosineSimilarityLoss(torch.nn.Module):
#     def __init__(self, eps=1e-6, reduction="mean"):
#         super().__init__()
#         self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=eps)
#         self.reduction = reduction
        
#     def forward(self, y_1, y_2, sign, weight=None):
#         cs = self.cos_sim(y_1, y_2)
#         loss = - cs * sign
#         if weight is not None:
#             loss = loss * weight
#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         return loss