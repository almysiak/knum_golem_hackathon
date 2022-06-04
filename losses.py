import torch

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, eps, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, y_1, y_2, sign, weight=None):
        same = (sign + 1) // 2
        sqr_dst = ((y_1 - y_2)**2).sum(dim=-1)
        loss = same * sqr_dst + (1 - same) * torch.clip(self.eps - torch.sqrt(sqr_dst) , min=0) ** 2
        if weight is not None:
            loss = loss * weight
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
    
class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=eps)
        self.reduction = reduction
        
    def forward(self, y_1, y_2, sign, weight=None):
        cs = self.cos_sim(y_1, y_2)
        loss = - cs * sign
        if weight is not None:
            loss = loss * weight
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss