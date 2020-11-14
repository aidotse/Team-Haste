import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskMedianMAELoss(nn.Module):
    def __init__(self):
        super(MaskMedianMAELoss, self,).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, target, mask):
        loss =  self.l1(output*mask, target*mask) 
        loss = loss/ torch.median(target*mask)
        return loss
