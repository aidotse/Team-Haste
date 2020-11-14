import torch
import torch.nn as nn
import torch.nn.functional as F


class LupiMaskLoss(nn.Module):
    def __init__(self,):
        super(LupiMaskLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.eps = 1e-8

    def forward(self, output, target, mask):
        # Weight accuracy within mask extra
        loss = torch.log(torch.div(((target*mask) + self.eps), ((output*mask) + self.eps)) )
        loss = (loss ** 2).mean()
        return loss