import torch
import torch.nn as nn
import torch.nn.functional as F


class NonWeightedBCELogitsLoss(nn.Module):
    def __init__(self):
        super(NonWeightedBCELogitsLoss, self,).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.bce(output, target)
