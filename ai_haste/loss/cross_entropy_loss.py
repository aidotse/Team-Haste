import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.2]):
        super(WeightedCrossEntropyLoss, self,).__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())

    def forward(self, output, target):
        return self.ce(output, target)
