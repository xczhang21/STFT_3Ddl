import torch.nn as nn

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        return ((pred - target) ** 2 * mask).sum() / mask.sum()
