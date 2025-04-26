import torch
import torch.nn as nn


class MaskedMSEMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        return ((pred - target) ** 2 * mask).sum() / mask.sum()


class MaskedMAEMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        return ((pred - target).abs() * mask).sum() / mask.sum()


class MaskedRMSEMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        error_sq = ((pred - target) ** 2 * mask).sum()
        return error_sq.sqrt() / mask.sum().sqrt()
