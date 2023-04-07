import torch

class DurationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor."""

    def forward(self, logw, logw_, lengths):
        return torch.sum((logw - logw_)**2) / torch.sum(lengths)
        