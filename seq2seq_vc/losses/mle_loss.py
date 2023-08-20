import math
import torch


class MLELoss(torch.nn.Module):
    """Loss function module for flow-based models."""

    def forward(self, z, m, logs, logdet, mask):
        l = torch.sum(logs) + 0.5 * torch.sum(
            torch.exp(-2 * logs) * ((z - m) ** 2)
        )  # neg normal likelihood w/o the constant term
        l = l - torch.sum(logdet)  # log jacobian determinant
        l = l / torch.sum(
            torch.ones_like(z) * mask
        )  # averaging across batch, channel and time axes
        l = l + 0.5 * math.log(2 * math.pi)  # add the remaining constant term
        return l
