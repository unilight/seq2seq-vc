import torch
from seq2seq_vc.layers.utils import make_non_pad_mask


class DiffSingerL2Loss(torch.nn.Module):
    """L1 Loss function module (for feed-forward Transformer)"""

    def __init__(self, use_masking=True, reduction="mean"):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.

        """
        super(DiffSingerL2Loss, self).__init__()
        self.use_masking = use_masking

        # define criterion
        self.l2_criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, noise, x_recon, olens):
        """Calculate forward propagation.

        Args:
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(x_recon.device)
            noise = noise.masked_select(masks)
            x_recon = x_recon.masked_select(masks)

        # calculate loss
        l2_loss = self.l2_criterion(noise, x_recon)

        return l2_loss
