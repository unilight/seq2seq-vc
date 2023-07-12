import torch
from seq2seq_vc.layers.utils import make_non_pad_mask


class DurationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, use_masking=True, offset=1.0, reduction="mean"):
        """Initilize duration predictor loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset
        self.use_masking = use_masking

    def forward(self, d_outs, ds, ilens):
        """Calculate forward propagation.

        Args:
            d_outs (Tensor): Batch of outputs of duration predictor (B, Tmax).
            ds (Tensor): Batch of durations (B, Tmax).
            ilens (LongTensor): Batch of the lengths of each input (B,).

        Returns:
            Tensor: Duration predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(ds.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)

        # calculate loss
        # NOTE: outputs is in log domain while targets in linear
        ds = torch.log(ds.float() + self.offset)
        duration_loss = self.criterion(d_outs, ds)

        return duration_loss
