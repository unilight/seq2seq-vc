# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import logging

import numpy as np
import torch
import torch.nn.functional as F

from seq2seq_vc.layers.utils import make_non_pad_mask

class Seq2SeqLoss(torch.nn.Module):
    """Loss function module for seq2seq VC."""

    def __init__(self, bce_pos_weight=5.0):
        """
        Args:
            bce_pos_weight (float): Weight of positive sample of stop token.
        """
        super(Seq2SeqLoss, self).__init__()

        # define criterions
        reduction = "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=torch.tensor(bce_pos_weight)
        )

    def forward(self, after_outs, before_outs, logits, ys, labels, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            logits (Tensor): Batch of stop logits (B, Lmax).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            labels (LongTensor): Batch of the sequences of stop token labels (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
            Tensor: Binary cross entropy loss value.

        """
        # make mask and apply it
        masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
        ys = ys.masked_select(masks)
        after_outs = after_outs.masked_select(masks)
        before_outs = before_outs.masked_select(masks)
        labels = labels.masked_select(masks[:, :, 0])
        logits = logits.masked_select(masks[:, :, 0])

        # calculate loss
        l1_loss = self.l1_criterion(after_outs, ys) + self.l1_criterion(before_outs, ys)
        bce_loss = self.bce_criterion(logits, labels)

        return l1_loss, bce_loss
