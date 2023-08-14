#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Duration predictor related modules."""

import torch
import torch.nn.functional as F

from seq2seq_vc.modules.transformer.layer_norm import LayerNorm

import math
from typing import Optional


from seq2seq_vc.modules.vits.flow import (
    ConvFlow,
    DilatedDepthSeparableConv,
    ElementwiseAffineFlow,
    FlipFlow,
    LogFlow,
)


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.

    This is a module of duration predictor described
    in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Note:
        The calculation domain of outputs is different
        between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`,
        those are calculated in linear domain.

    """

    def __init__(
        self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)

        if is_inference:
            # # NOTE: calculate in linear domain
            xs = torch.clamp(
                torch.round(xs.exp() - self.offset), min=0
            ).long()  # avoid negative value

        # if x_masks is not None:
        # xs = xs.masked_fill(x_masks, 0.0)
        if x_masks is not None:
            xs = xs * x_masks

        return xs

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(xs, x_masks, True)


class StochasticDurationPredictor(torch.nn.Module):
    """Stochastic duration predictor module.

    This is a module of stochastic duration predictor described in `Conditional
    Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech`_.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        channels: int = 192,
        kernel_size: int = 3,
        dropout_rate: float = 0.5,
        flows: int = 4,
        dds_conv_layers: int = 3,
        global_channels: int = -1,
    ):
        """Initialize StochasticDurationPredictor module.

        Args:
            channels (int): Number of channels.
            kernel_size (int): Kernel size.
            dropout_rate (float): Dropout rate.
            flows (int): Number of flows.
            dds_conv_layers (int): Number of conv layers in DDS conv.
            global_channels (int): Number of global conditioning channels.

        """
        super().__init__()

        self.pre = torch.nn.Conv1d(channels, channels, 1)
        self.dds = DilatedDepthSeparableConv(
            channels,
            kernel_size,
            layers=dds_conv_layers,
            dropout_rate=dropout_rate,
        )
        self.proj = torch.nn.Conv1d(channels, channels, 1)

        self.log_flow = LogFlow()
        self.flows = torch.nn.ModuleList()
        self.flows += [ElementwiseAffineFlow(2)]
        for i in range(flows):
            self.flows += [
                ConvFlow(
                    2,
                    channels,
                    kernel_size,
                    layers=dds_conv_layers,
                )
            ]
            self.flows += [FlipFlow()]

        self.post_pre = torch.nn.Conv1d(1, channels, 1)
        self.post_dds = DilatedDepthSeparableConv(
            channels,
            kernel_size,
            layers=dds_conv_layers,
            dropout_rate=dropout_rate,
        )
        self.post_proj = torch.nn.Conv1d(channels, channels, 1)
        self.post_flows = torch.nn.ModuleList()
        self.post_flows += [ElementwiseAffineFlow(2)]
        for i in range(flows):
            self.post_flows += [
                ConvFlow(
                    2,
                    channels,
                    kernel_size,
                    layers=dds_conv_layers,
                )
            ]
            self.post_flows += [FlipFlow()]

        if global_channels > 0:
            self.global_conv = torch.nn.Conv1d(global_channels, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        inverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T_text).
            x_mask (Tensor): Mask tensor (B, 1, T_text).
            w (Optional[Tensor]): Duration tensor (B, 1, T_text).
            g (Optional[Tensor]): Global conditioning tensor (B, channels, 1)
            inverse (bool): Whether to inverse the flow.
            noise_scale (float): Noise scale value.

        Returns:
            Tensor: If not inverse, negative log-likelihood (NLL) tensor (B,).
                If inverse, log-duration tensor (B, 1, T_text).

        """
        x = x.detach()  # stop gradient
        x = self.pre(x)
        if g is not None:
            x = x + self.global_conv(g.detach())  # stop gradient
        x = self.dds(x, x_mask)
        x = self.proj(x) * x_mask

        if not inverse:
            assert w is not None, "w must be provided."
            h_w = self.post_pre(w)
            h_w = self.post_dds(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(
                    w.size(0),
                    2,
                    w.size(2),
                ).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            logdet_tot_q = 0.0
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in self.flows:
                z, logdet = flow(z, x_mask, g=x, inverse=inverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # (B,)
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(
                    x.size(0),
                    2,
                    x.size(2),
                ).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, inverse=inverse)
            z0, z1 = z.split(1, 1)
            logw = z0

            w = torch.exp(logw) * x_mask

            dur = torch.ceil(w)
            # dur = torch.clamp(torch.round(w), min=0)
            return dur
            # return logw
