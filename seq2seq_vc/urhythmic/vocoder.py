# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Urhythmic vocoder.

   Based on Urhythmic: https://github.com/bshall/urhythmic
"""

from typing import List, Tuple

# adapted from https://github.com/jik876/hifi-gan/blob/master/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq_vc.urhythmic.utils import get_padding
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class HifiganGenerator(torch.nn.Module):
    """HiFiGAN Generator. Converts speech units into an audio waveform."""

    def __init__(
        self,
        in_channels: int = 256,
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        upsample_kernel_sizes: Tuple[int, ...] = (20, 16, 4, 4),
        upsample_channels: int = 512,
        upsample_factors: Tuple[int, ...] = (10, 8, 2, 2),
        sample_rate: int = 16000,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            resblock_dilation_sizes (Tuple[Tuple[int, ...], ...]): list of dilation values in each layer of a `ResBlock`.
            resblock_kernel_sizes (Tuple[int, ...]): list of kernel sizes for each `ResBlock`.
            upsample_kernel_sizes (Tuple[int, ...]): list of kernel sizes for each transposed convolution.
            upsample_initial_channel (int): number of channels for the first upsampling layer. This is divided by 2
                for each consecutive upsampling layer.
            upsample_factors (Tuple[int, ...]): upsampling factors (stride) for each upsampling layer.
            sample_rate (int): sample rate of the generated audio.
        """
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.sample_rate = sample_rate

        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_channels, 5, 1, padding=2),
        )

        # upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_channels // (2**i),
                        upsample_channels // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_channels // (2 ** (i + 1))
            for _, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        # post convolution layer
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): soft speech units of shape (B, D, N) where B is the batch size, D is the unit dimensions, and N is the number of frames.
        """
        output = self.conv_pre(x)
        for i in range(self.num_upsamples):
            output = F.leaky_relu(output, LRELU_SLOPE)
            output = self.ups[i](output)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](output)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](output)
            output = z_sum / self.num_kernels
        output = F.leaky_relu(output)
        output = self.conv_post(output)
        output = torch.tanh(output)
        return output

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class PeriodDiscriminator(torch.nn.Module):
    """HiFiGAN Period Discriminator"""

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.period = period
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x (Tensor): input waveform.
        Returns:
            [Tensor]: discriminator scores per sample in the batch.
            [List[Tensor]]: list of features from each convolutional layer.
        """
        feat = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feat


class MultiPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN Multi-Period Discriminator (MPD)"""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(2),
                PeriodDiscriminator(3),
                PeriodDiscriminator(5),
                PeriodDiscriminator(7),
                PeriodDiscriminator(11),
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x (Tensor): input waveform.
        Returns:
            [List[Tensor]]: list of scores from each discriminator.
            [List[List[Tensor]]]: list of features from each discriminator's convolutional layers.
        """
        scores = []
        feats = []
        for _, d in enumerate(self.discriminators):
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats


class ScaleDiscriminator(torch.nn.Module):
    """HiFiGAN Scale Discriminator."""

    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x (Tensor): input waveform.
        Returns:
            Tensor: discriminator scores.
            List[Tensor]: list of features from the convolutional layers.
        """
        feat = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feat


class MultiScaleDiscriminator(torch.nn.Module):
    """HiFiGAN Multi-Scale Discriminator."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(use_spectral_norm=True),
                ScaleDiscriminator(),
                ScaleDiscriminator(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x (Tensor): input waveform.
        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of features from each discriminator's convolutional layers.
        """
        scores = []
        feats = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats


class HifiganDiscriminator(nn.Module):
    """HiFiGAN discriminator"""

    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x (Tensor): input waveform.
        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of features from from each discriminator's convolutional layers.
        """
        scores, feats = self.mpd(x)
        scores_, feats_ = self.msd(x)
        return scores + scores_, feats + feats_


def feature_loss(
    features_real: List[List[torch.Tensor]],
    features_generated: List[List[torch.Tensor]],
) -> float:
    loss = 0
    for r, g in zip(features_real, features_generated):
        for rl, gl in zip(r, g):
            loss += torch.mean(torch.abs(rl - gl))
    return loss


def discriminator_loss(
    real: List[torch.Tensor], generated: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[float], List[float]]:
    loss = 0
    real_losses = []
    generated_losses = []
    for r, g in zip(real, generated):
        r_loss = torch.mean((1 - r) ** 2)
        g_loss = torch.mean(g**2)
        loss += r_loss + g_loss
        real_losses.append(r_loss.item())
        generated_losses.append(g_loss.item())

    return loss, real_losses, generated_losses


def generator_loss(
    discriminator_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    loss = 0
    generator_losses = []
    for x in discriminator_outputs:
        l = torch.mean((1 - x) ** 2)
        generator_losses.append(l)
        loss += l

    return loss, generator_losses
