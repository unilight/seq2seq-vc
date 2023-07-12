#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import numpy as np
import torch
import torch.nn.functional as F


class NARVCCollater(object):
    """Customized collater for Pytorch DataLoader in non-autoregressive VC training."""

    def __init__(self):
        """Initialize customized collater for PyTorch DataLoader."""

    def __call__(self, batch):
        """Convert into batch tensors."""

        def pad_list(xs, pad_value):
            """Perform padding for the list of tensors.

            Args:
                xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
                pad_value (float): Value for padding.

            Returns:
                Tensor: Padded tensor (B, Tmax, `*`).

            Examples:
                >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
                >>> x
                [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
                >>> pad_list(x, 0)
                tensor([[1., 1., 1., 1.],
                        [1., 1., 0., 0.],
                        [1., 0., 0., 0.]])

            """
            n_batch = len(xs)
            max_len = max(x.size(0) for x in xs)
            pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

            for i in range(n_batch):
                pad[i, : xs[i].size(0)] = xs[i]

            return pad

        xs = []
        ys = []
        dp_inputs = []

        for b in batch:
            xs.append(b["src_feat"])
            ys.append(b["trg_feat"])
            dp_inputs.append(b["dp_input"])

        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long()
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long()
        dplens = torch.from_numpy(np.array([dp.shape[0] for dp in dp_inputs])).long()

        # perform padding and conversion to tensor
        xs = pad_list([torch.from_numpy(x).float() for x in xs], 0)
        ys = pad_list([torch.from_numpy(y).float() for y in ys], 0)
        dp_inputs = pad_list(
            [torch.from_numpy(dp_input).float() for dp_input in dp_inputs], 0
        )

        items = {
            "xs": xs,
            "ilens": ilens,
            "ys": ys,
            "olens": olens,
            "dp_inputs": dp_inputs,
            "dplens": dplens,
            "spembs": None,
        }

        # get duration if exists
        if "duration" in batch[0]:
            durations = [b["duration"] for b in batch]
            durations = pad_list([torch.from_numpy(d).long() for d in durations], 0)
            duration_lens = torch.from_numpy(
                np.array([d.shape[0] for d in durations])
            ).long()
            items["durations"] = durations
            items["duration_lens"] = duration_lens

        return items
