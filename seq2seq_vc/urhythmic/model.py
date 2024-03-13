#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Urhythmic model related

   Based on Urhythmic: https://github.com/bshall/urhythmic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq_vc.urhythmic.rhythm_model import RhythmModelFineGrained
from seq2seq_vc.urhythmic.segmenter import Segmenter
from seq2seq_vc.urhythmic.stretcher import TimeStretcherFineGrained
from seq2seq_vc.urhythmic.vocoder import HifiganGenerator


@torch.inference_mode()
def encode(hubert, wav):
    r"""Encode an audio waveform into soft speech units and the log probabilities of the associated discrete units.

    Args:
        hubert (HubertSoft): the HubertSoft content encoder.
        wav (Tensor): an audio waveform of shape (B, 1, T) where B is the batch size and T is the number of samples.

    Returns:
        Tensor: soft speech units of shape (B, D, N) where N is the number of frames, and D is the unit dimensions.
        Tensor: the predicted log probabilities over the discrete units of shape (B, N, K) where K is the number of discrete units.
    """
    units = hubert.units(wav)
    logits = hubert.logits(units)
    log_probs = F.log_softmax(logits, dim=-1)
    return units.transpose(1, 2), log_probs


class UrhythmicFine(nn.Module):
    """Urhythmic (Fine-Grained), a voice and rhythm conversion system that does not require text or parallel data."""

    def __init__(
        self,
        segmenter: Segmenter,
        rhythm_model: RhythmModelFineGrained,
        time_stretcher: TimeStretcherFineGrained,
        vocoder: HifiganGenerator,
    ):
        """
        Args:
            segmenter (Segmenter): the segmentation and clustering block groups similar units into short segments.
                The segments are then combined into coarser groups approximating sonorants, obstruents, and silences.
            rhythm_model (RhythmModelFineGrained): the rhythm modeling block estimates the duration distribution of each group.
            time_stretcher (TimeStretcherFineGrained): the time-stretching block down/up-samples the speech units to match the target rhythm.
            vocoder (HifiganGenerator): the vocoder converts the speech units into an audio waveform.
        """
        super().__init__()
        self.segmenter = segmenter
        self.rhythm_model = rhythm_model
        self.time_stretcher = time_stretcher
        self.vocoder = vocoder

    @torch.inference_mode()
    def forward(self, units: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """Convert the to the target speaker's voice and rhythm

        Args:
            units (Tensor): soft speech units of shape (1, D, N) where D is the unit dimensions and N is the number of frames.
            log_probs (Tensor): the predicted log probabilities over the discrete units of shape (1, N, K) where K is the number of discrete units.

        Returns:
            Tensor: the converted waveform of shape (1, 1, T) where T is the number of samples.
        """
        clusters, boundaries = self.segmenter(log_probs.squeeze().cpu().numpy())
        tgt_durations = self.rhythm_model(clusters, boundaries)
        units = self.time_stretcher(units, clusters, boundaries, tgt_durations)
        wav = self.vocoder(units)
        return wav
