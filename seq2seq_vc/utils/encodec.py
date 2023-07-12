#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Feature extraction with Encodec."""

from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch


def get_encodec_model():
    return EncodecModel.encodec_model_24khz()


def encodec_encode(wav, model):
    # Load and pre-process the audio waveform
    wav = wav.unsqueeze(0)

    # Set variables
    length = wav.shape[2]
    segment_length = model.segment_length
    if segment_length is None:
        # only call encode_frame once
        segment_length = length
        stride = length
    else:
        stride = model.segment_stride  # type: ignore
        assert stride is not None

    def encode_frame(x):
        length = x.shape[-1]
        duration = length / model.sample_rate
        assert model.segment is None or duration <= 1e-5 + model.segment

        emb = model.encoder(x)
        return emb

    # Extract discrete codes from EnCodec
    encoded_frames = []
    with torch.no_grad():
        for offset in range(0, length, stride):
            frame = wav[:, :, offset : offset + segment_length]
            encoded_frames.append(encode_frame(frame))  # [1, 128, T]
        return encoded_frames
