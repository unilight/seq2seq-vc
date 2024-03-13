#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Conversion.

   Based on Urhythmic: https://github.com/bshall/urhythmic
"""

import argparse
import logging
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from seq2seq_vc.datasets import AudioSCPDataset
from seq2seq_vc.urhythmic.model import UrhythmicFine, encode
from seq2seq_vc.urhythmic.rhythm_model import RhythmModelFineGrained
from seq2seq_vc.urhythmic.stretcher import TimeStretcherFineGrained
from seq2seq_vc.urhythmic.vocoder import HifiganGenerator
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from tqdm import tqdm

SAMPLING_RATE = 16000


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(description=("Conversion."))
    parser.add_argument(
        "--wav-scp",
        "--scp",
        required=True,
        type=str,
        help="kaldi-style wav.scp file.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help=(
            "kaldi-style segments file. if use, you must to specify both scp and"
            " segments."
        ),
    )
    parser.add_argument(
        "--src_rhythm_model_path",
        type=str,
        required=True,
        help="path to the source rhythm model.",
    )
    parser.add_argument(
        "--trg_rhythm_model_path",
        type=str,
        required=True,
        help="path to the target rhythm model.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        required=True,
        help="path to the vocoder.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save results.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    dataset = AudioSCPDataset(
        args.wav_scp,
        segments=args.segments,
        return_utt_id=True,
        return_sampling_rate=True,
    )

    # Load hubert soft model
    logging.info("Loading hubert_soft checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

    # Load segmenter
    logging.info("Loading segmenter checkpoint")
    segmenter = torch.hub.load("bshall/urhythmic:main", "segmenter", num_clusters=3)

    # Load rhythm model
    rhythm_model = RhythmModelFineGrained()
    rhythm_model.load_state_dict(
        {
            "source": torch.load(args.src_rhythm_model_path),
            "target": torch.load(args.trg_rhythm_model_path),
        }
    )

    # Initialize time stretcher instance
    time_stretcher = TimeStretcherFineGrained()

    # Load vocoder
    hifigan = HifiganGenerator()
    hifigan_checkpoint = torch.load(args.vocoder_path, map_location=torch.device("cpu"))
    consume_prefix_in_state_dict_if_present(
        hifigan_checkpoint["generator"]["model"], "module."
    )
    hifigan.load_state_dict(hifigan_checkpoint["generator"]["model"])
    hifigan.eval()
    hifigan.remove_weight_norm()

    # Assemble!
    urhythmic_fine = UrhythmicFine(segmenter, rhythm_model, time_stretcher, hifigan)

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."

        # resample to 16kHz if not (required by )
        if fs != SAMPLING_RATE:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=SAMPLING_RATE)

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        with torch.inference_mode():
            # encode
            units, log_probs = encode(
                hubert, torch.tensor(audio).unsqueeze(0).unsqueeze(0)
            )

            # conversion
            converted_wav = urhythmic_fine(units, log_probs)

        sf.write(
            os.path.join(args.outdir, "wav", f"{utt_id}.wav"),
            converted_wav.squeeze().cpu().numpy(),
            fs,
            "PCM_16",
        )


if __name__ == "__main__":
    main()
