#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Extract Soft Speech Units and Log Probabilities.

   Based on Urhythmic: https://github.com/bshall/urhythmic
"""

import argparse
import logging
import os

import librosa
import numpy as np
import torch
import yaml
from seq2seq_vc.datasets import AudioSCPDataset
from seq2seq_vc.urhythmic.model import encode
from tqdm import tqdm

SAMPLING_RATE = 16000


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio and then extract features (See detail in"
            " bin/preprocess.py)."
        )
    )
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
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.",
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

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)
    if not os.path.exists(os.path.join(args.dumpdir, "soft")):
        os.makedirs(os.path.join(args.dumpdir, "soft"), exist_ok=True)
    if not os.path.exists(os.path.join(args.dumpdir, "logprobs")):
        os.makedirs(os.path.join(args.dumpdir, "logprobs"), exist_ok=True)

    # load upstream extractor
    device = torch.device("cpu")
    logging.info("Loading hubert_soft checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").to(device)

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

        # actual encoding
        with torch.inference_mode():
            units, log_probs = encode(
                hubert, torch.tensor(audio).unsqueeze(0).unsqueeze(0)
            )

        # save softs
        units_out_path = os.path.join(args.dumpdir, "soft", f"{utt_id}.npy")
        np.save(units_out_path, units.squeeze().cpu().numpy())

        # save logprobs
        probs_out_path = os.path.join(args.dumpdir, "logprobs", f"{utt_id}.npy")
        np.save(probs_out_path, log_probs.squeeze().cpu().numpy())


if __name__ == "__main__":
    main()
