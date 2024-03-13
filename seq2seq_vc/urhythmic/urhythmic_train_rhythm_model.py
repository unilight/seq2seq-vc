#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Train rhythm model.

   Based on Urhythmic: https://github.com/bshall/urhythmic
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from seq2seq_vc.urhythmic.rhythm_model import RhythmModelFineGrained
from tqdm import tqdm

HOP_LENGTH = 320
SAMPLE_RATE = 16000


def main():
    """Run segmentation process."""
    parser = argparse.ArgumentParser(description=("Train the rhythm model."))
    parser.add_argument(
        "--data_dir",
        required=True,
        type=Path,
        help="data directory containing the segment results.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Path to save the trained rhythm model.",
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

    # Setup rhythm model
    rhythm_model = RhythmModelFineGrained(
        hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE
    )

    # Load segmenter to avoid no module error when loading segments
    logging.info("Loading segmenter checkpoint")
    segmenter = torch.hub.load("bshall/urhythmic:main", "segmenter", num_clusters=3)

    # Load segments
    utterances = []
    for path in tqdm(list(args.data_dir.rglob("*.npz"))):
        file = np.load(path, allow_pickle=True)
        segments = list(file["segments"])
        boundaries = list(file["boundaries"])
        utterances.append((segments, boundaries))

    # Train rhythm model
    logging.info("Training rhythm model.")
    dists = rhythm_model._fit(utterances)

    # Save model
    logging.info(f"Saving checkpoint to {args.checkpoint_path}")
    torch.save(dists, args.checkpoint_path)


if __name__ == "__main__":
    main()
