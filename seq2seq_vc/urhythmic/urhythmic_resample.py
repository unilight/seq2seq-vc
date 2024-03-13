#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Resample dataset to a specific sampling rate.

   Based on Urhythmic: https://github.com/bshall/urhythmic
"""

import argparse
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resample_file(path, out_path, sample_rate):
    wav, sr = torchaudio.load(path)
    wav = AF.resample(wav, sr, sample_rate)
    torchaudio.save(out_path, wav, sample_rate, bits_per_sample=8)
    return wav.size(-1) / sample_rate


def resample_dataset(args):
    logger.info(f"Resampling dataset at {args.in_dir}")
    paths = list(args.in_dir.rglob("*.wav"))
    out_paths = [args.out_dir / path.relative_to(args.in_dir) for path in paths]

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(
            tqdm(
                executor.map(
                    resample_file, paths, out_paths, itertools.repeat(args.sample_rate)
                ),
                total=len(paths),
            )
        )
    logger.info(f"Processed {np.sum(results) / 60 / 60:4f} hours of audio.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample an audio dataset.")
    parser.add_argument(
        "--in_dir",
        metavar="in-dir",
        type=Path,
        help="dataset directory.",
    )
    parser.add_argument(
        "--out_dir",
        metavar="out-dir",
        type=Path,
        help="directory to save resampled dataset.",
    )
    parser.add_argument(
        "--sample_rate",
        help="target sample rate (defaults to 16000).",
        type=int,
        default=16000,
    )
    args = parser.parse_args()
    resample_dataset(args)
