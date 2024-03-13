#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform segmentation.

   Based on Urhythmic: https://github.com/bshall/urhythmic
"""

import argparse
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def segment_file(segmenter, in_path, out_path):
    log_probs = np.load(in_path)
    segments, boundaries = segmenter(log_probs)
    np.savez(out_path.with_suffix(".npz"), segments=segments, boundaries=boundaries)
    return log_probs.shape[0], np.mean(np.diff(boundaries))


def main():
    """Run segmentation process."""
    parser = argparse.ArgumentParser(description=("Segment all samples."))
    parser.add_argument(
        "--data_dir",
        required=True,
        type=Path,
        help="data directory containing the log probabilities.",
    )
    parser.add_argument(
        "--dumpdir",
        type=Path,
        required=True,
        help="directory to dump results.",
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

    # Load segmenter
    logging.info("Loading segmenter checkpoint")
    segmenter = torch.hub.load("bshall/urhythmic:main", "segmenter", num_clusters=3)

    # Get all log prob files
    log_probs_paths = list(args.data_dir.rglob("*.npy"))

    # Setup output dir
    logging.info("Setting up output folder structure")
    out_paths = [
        args.dumpdir / path.relative_to(args.data_dir) for path in log_probs_paths
    ]

    # Actual segmentation
    logging.info("Segmenting dataset")
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(
            tqdm(
                executor.map(
                    segment_file,
                    itertools.repeat(segmenter),
                    log_probs_paths,
                    out_paths,
                ),
                total=len(log_probs_paths),
            )
        )

    frames, boundary_length = zip(*results)
    logging.info(f"Segmented {sum(frames) * 0.02 / 60 / 60:.2f} hours of audio")
    logging.info(
        f"Average segment length: {np.mean(boundary_length) * 0.02:.4f} seconds"
    )


if __name__ == "__main__":
    main()
