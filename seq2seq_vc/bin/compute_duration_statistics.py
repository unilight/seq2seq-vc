#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from pathlib import Path

import librosa
import matplotlib
import numpy as np

from seq2seq_vc.utils import find_files

matplotlib.use("Agg")  # noqa #isort:skip
import matplotlib.pyplot as plt  # noqa isort:skip


def main():
    dcp = "Calculate duration statistics given wav directory or scp file"
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument("--n_jobs", type=int, default=16, help="# of CPUs")
    parser.add_argument(
        "--wav_dir", type=str, default=None, help="Directory of wav file"
    )
    parser.add_argument(
        "--scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file.",
    )
    parser.add_argument("--figure_dir", type=str, help="Directory for figure output")
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    histogrampath = os.path.join(args.figure_dir, "duration_histogram.png")

    if not os.path.exists(histogrampath):

        # sanity check
        assert (args.scp is None and args.wav_dir is not None) or (
            args.scp is not None and args.wav_dir is None
        ), "Please assure only either --scp or --wav_dir is specified."

        # get file list
        if args.scp is not None:
            with open(args.scp, "r") as f:
                file_list = [line.split(" ")[1] for line in f.read().splitlines()]
        else:
            file_list = sorted(find_files(args.wav_dir))

        durs = np.array([librosa.get_duration(path=f) for f in file_list])

        logging.info(f"[Info] Total durations: {np.sum(durs)}")
        logging.info(f"[Info] Mean durations: {np.mean(durs)}")
        logging.info(f"[Info] Max duration: {np.max(durs)}")
        logging.info(f"[Info] Min duration: {np.min(durs)}")

        # plot histgram
        plt.hist(
            durs,
            bins=200,
            range=(0, int(np.max(durs)+1)),
            histtype="stepfilled",
        )
        plt.xlabel("Duration (sec)")
        plt.xticks(np.arange(0, int(np.max(durs)+1), 1))

        if not os.path.exists(args.figure_dir):
            os.makedirs(args.figure_dir)
        plt.savefig(histogrampath)
        plt.close()


if __name__ == "__main__":
    main()
