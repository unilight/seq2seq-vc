#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from pathlib import Path
from joblib import Parallel, delayed

import kaldiio
import librosa
import matplotlib
import numpy as np

from seq2seq_vc.utils.signal import world_extract
from seq2seq_vc.utils import find_files

matplotlib.use("Agg")  # noqa #isort:skip
import matplotlib.pyplot as plt  # noqa isort:skip


def create_histogram(
    data, figure_path, range_min=-70, range_max=20, step=10, xlabel="Power [dB]"
):
    """Create histogram

    Parameters
    ----------
    data : list,
        List of several data sequences
    figure_path : str,
        Filepath to be output figure
    range_min : int, optional,
        Minimum range for histogram
        Default set to -70
    range_max : int, optional,
        Maximum range for histogram
        Default set to -20
    step : int, optional
        Stap size of label in horizontal axis
        Default set to 10
    xlabel : str, optional
        Label of the horizontal axis
        Default set to 'Power [dB]'

    """

    # plot histgram
    plt.hist(
        data,
        bins=200,
        range=(range_min, range_max),
        density=True,
        histtype="stepfilled",
    )
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xticks(np.arange(range_min, range_max, step))

    figure_dir = os.path.dirname(figure_path)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(figure_path)
    plt.close()


def extract_f0_and_npow(wavf=None, utt_id=None, audio_loader=None, f0min=40, f0max=500):
    """
    F0 and npow extraction

    Parameters
    ----------
    wavf : str,
        File path of waveform file

    Returns
    -------
    dict :
        Dictionary consisting of F0 and npow arrays

    """
    assert (wavf is not None and (utt_id is None and audio_loader is None)) or (wavf is None and (utt_id is not None and audio_loader is not None))

    if wavf is not None:
        x, fs = librosa.load(wavf, sr=None)
    else:
        fs, x = audio_loader[utt_id]
    return world_extract(x, fs, f0min, f0max)


def main():
    dcp = "Create histogram for speaker-dependent configure"
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

    f0histogrampath = os.path.join(args.figure_dir, "f0histogram.png")
    npowhistogrampath = os.path.join(args.figure_dir, "npowhistogram.png")

    if not os.path.exists(f0histogrampath) and not os.path.exists(npowhistogrampath):

        # sanity check
        assert (args.scp is None and args.wav_dir is not None) or (
            args.scp is not None and args.wav_dir is None
        ), "Please assure only either --scp or --wav_dir is specified."

        # get file list
        if args.scp is not None:
            audio_loader = kaldiio.load_scp(args.scp)
            audio_keys = list(audio_loader.keys())

            # extract features in parallel
            results = Parallel(n_jobs=args.n_jobs)(
                [delayed(extract_f0_and_npow)(None, utt_id, audio_loader) for utt_id in audio_keys]
            )
        else:
            file_list = sorted(find_files(args.wav_dir))

            # extract features in parallel
            results = Parallel(n_jobs=args.n_jobs)(
                [delayed(extract_f0_and_npow)(str(f)) for f in file_list]
            )

        # parse results
        f0s = [r["f0"] for r in results]
        npows = [r["npow"] for r in results]

        # stack feature vectors
        f0s = np.hstack(f0s).flatten()
        npows = np.hstack(npows).flatten()

        # create a histogram to visualize F0 range of the speaker
        create_histogram(
            f0s,
            f0histogrampath,
            range_min=40,
            range_max=700,
            step=50,
            xlabel="Fundamental frequency [Hz]",
        )

        # create a histogram to visualize npow range of the speaker
        create_histogram(
            npows,
            npowhistogrampath,
            range_min=-70,
            range_max=20,
            step=10,
            xlabel="Frame power [dB]",
        )


if __name__ == "__main__":
    main()
