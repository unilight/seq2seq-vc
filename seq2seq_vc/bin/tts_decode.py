#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained TTS model."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

import seq2seq_vc.models
from seq2seq_vc.datasets.tts_dataset import TTSDataset
from seq2seq_vc.utils import read_hdf5
from seq2seq_vc.utils.plot import plot_attention, plot_generated_and_ref_2d, plot_1d
from seq2seq_vc.vocoder import Vocoder
from seq2seq_vc.vocoder.griffin_lim import Spectrogram2Waveform

def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode with trained TTS model "
            "(See detail in bin/tts_decode.py)."
        )
    )
    parser.add_argument(
        "--feats-scp",
        "--scp",
        default=None,
        type=str,
        help=(
            "kaldi-style feats.scp file. "
            "you need to specify either feats-scp or dumpdir."
        ),
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        help=(
            "directory including feature files. "
            "you need to specify either feats-scp or dumpdir."
        ),
    )
    parser.add_argument(
        "--text",
        required=True,
        type=str,
        help=(
            "raw input text file. "
        ),
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="stats file for target denormalization.",
    )
    parser.add_argument(
        "--token-list",
        type=str,
        required=True,
        help="a text mapping int-id to token",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help=(
            "yaml format configuration file. if not explicitly provided, "
            "it will be searched in the checkpoint directory. (default=None)"
        ),
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

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # load target stats for denormalization
    config["stats"] = {
        "mean": read_hdf5(args.stats, "mean"),
        "scale": read_hdf5(args.stats, "scale")
    }

    # check arguments
    if (args.feats_scp is not None and args.dumpdir is not None) or (
        args.feats_scp is None and args.dumpdir is None
    ):
        raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # get dataset
    if args.dumpdir is not None:
        mel_query = "*.h5"
        mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
        dataset = TTSDataset(
            root_dir=args.dumpdir,
            text_path=args.text,
            non_linguistic_symbols=config["non_linguistic_symbols"],
            cleaner=config["cleaner"],
            g2p=config["g2p"],
            token_list=args.token_list,
            token_type=config["token_type"],
            mel_query=mel_query,
            mel_load_fn=mel_load_fn,
            allow_cache=config.get("allow_cache", False),  # keep compatibility
            return_utt_id=True,
        )
    else:
        raise NotImplementedError
        dataset = MelSCPDataset(
            feats_scp=args.feats_scp,
            return_utt_id=True,
        )
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # get model and load parameters
    model_class = getattr(
        seq2seq_vc.models,
        config["model_type"]
    )
    model = model_class(**config["model_params"])
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["model"]
    )
    model = model.eval().to(device)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")

    # load vocoder
    if config.get("vocoder", False):
        vocoder = Vocoder(
            config["vocoder"]["checkpoint"],
            config["vocoder"]["config"],
            config["vocoder"]["stats"],
            config["trg_stats"],
            device
        )
    else:
        vocoder = Spectrogram2Waveform(
            stats=config["stats"],
            n_fft=config["fft_size"],
            n_shift=config["hop_size"],
            fs=config["sampling_rate"],
            n_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            griffin_lim_iters=64
        )

    # start generation
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for _, (utt_id, x, _, _) in enumerate(pbar, 1):
            x = torch.tensor(x, dtype=torch.long).to(device)
            start_time = time.time()
            outs, probs, att_ws = model.inference(x, config["inference"], spemb=None)
            logging.info(
                "inference speed = %.1f frames / sec."
                % (int(outs.size(0)) / (time.time() - start_time))
            )

            plot_generated_and_ref_2d(
                outs.cpu().numpy(),
                config["outdir"] + f"/outs/{utt_id}.png",
                origin="lower"
            )
            plot_1d(
                probs.cpu().numpy(),
                config["outdir"] + f"/probs/{utt_id}_prob.png",
            )
            plot_attention(
                att_ws.cpu().numpy(),
                config["outdir"] + f"/att_ws/{utt_id}_att_ws.png",
            )

            if not os.path.exists(os.path.join(config["outdir"], "wav")):
                os.makedirs(os.path.join(config["outdir"], "wav"), exist_ok=True)

            y, sr = vocoder.decode(outs)
            sf.write(
                os.path.join(config["outdir"], "wav", f"{utt_id}.wav"),
                y.cpu().numpy(),
                sr,
                "PCM_16",
            )


if __name__ == "__main__":
    main()
