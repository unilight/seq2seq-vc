#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained VC model."""

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
from seq2seq_vc.datasets import ParallelVCMelDataset, SourceVCMelDataset
from seq2seq_vc.utils import read_hdf5, write_hdf5
from seq2seq_vc.utils.plot import plot_attention, plot_generated_and_ref_2d, plot_1d
from seq2seq_vc.vocoder import Vocoder
from seq2seq_vc.vocoder.s3prl_feat2wav import S3PRL_Feat2Wav
from seq2seq_vc.vocoder.encodec import EnCodec_decoder
from seq2seq_vc.utils.types import str2bool
from seq2seq_vc.utils.duration_calculator import DurationCalculator


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode with trained VC model " "(See detail in bin/vc_decode.py)."
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
        "--dp_input_dumpdir",
        default=None,
        type=str,
        help=("directory including duration predictor input feature files. "),
    )
    parser.add_argument(
        "--trg-stats",
        type=str,
        required=True,
        help="stats file for target denormalization.",
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
        "--src-feat-type",
        type=str,
        default="feats",
        help=(
            "source feature type. this is used as key name to read h5 feature files. "
        ),
    )
    parser.add_argument(
        "--trg-feat-type",
        type=str,
        default="feats",
        help=(
            "target feature type. this is used as key name to read h5 feature files. "
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--use-teacher-forcing",
        type=str2bool,
        default=False,
        help="Whether to use teacher forcing",
    )
    parser.add_argument(
        "--trg-dumpdir",
        default=None,
        type=str,
        help=(
            "directory including feature files. "
            "you need to specify either feats-scp or dumpdir."
        ),
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

    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # load target stats for denormalization
    config["trg_stats"] = {
        "mean": read_hdf5(args.trg_stats, f"{args.trg_feat_type}_mean"),
        "scale": read_hdf5(args.trg_stats, f"{args.trg_feat_type}_scale"),
    }

    # check arguments
    if (args.feats_scp is not None and args.dumpdir is not None) or (
        args.feats_scp is None and args.dumpdir is None
    ):
        raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # get dataset
    if args.dumpdir is not None:
        mel_query = "*.h5"
        mel_load_fn = lambda x: read_hdf5(x, args.src_feat_type)  # NOQA
        dp_input_load_fn = lambda x: read_hdf5(
            x, config.get("duration_predictor_feat", "mel")
        )  # NOQA
        if args.use_teacher_forcing:
            dataset = ParallelVCMelDataset(
                src_root_dir=args.dumpdir,
                trg_root_dir=args.trg_dumpdir,
                mel_query=mel_query,
                src_load_fn=mel_load_fn,
                trg_load_fn=mel_load_fn,
                dp_input_load_fn=dp_input_load_fn,
                return_utt_id=True,
            )
        else:
            dataset = SourceVCMelDataset(
                src_root_dir=args.dumpdir,
                mel_query=mel_query,
                mel_load_fn=mel_load_fn,
                dp_input_root_dir=args.dp_input_dumpdir,
                dp_input_load_fn=dp_input_load_fn,
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
    model_class = getattr(seq2seq_vc.models, config["model_type"])
    model = model_class(**config["model_params"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model = model.eval().to(device)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")

    # check autoregressive or non-autoregressive
    if model_class in seq2seq_vc.models.AR_VC_MODELS:
        ar = True
    elif model_class in seq2seq_vc.models.NAR_VC_MODELS:
        ar = False

    # load vocoder if provided
    if config.get("vocoder", False):
        vocoder_type = config["vocoder"].get("vocoder_type", "")
        if vocoder_type == "s3prl_vc":
            vocoder = S3PRL_Feat2Wav(
                config["vocoder"]["checkpoint"],
                config["vocoder"]["config"],
                config["vocoder"]["stats"],
                config[
                    "trg_stats"
                ],  # this is used to denormalized the converted features,
                device,
            )
        elif vocoder_type == "encodec":
            vocoder = EnCodec_decoder(
                config[
                    "trg_stats"
                ],  # this is used to denormalized the converted features,
                device,
            )
        else:
            vocoder = Vocoder(
                config["vocoder"]["checkpoint"],
                config["vocoder"]["config"],
                config["vocoder"]["stats"],
                device,
                trg_stats=config[
                    "trg_stats"
                ],  # this is used to denormalized the converted features,
            )

    # build duration calculator in teacher-forcing mode
    if args.use_teacher_forcing:
        duration_calculator = DurationCalculator()

    # start generation
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, batch in enumerate(pbar, 1):
            start_time = time.time()

            if ar:
                if args.use_teacher_forcing:
                    utt_id = batch["utt_id"]
                    x = batch["src_feat"]
                    y = batch["trg_feat"]
                    x = torch.tensor(x, dtype=torch.float).to(device)
                    y = torch.tensor(y, dtype=torch.float).to(device)
                    xs, ys = x.unsqueeze(0), y.unsqueeze(0)
                    ilens = x.new_tensor([xs.size(1)]).long()
                    olens = y.new_tensor([ys.size(1)]).long()
                    labels = ys.new_zeros(ys.size(0), ys.size(1))
                    for i, l in enumerate(olens):
                        labels[i, l - 1 :] = 1.0
                    (
                        outs,
                        _,
                        probs,
                        _,
                        _,
                        _,
                        (att_ws, _, _),
                    ) = model(xs, ilens, ys, labels, olens, None)
                    outs = outs.squeeze(0)
                    probs = probs.squeeze(0)
                    att_ws = torch.cat(att_ws, dim=0)
                else:
                    utt_id = batch["utt_id"]
                    x = batch["src_feat"]
                    x = torch.tensor(x, dtype=torch.float).to(device)
                    outs, probs, att_ws = model.inference(
                        x, config["inference"], spemb=None
                    )
            else:
                utt_id = batch["utt_id"]
                x = batch["src_feat"]
                dp_input = batch["dp_input"]
                x = torch.tensor(x, dtype=torch.float).to(device)
                dp_input = torch.tensor(dp_input, dtype=torch.float).to(device)
                outs, d_outs = model.inference(x, dp_input=dp_input)
                duration = [str(int(d)) for d in d_outs.cpu().numpy()]

            logging.info(
                "inference speed = %.1f frames / sec."
                % (int(outs.size(0)) / (time.time() - start_time))
            )

            # plot figures
            plot_generated_and_ref_2d(
                outs.cpu().numpy(),
                config["outdir"] + f"/outs/{utt_id}.png",
                origin="lower",
            )
            if ar:
                plot_1d(
                    probs.cpu().numpy(),
                    config["outdir"] + f"/probs/{utt_id}_prob.png",
                )
                plot_attention(
                    att_ws.cpu().numpy(),
                    config["outdir"] + f"/att_ws/{utt_id}_att_ws.png",
                )

            # write feats
            if not os.path.exists(os.path.join(config["outdir"], args.trg_feat_type)):
                os.makedirs(
                    os.path.join(config["outdir"], args.trg_feat_type), exist_ok=True
                )

            write_hdf5(
                config["outdir"] + f"/{args.trg_feat_type}/{utt_id}.h5",
                args.trg_feat_type,
                outs.cpu().numpy().astype(np.float32),
            )

            # write waveform if vocoder is provided
            if config.get("vocoder", False):

                if not os.path.exists(os.path.join(config["outdir"], "wav")):
                    os.makedirs(os.path.join(config["outdir"], "wav"), exist_ok=True)

                y, sr = vocoder.decode(outs)
                sf.write(
                    os.path.join(config["outdir"], "wav", f"{utt_id}.wav"),
                    y.cpu().numpy(),
                    sr,
                    "PCM_16",
                )

            if ar and args.use_teacher_forcing:
                # generate durations from att_ws
                duration, focus_rate = duration_calculator(att_ws)
                logging.info(f"focus rate = {focus_rate:.3f}")
                duration = [str(int(d)) for d in duration.cpu().numpy()]

            if (ar and args.use_teacher_forcing) or (not ar):
                # write durations
                if not os.path.exists(os.path.join(config["outdir"], "durations")):
                    os.makedirs(
                        os.path.join(config["outdir"], "durations"), exist_ok=True
                    )

                with open(
                    os.path.join(config["outdir"], "durations", utt_id + ".txt"), "w"
                ) as f:
                    f.write(" ".join(duration))


if __name__ == "__main__":
    main()
