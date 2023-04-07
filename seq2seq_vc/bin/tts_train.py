#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Train TTS model."""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader

import seq2seq_vc
import seq2seq_vc.models
import seq2seq_vc.losses
import seq2seq_vc.trainers
import seq2seq_vc.collaters

from seq2seq_vc.datasets.tts_dataset import TTSDataset
from seq2seq_vc.losses import Seq2SeqLoss, GuidedMultiHeadAttentionLoss
from seq2seq_vc.utils import read_hdf5
from seq2seq_vc.vocoder import Vocoder
from seq2seq_vc.vocoder.s3prl_feat2wav import S3PRL_Feat2Wav
from seq2seq_vc.vocoder.griffin_lim import Spectrogram2Waveform

from seq2seq_vc.schedulers.warmup_lr import WarmupLR
from torch.optim.lr_scheduler import ExponentialLR
scheduler_classes = {
    "warmuplr": WarmupLR,
    "exponentiallr": ExponentialLR
}


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=("Train TTS model (See detail in bin/tts_train.py).")
    )
    parser.add_argument(
        "--train-dumpdir",
        required=True,
        type=str,
        help=("directory including source training data. "),
    )
    parser.add_argument(
        "--dev-dumpdir",
        required=True,
        type=str,
        help=("directory including source development data. "),
    )
    parser.add_argument(
        "--train-text",
        required=True,
        type=str,
        help=("original training text file. "),
    )
    parser.add_argument(
        "--dev-text",
        required=True,
        type=str,
        help=("original development text file. "),
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="stats file for denormalization.",
    )
    parser.add_argument(
        "--token-type",
        type=str,
        required=True,
        choices=["char", "phn"],
        help="token type",
    )
    parser.add_argument(
        "--non-linguistic-symbols",
        type=str,
        default=None,
        help="non_linguistic_symbols file path",
    )
    parser.add_argument(
        "--cleaner",
        type=str,
        choices=[None, "tacotron", "jaconv"],
        default=None,
        help="Apply text cleaning",
    )
    parser.add_argument(
        "--g2p",
        type=str,
        default=None,
        help="Specify g2p method if --token_type=phn",
    )
    parser.add_argument(
        "--token-list",
        type=str,
        required=True,
        help="a text mapping int-id to token",
    )
    parser.add_argument(
        "--feat-type",
        type=str,
        default="feats",
        help=(
            "target feature type. this is used as key name to read h5 feature files. "
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--pretrain",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = seq2seq_vc.__version__  # add version info

    # load target stats for denormalization
    config["stats"] = {
        "mean": read_hdf5(args.stats, "mean"),
        "scale": read_hdf5(args.stats, "scale"),
    }

    # write idim
    with open(args.token_list, encoding="utf-8") as f:
        token_list = [line.rstrip() for line in f]
    vocab_size = len(token_list)
    logging.info(f"Vocabulary size: {vocab_size }")
    config["model_params"]["idim"] = vocab_size

    # save config
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["format"] == "hdf5":
        mel_query = "*.h5"
        mel_load_fn = lambda x: read_hdf5(x, args.feat_type)  # NOQA
    elif config["format"] == "npy":
        mel_query = "*-feats.npy"
        mel_load_fn = np.load
    else:
        raise ValueError("support only hdf5 or npy format.")
    train_dataset = TTSDataset(
        root_dir=args.train_dumpdir,
        text_path=args.train_text,
        non_linguistic_symbols=args.non_linguistic_symbols,
        cleaner=args.cleaner,
        g2p=args.g2p,
        token_list=args.token_list,
        token_type=args.token_type,
        mel_query=mel_query,
        mel_load_fn=mel_load_fn,
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = TTSDataset(
        root_dir=args.dev_dumpdir,
        text_path=args.dev_text,
        non_linguistic_symbols=args.non_linguistic_symbols,
        cleaner=args.cleaner,
        g2p=args.g2p,
        token_list=args.token_list,
        token_type=args.token_type,
        mel_query=mel_query,
        mel_load_fn=mel_load_fn,
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collater_class = getattr(
        seq2seq_vc.collaters,
        config.get("collater_type", "ARTTSCollater"),
    )
    collater = collater_class()
    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models
    model_class = getattr(
        seq2seq_vc.models,
        config.get("model_type", "TransformerTTS"),
    )
    model = model_class(
        **config["model_params"],
    ).to(device)

    # load vocoder
    if config.get("vocoder", False):
        if config["vocoder"].get("vocoder_type", "") == "s3prl_vc":
            vocoder = S3PRL_Feat2Wav(
                config["vocoder"]["checkpoint"],
                config["vocoder"]["config"],
                config["vocoder"]["stats"],
                config["stats"],  # this is used to denormalized the converted features,
                device,
            )
        else:
            vocoder = Vocoder(
                config["vocoder"]["checkpoint"],
                config["vocoder"]["config"],
                config["vocoder"]["stats"],
                device,
                trg_stats=config[
                    "stats"
                ],  # this is used to denormalized the converted features,
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
            griffin_lim_iters=64,
        )

    # define criterions
    criterion = {
        "seq2seq": Seq2SeqLoss(
            # keep compatibility
            **config.get("seq2seq_loss_params", {})
        ).to(device)
    }
    if config.get("use_guided_attn_loss", False):  # keep compatibility
        criterion["guided_attn"] = GuidedMultiHeadAttentionLoss(
            # keep compatibility
            **config.get("guided_attn_loss_params", {}),
        ).to(device)
    else:
        config["use_guided_attn_loss"] = False
    
    if config.get("criterions", None):
        criterion = {
            criterion_name: getattr(seq2seq_vc.losses, criterion_class)() for criterion_name, criterion_class in config["criterions"].items()
        }

    # define optimizers and schedulers
    optimizer_class = getattr(
        torch.optim,
        # keep compatibility
        config.get("optimizer_type", "Adam"),
    )
    optimizer = optimizer_class(
        model.parameters(),
        **config["optimizer_params"],
    )
    scheduler_class = scheduler_classes.get(config.get("scheduler_type", "warmuplr"))
    scheduler = scheduler_class(
        optimizer=optimizer,
        **config["scheduler_params"],
    )

    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError(
                "apex is not installed. please check https://github.com/NVIDIA/apex."
            )
        model = DistributedDataParallel(model)

    # show settings
    logging.info(model)
    logging.info(optimizer)
    logging.info(scheduler)
    logging.info(criterion)

    # define trainer
    trainer_class = getattr(
        seq2seq_vc.trainers,
        config.get("trainer_type", "ARTTSTrainer"),
    )
    trainer = trainer_class(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        vocoder=vocoder,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
