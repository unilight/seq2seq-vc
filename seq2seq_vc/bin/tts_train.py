#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Train TTS model."""

import argparse
import logging
import os
import sys
import time

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import seq2seq_vc
import seq2seq_vc.models

from seq2seq_vc.datasets.tts_dataset import TTSDataset
from seq2seq_vc.losses import Seq2SeqLoss
from seq2seq_vc.utils import read_hdf5
from seq2seq_vc.vocoder import Vocoder

# set to avoid matplotlib error in CLI environment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seq2seq_vc.schedulers.warmup_lr import WarmupLR
scheduler_classes = dict(
    warmuplr=WarmupLR
)

class Trainer(object):
    """Customized trainer module for VC training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        vocoder,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.vocoder = vocoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        xs, ilens, ys, labels, olens, spembs = tuple([_.to(self.device) if _ is not None else _ for _ in batch])

        # model forward
        after_outs, before_outs, logits, ys_, labels_, olens_ = self.model(xs, ilens, ys, labels, olens, spembs)

        # seq2seq loss
        l1_loss, bce_loss = self.criterion(after_outs, before_outs, logits, ys_, labels_, olens_)
        gen_loss = l1_loss + bce_loss
        self.total_train_loss["train/l1_loss"] += l1_loss.item()
        self.total_train_loss["train/bce_loss"] += bce_loss.item()
        self.total_train_loss["train/loss"] += gen_loss.item()

        # update model
        self.optimizer.zero_grad()
        gen_loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["grad_norm"],
            )
        self.optimizer.step()
        self.scheduler.step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""

        pass

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        self.model.eval()

        # save intermediate result
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # restore mode
        self.model.train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # define function for plot prob and att_ws
        def _plot_and_save(array, figname, figsize=(6, 4), dpi=150, ref=None, origin="upper"):
            shape = array.shape
            if len(shape) == 1:
                # for eos probability
                plt.figure(figsize=figsize, dpi=dpi)
                plt.plot(array)
                plt.xlabel("Frame")
                plt.ylabel("Probability")
                plt.ylim([0, 1])
            elif len(shape) == 2:
                # for tacotron 2 attention weights, whose shape is (out_length, in_length)
                if ref is None:
                    plt.figure(figsize=figsize, dpi=dpi)
                    plt.imshow(array.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
                else:
                    plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
                    plt.subplot(1, 2, 1)
                    plt.imshow(array.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
                    plt.subplot(1, 2, 2)
                    plt.imshow(ref.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
            elif len(shape) == 4:
                # for transformer attention weights,
                # whose shape is (#leyers, #heads, out_length, in_length)
                plt.figure(figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi)
                for idx1, xs in enumerate(array):
                    for idx2, x in enumerate(xs, 1):
                        plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                        plt.imshow(x, aspect="auto")
                        plt.xlabel("Input")
                        plt.ylabel("Output")
            else:
                raise NotImplementedError("Support only from 1D to 4D array.")
            plt.tight_layout()
            if not os.path.exists(os.path.dirname(figname)):
                # NOTE: exist_ok = True is needed for parallel process decoding
                os.makedirs(os.path.dirname(figname), exist_ok=True)
            plt.savefig(figname)
            plt.close()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # generate
        xs, _, ys, _, olens, spembs = tuple([_.to(self.device) if _ is not None else _ for _ in batch])
        if spembs is None: spembs = [None] * len(xs)
        for idx, (x, y, olen, spemb) in enumerate(zip(xs, ys, olens, spembs)):
            start_time = time.time()
            if self.config["distributed"]:
                outs, probs, att_ws = self.model.module.inference(x, self.config["inference"], spemb=spemb)
            else:
                outs, probs, att_ws = self.model.inference(x, self.config["inference"], spemb=spemb)
            logging.info(
                "inference speed = %.1f frames / sec."
                % (int(outs.size(0)) / (time.time() - start_time))
            )

            _plot_and_save(
                outs.cpu().numpy(),
                dirname + f"/outs/{idx}_out.png",
                ref=y[:olen].cpu().numpy(),
                origin="lower"
            )
            _plot_and_save(
                probs.cpu().numpy(),
                dirname + f"/probs/{idx}_prob.png",
            )
            _plot_and_save(
                att_ws.cpu().numpy(),
                dirname + f"/att_ws/{idx}_att_ws.png",
            )

            if self.vocoder is not None:
                if not os.path.exists(os.path.join(dirname, "wav")):
                    os.makedirs(os.path.join(dirname, "wav"), exist_ok=True)
                y, sr = self.vocoder.decode(outs)
                sf.write(
                    os.path.join(dirname, "wav", f"{idx}_gen.wav"),
                    y.cpu().numpy(),
                    sr,
                    "PCM_16",
                )

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self):
        """Initialize customized collater for PyTorch DataLoader. """

    def __call__(self, batch):
        """Convert into batch tensors. """

        def pad_list(xs, pad_value):
            """Perform padding for the list of tensors.

            Args:
                xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
                pad_value (float): Value for padding.

            Returns:
                Tensor: Padded tensor (B, Tmax, `*`).

            Examples:
                >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
                >>> x
                [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
                >>> pad_list(x, 0)
                tensor([[1., 1., 1., 1.],
                        [1., 1., 0., 0.],
                        [1., 0., 0., 0.]])

            """
            n_batch = len(xs)
            max_len = max(x.size(0) for x in xs)
            pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

            for i in range(n_batch):
                pad[i, : xs[i].size(0)] = xs[i]

            return pad

        xs, ys = [b[0] for b in batch], [b[1] for b in batch]

        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long()
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long()

        # perform padding and conversion to tensor
        xs = pad_list([torch.from_numpy(x).long() for x in xs], 0)
        ys = pad_list([torch.from_numpy(y).float() for y in ys], 0)

        # make labels for stop prediction
        labels = ys.new_zeros(ys.size(0), ys.size(1))
        for i, l in enumerate(olens):
            labels[i, l - 1 :] = 1.0

        return xs, ilens, ys, labels, olens, None


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=(
            "Train TTS model (See detail in bin/tts_train.py)."
        )
    )
    parser.add_argument(
        "--train-dumpdir",
        required=True,
        type=str,
        help=(
            "directory including source training data. "
        ),
    )
    parser.add_argument(
        "--dev-dumpdir",
        required=True,
        type=str,
        help=(
            "directory including source development data. "
        ),
    )
    parser.add_argument(
        "--train-text",
        required=True,
        type=str,
        help=(
            "original training text file. "
        ),
    )
    parser.add_argument(
        "--dev-text",
        required=True,
        type=str,
        help=(
            "original development text file. "
        ),
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
        "scale": read_hdf5(args.stats, "scale")
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
        mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
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
    collater = Collater()
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
        vocoder = Vocoder(
            config["vocoder"]["checkpoint"],
            config["vocoder"]["config"],
            config["vocoder"]["stats"],
            config["trg_stats"],
            device
        )
    else:
        vocoder = None

    # define criterions
    criterion = Seq2SeqLoss(
        # keep compatibility
        **config.get("seq2seq_loss_params", {})
    ).to(device)

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
    trainer = Trainer(
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
