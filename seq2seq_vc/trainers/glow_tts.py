#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
import soundfile as sf
import time
import torch

from seq2seq_vc.trainers.base import Trainer

# set to avoid matplotlib error in CLI environment
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GlowTTSTrainer(Trainer):
    def _train_step(self, batch):
        # parse batch

        xs, ilens, ys, olens, spembs = tuple(
            [_.to(self.device) if _ is not None else _ for _ in batch]
        )

        # model forward
        (
            (z, z_m, z_logs, logdet, z_mask),
            (x_m, x_logs, x_mask),
            (attn, logw, logw_),
        ) = self.model(xs, ilens, ys, olens)

        # loss
        mle_loss = self.criterion["mle"](z, z_m, z_logs, logdet, z_mask)
        dp_loss = self.criterion["dp"](logw, logw_, ilens)
        gen_loss = mle_loss + dp_loss
        self.total_train_loss["train/mle_loss"] += mle_loss.item()
        self.total_train_loss["train/dp_loss"] += dp_loss.item()

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

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # define function for plot prob and att_ws
        def _plot_and_save(
            array, figname, figsize=(6, 4), dpi=150, ref=None, origin="upper"
        ):
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
                plt.figure(
                    figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi
                )
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
        xs, ilens, ys, olens, _ = tuple(
            [_.to(self.device) if _ is not None else _ for _ in batch]
        )
        for idx, (x, ilen, y, olen) in enumerate(zip(xs, ilens, ys, olens)):
            start_time = time.time()
            if self.config["distributed"]:
                outs, probs, att_ws = self.model.module.inference(
                    x, self.config["inference"], spemb=spemb
                )
            else:
                (outs, *_), *_, (attn_gen, log_durations, log_durations_) = self.model.inference(
                    x,
                    ilen,
                    noise_scale=self.config["inference"]["noise_scale"],
                    length_scale=self.config["inference"]["length_scale"],
                )
                # outs, probs, att_ws = self.model.inference(
                # x, self.config["inference"], spemb=spemb
                # )
            logging.info(
                f"input # tokens {ilen.cpu().numpy()}, output length {outs.shape[0]}"
            )
            logging.info(
                f"predicted durations: {str(torch.ceil(torch.exp(log_durations)).cpu().numpy())}"
            )

            _plot_and_save(
                outs.cpu().numpy(),
                dirname + f"/outs/{idx}_out.png",
                ref=y[:olen].cpu().numpy(),
                origin="lower",
            )
            # _plot_and_save(
            # probs.cpu().numpy(),
            # dirname + f"/probs/{idx}_prob.png",
            # )
            # _plot_and_save(
            # att_ws.cpu().numpy(),
            # dirname + f"/att_ws/{idx}_att_ws.png",
            # )

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
