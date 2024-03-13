# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Urhythmic utility file.

   Based on Urhythmic: https://github.com/bshall/urhythmic
"""


from enum import Flag, auto

import torch


class SoundType(Flag):
    VOWEL = auto()
    APPROXIMANT = auto()
    NASAL = auto()
    FRICATIVE = auto()
    STOP = auto()
    SILENCE = auto()


SONORANT = SoundType.VOWEL | SoundType.APPROXIMANT | SoundType.NASAL
OBSTRUENT = SoundType.FRICATIVE | SoundType.STOP
SILENCE = SoundType.SILENCE


def get_padding(k, d):
    return int((k * d - d) / 2)


class Metric:
    def __init__(self):
        self.steps = 0
        self.value = 0

    def update(self, value):
        self.steps += 1
        self.value += (value - self.value) / self.steps
        return self.value

    def reset(self):
        self.steps = 0
        self.value = 0


def save_checkpoint(
    checkpoint_dir,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    step,
    loss,
    best,
    logger,
):
    state = {
        "generator": {
            "model": generator.state_dict(),
            "optimizer": optimizer_generator.state_dict(),
            "scheduler": scheduler_generator.state_dict(),
        },
        "discriminator": {
            "model": discriminator.state_dict(),
            "optimizer": optimizer_discriminator.state_dict(),
            "scheduler": scheduler_discriminator.state_dict(),
        },
        "step": step,
        "loss": loss,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"model-{step}.pt"
    torch.save(state, checkpoint_path)
    if best:
        best_path = checkpoint_dir / "model-best.pt"
        torch.save(state, best_path)
    logger.info(f"Saved checkpoint: {checkpoint_path.stem}")


def load_checkpoint(
    load_path,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    rank,
    logger,
    finetune=False,
):
    verb = "Resuming" if not finetune else "Finetuning"
    logger.info(f"{verb} checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    generator.load_state_dict(checkpoint["generator"]["model"])
    discriminator.load_state_dict(checkpoint["discriminator"]["model"])
    if not finetune:
        optimizer_generator.load_state_dict(checkpoint["generator"]["optimizer"])
        scheduler_generator.load_state_dict(checkpoint["generator"]["scheduler"])
        optimizer_discriminator.load_state_dict(
            checkpoint["discriminator"]["optimizer"]
        )
        scheduler_discriminator.load_state_dict(
            checkpoint["discriminator"]["scheduler"]
        )
        return checkpoint["step"], checkpoint["loss"]
    else:
        return 0, float("inf")
