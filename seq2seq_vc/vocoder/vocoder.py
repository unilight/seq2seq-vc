import logging
import time
import torch
import yaml

from parallel_wavegan.utils import load_model
from seq2seq_vc.utils import read_hdf5


class Vocoder(object):
    def __init__(
        self, checkpoint, config, stats, device, trg_stats=None, take_norm_feat=True
    ):
        self.device = device
        if take_norm_feat:
            assert (
                trg_stats is not None
            ), "trg_stats must be given if take_norm_feat=True"

            self.trg_stats = {
                "mean": torch.tensor(trg_stats["mean"], dtype=torch.float).to(
                    self.device
                ),
                "scale": torch.tensor(trg_stats["scale"], dtype=torch.float).to(
                    self.device
                ),
            }
        self.take_norm_feat = take_norm_feat

        # load config
        with open(config) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        # load model
        self.model = load_model(checkpoint, self.config)
        logging.info(f"Loaded model parameters from {checkpoint}.")
        self.model.remove_weight_norm()
        self.model = self.model.eval().to(device)

        # load stats for normalization
        self.stats = {
            "mean": torch.tensor(read_hdf5(stats, "mean"), dtype=torch.float).to(
                self.device
            ),
            "scale": torch.tensor(read_hdf5(stats, "scale"), dtype=torch.float).to(
                self.device
            ),
        }

    def decode(self, c):
        if self.take_norm_feat:
            # denormalize with target stats
            c = c * self.trg_stats["scale"] + self.trg_stats["mean"]
        # normalize with vocoder stats
        c = (c - self.stats["mean"]) / self.stats["scale"]

        start = time.time()
        y = self.model.inference(c, normalize_before=False).view(-1)
        rtf = (time.time() - start) / (len(y) / self.config["sampling_rate"])
        logging.info(f"Finished waveform generation. (RTF = {rtf:.03f}).")
        return y, self.config["sampling_rate"]
