import logging
import time
import torch
import yaml

from seq2seq_vc.utils import read_hdf5
from seq2seq_vc.vocoder import Vocoder
from seq2seq_vc.vocoder.griffin_lim import Spectrogram2Waveform
from s3prl.nn import Featurizer
import s3prl_vc.models
from s3prl_vc.upstream.interface import get_upstream


class S3PRL_Feat2Wav(object):
    def __init__(self, checkpoint, config, stats, trg_stats, device):

        self.device = device

        # load config for the s3prl downstream model
        with open(config) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        # upstream stats, used to denormalize the converted feature
        self.trg_stats = {
            "mean": torch.tensor(trg_stats["mean"], dtype=torch.float).to(self.device),
            "scale": torch.tensor(trg_stats["scale"], dtype=torch.float).to(
                self.device
            ),
        }

        # mel stats of the downstream model, only used in the autoregressive downstream model
        downstream_mel_stats = {
            "mean": torch.tensor(read_hdf5(stats, "mean"), dtype=torch.float).to(
                self.device
            ),
            "scale": torch.tensor(read_hdf5(stats, "scale"), dtype=torch.float).to(
                self.device
            ),
        }

        # get model and load parameters
        upstream_model = get_upstream(self.config["upstream"])
        upstream_featurizer = Featurizer(upstream_model)
        model_class = getattr(s3prl_vc.models, self.config["model_type"])
        model = model_class(
            upstream_featurizer.output_size,
            self.config["num_mels"],
            self.config["sampling_rate"]
            / self.config["hop_size"]
            * upstream_featurizer.downsample_rate
            / 16000,
            downstream_mel_stats,
            **self.config["model_params"],
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
        model = model.eval().to(device)
        self.model = model
        logging.info(f"Loaded S3PRL model parameters from {checkpoint}.")

        if self.config.get("vocoder", None) is not None:
            self.vocoder = Vocoder(
                self.config["vocoder"]["checkpoint"],
                self.config["vocoder"]["config"],
                self.config["vocoder"]["stats"],
                device,
                take_norm_feat=False,
            )
        else:
            self.vocoder = Spectrogram2Waveform(
                n_fft=self.config["fft_size"],
                n_shift=self.config["hop_size"],
                fs=self.config["sampling_rate"],
                n_mels=self.config["num_mels"],
                fmin=self.config["fmin"],
                fmax=self.config["fmax"],
                griffin_lim_iters=64,
                take_norm_feat=False,
            )

    def decode(self, c):
        # denormalize with target stats
        c = c * self.trg_stats["scale"] + self.trg_stats["mean"]
        lens = torch.LongTensor([c.shape[0]]).to(self.device)
        c = c.unsqueeze(0).float()

        start = time.time()
        outs, _ = self.model(c, lens, spk_embs=None)
        out = outs[0]
        y, sr = self.vocoder.decode(out)
        rtf = (time.time() - start) / (len(y) / self.config["sampling_rate"])
        logging.info(f"Finished waveform generation. (RTF = {rtf:.03f}).")
        return y, self.config["sampling_rate"]
