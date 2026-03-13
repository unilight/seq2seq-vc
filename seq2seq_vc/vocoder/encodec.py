import logging
import time
import torch
import yaml

from seq2seq_vc.utils import read_hdf5
from seq2seq_vc.utils.encodec import get_encodec_model


class EnCodec_decoder(object):
    def __init__(self, trg_stats, device, sr=24000):

        self.device = device
        self.sr = sr

        # upstream stats, used to denormalize the converted feature
        self.trg_stats = {
            "mean": torch.tensor(trg_stats["mean"], dtype=torch.float).to(self.device),
            "scale": torch.tensor(trg_stats["scale"], dtype=torch.float).to(
                self.device
            ),
        }

        # initialize EnCodec model
        self.model = get_encodec_model().to(device)

    def decode_frame(self, frame, qdq=True):
        # qdq: quantize -> de-quantize
        if qdq:
            # codes is [K, V, T], with T frames, K nb of codebooks.
            codes = self.model.quantizer.encode(
                frame, self.model.frame_rate, self.model.bandwidth
            )
            emb = self.model.quantizer.decode(codes)  # [B, 128, T]
        else:
            emb = frame
        return self.model.decoder(emb)

    def decode(self, c):
        # c has shape [t, d]

        # denormalize with target stats
        c = c * self.trg_stats["scale"] + self.trg_stats["mean"]

        start = time.time()

        # actual decoding
        assert self.model.segment_length is None
        with torch.no_grad():
            y = self.decode_frame(c.transpose(0, 1).unsqueeze(0))  # [1, 1, T]
        y = y.squeeze(0).squeeze(0)

        rtf = (time.time() - start) / (len(y) / self.sr)
        logging.info(f"Finished waveform generation. (RTF = {rtf:.03f}).")

        return y, self.sr
