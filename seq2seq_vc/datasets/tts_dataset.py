# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import os

from multiprocessing import Manager

import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

from seq2seq_vc.text.build_tokenizer import build_tokenizer
from seq2seq_vc.text.cleaner import TextCleaner
from seq2seq_vc.text.token_id_converter import TokenIDConverter
from seq2seq_vc.utils import find_files
from seq2seq_vc.utils import read_hdf5

def read_2column_text(path):
    """Read a text file having 2 column as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2column_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v
    return data

class TTSDataset(Dataset):
    """PyTorch compatible dataset for TTS."""

    def __init__(
        self,
        root_dir,
        text_path,
        non_linguistic_symbols,
        cleaner,
        g2p,
        token_list,
        token_type,
        mel_query="*-feats.npy",
        mel_load_fn=np.load,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            text_path (str): File path containing text.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        self.text_cleaner = TextCleaner(cleaner)
        self.tokenizer = build_tokenizer(
            token_type=token_type,
            non_linguistic_symbols=non_linguistic_symbols,
            g2p_type=g2p,
        )
        self.token_id_converter = TokenIDConverter(
            token_list=token_list,
            unk_symbol="<unk>",
        )

        # find all of the mel files
        self.mel_files = sorted(find_files(root_dir, mel_query))

        # assert the number of files
        assert len(self.mel_files) != 0, f"Not found any mel files in ${root_dir}."
        
        # load all text and filter those not in mel files
        mel_utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.mel_files]
        self.utt_ids = mel_utt_ids
        texts = read_2column_text(text_path)
        self.texts = {k: v for k, v in texts.items() if k in mel_utt_ids}
        
        self.mel_load_fn = mel_load_fn
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.mel_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        mel = self.mel_load_fn(self.mel_files[idx])

        # load and process text
        raw_text = self.texts[utt_id]
        text = self.text_cleaner(raw_text)
        tokens = self.tokenizer.text2tokens(text)
        text_ints = self.token_id_converter.tokens2ids(tokens)
        ret_text = np.array(text_ints, dtype=np.int64)

        if self.return_utt_id:
            items = utt_id, ret_text, mel, tokens
        else:
            items = ret_text, mel, tokens

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.mel_files)