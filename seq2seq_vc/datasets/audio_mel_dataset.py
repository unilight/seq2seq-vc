# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import os

from multiprocessing import Manager

import numpy as np

from torch.utils.data import Dataset

from seq2seq_vc.utils import find_files, read_hdf5, get_basename


class AudioMelDataset(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*.h5",
        mel_query="*.h5",
        audio_load_fn=lambda x: read_hdf5(x, "wave"),
        mel_load_fn=lambda x: read_hdf5(x, "feats"),
        audio_length_threshold=None,
        mel_length_threshold=None,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))
        mel_files = sorted(find_files(root_dir, mel_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    "Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    "Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(mel_files), (
            f"Number of audio and mel files are different ({len(audio_files)} vs"
            f" {len(mel_files)})."
        )

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.mel_files = mel_files
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        mel = self.mel_load_fn(self.mel_files[idx])

        if self.return_utt_id:
            items = utt_id, audio, mel
        else:
            items = audio, mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class AudioDataset(Dataset):
    """PyTorch compatible audio dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*-wave.npy",
        audio_length_threshold=None,
        audio_load_fn=np.load,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            audio_load_fn (func): Function to load audio file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.waning(
                    "some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.return_utt_id = return_utt_id
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio (T,).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])

        if self.return_utt_id:
            items = utt_id, audio
        else:
            items = audio

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class MelDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        root_dir,
        mel_query="*-feats.npy",
        mel_length_threshold=None,
        mel_load_fn=np.load,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of the mel files
        mel_files = sorted(find_files(root_dir, mel_query))

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    "Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mel files in ${root_dir}."

        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        if ".npy" in mel_query:
            self.utt_ids = [
                os.path.basename(f).replace("-feats.npy", "") for f in mel_files
            ]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(mel_files))]

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

        if self.return_utt_id:
            items = utt_id, mel
        else:
            items = mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.mel_files)


class ParallelVCMelDataset(Dataset):
    """PyTorch compatible mel-to-mel dataset for parallel VC."""

    def __init__(
        self,
        src_root_dir,
        trg_root_dir,
        mel_query="*-feats.npy",
        src_load_fn=np.load,
        trg_load_fn=np.load,
        dp_input_root_dir=None,
        dp_input_query="*.h5",
        dp_input_load_fn=np.load,
        durations_dir=None,
        duration_query="*.txt",
        reduction_factor: int = 1,
        return_utt_id=False,
        allow_cache=False,
        mp=True,
    ):
        """Initialize dataset.

        Args:
            src_root_dir (str): Root directory including dumped files for the source.
            trg_root_dir (str): Root directory including dumped files for the target.
            mel_query (str): Query to find feature files in root_dir.
            src_load_fn (func): Function to load source feature files.
            trg_load_fn (func): Function to load target feature files.
            dp_input_root_dir (str): Root directory including dumped files for the dp input.
            dp_input_load_fn (func): Function to load feature files for dp.
            durations_dir (str): Root directory including duration files.
            duration_query (str): Query to find duration files in src/trg_durations_dir.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of the mel files
        src_mel_files = sorted(find_files(src_root_dir, mel_query))
        trg_mel_files = sorted(find_files(trg_root_dir, mel_query))

        # assert the number of files
        assert len(src_mel_files) != 0, f"Not found any mel files in ${src_root_dir}."
        assert len(trg_mel_files) != 0, f"Not found any mel files in ${trg_root_dir}."

        self.src_mel_files = src_mel_files
        self.trg_mel_files = trg_mel_files
        self.src_load_fn = src_load_fn
        self.trg_load_fn = trg_load_fn
        self.dp_input_load_fn = dp_input_load_fn

        # make sure the utt ids match
        src_utt_ids = sorted(
            [os.path.splitext(os.path.basename(f))[0] for f in src_mel_files]
        )
        trg_utt_ids = sorted(
            [os.path.splitext(os.path.basename(f))[0] for f in trg_mel_files]
        )
        assert set(src_utt_ids) == set(
            trg_utt_ids
        ), f"{len(set(src_utt_ids))} {len(set(trg_utt_ids))}{set(src_utt_ids).difference(set(trg_utt_ids))}"
        self.utt_ids = src_utt_ids

        # use map(list, zip(...)) to get list of list
        self.mel_files = list(map(list, zip(self.src_mel_files, self.trg_mel_files)))
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            if mp:
                # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
                self.manager = Manager()
                self.caches = self.manager.list()
                self.caches += [() for _ in range(len(self.mel_files))]
            else:
                self.caches = [() for _ in range(len(self.mel_files))]

        # load dp input feature files and zip with mel_files
        if dp_input_root_dir is not None:
            # find all dp input files
            dp_input_feat_files = sorted(find_files(dp_input_root_dir, dp_input_query))
            assert len(dp_input_feat_files) == len(self.mel_files)

            self.mel_files = [
                v + [dp_input_feat_files[i]] for i, v in enumerate(self.mel_files)
            ]
            self.use_dp_input = True
        else:
            self.use_dp_input = False

        # load duration files, and zip with mel_files
        if durations_dir is not None:
            # find all duration files
            duration_files = sorted(find_files(durations_dir, duration_query))
            assert len(duration_files) == len(self.mel_files)

            self.mel_files = [
                v + [duration_files[i]] for i, v in enumerate(self.mel_files)
            ]
            self.use_durations = True
            self.reduction_factor = reduction_factor
        else:
            self.use_durations = False

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
        src_mel = self.src_load_fn(self.mel_files[idx][0])
        trg_mel = self.trg_load_fn(self.mel_files[idx][1])

        items = {"src_feat": src_mel, "trg_feat": trg_mel}

        if self.use_dp_input:
            # read dp input feat
            dp_input = self.dp_input_load_fn(self.mel_files[idx][2])
            items["dp_input"] = dp_input

        if self.use_durations:
            # read durations if exists
            with open(self.mel_files[idx][3], "r") as f:
                lines = f.read().splitlines()
                assert len(lines) == 1
                durations = np.array([int(d) for d in lines[0].split(" ")])

            # force the target to have the same length as the duration sum
            total_length = np.sum(durations) * self.reduction_factor
            trg_mel = trg_mel[:total_length]
            items["duration"] = durations

        items["trg_feat"] = trg_mel

        if self.return_utt_id:
            items["utt_id"] = utt_id

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.mel_files)


class SourceVCMelDataset(Dataset):
    """PyTorch compatible mel dataset for VC.
    Contains source side mel only, mainly designed for evaluation.
    """

    def __init__(
        self,
        src_root_dir,
        mel_query="*-feats.npy",
        mel_load_fn=np.load,
        dp_input_root_dir=None,
        dp_input_query="*.h5",
        dp_input_load_fn=np.load,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            src_root_dir (str): Root directory including dumped files for the source.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of the mel files
        self.src_mel_files = sorted(find_files(src_root_dir, mel_query))

        # assert the number of files
        assert (
            len(self.src_mel_files) != 0
        ), f"Not found any mel files in ${src_root_dir}."

        self.mel_load_fn = mel_load_fn
        self.dp_input_load_fn = dp_input_load_fn
        self.utt_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in self.src_mel_files
        ]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.src_mel_files))]

        if dp_input_root_dir is not None:
            self.use_dp_input = True

            # find all dp input files
            dp_input_feat_files = sorted(find_files(dp_input_root_dir, dp_input_query))
            assert len(dp_input_feat_files) == len(self.src_mel_files)

            self.src_mel_files = [
                [v] + [dp_input_feat_files[i]] for i, v in enumerate(self.src_mel_files)
            ]
        else:
            self.src_mel_files = [[v] for v in self.src_mel_files]
            self.use_dp_input = False

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
        src_mel = self.mel_load_fn(self.src_mel_files[idx][0])

        items = {"src_feat": src_mel}

        if self.return_utt_id:
            items["utt_id"] = utt_id

        if self.use_dp_input:
            items["dp_input"] = self.dp_input_load_fn(self.src_mel_files[idx][1])

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.src_mel_files)
