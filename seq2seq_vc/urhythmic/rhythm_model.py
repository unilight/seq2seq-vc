# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Urhythmic rhythm model related

   Based on Urhythmic: https://github.com/bshall/urhythmic
"""


import itertools
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import scipy.stats as stats
from seq2seq_vc.urhythmic.utils import SILENCE, SONORANT, SoundType


def transform(
    source: stats.rv_continuous, target: stats.rv_continuous, sample: float
) -> float:
    return target.ppf(source.cdf(sample))


def segment_rate(
    codes: List[SoundType],
    boundaries: List[int],
    sonorant: SoundType = SONORANT,
    silence: SoundType = SILENCE,
    unit_rate=0.02,
) -> float:
    times = np.round(np.array(boundaries) * unit_rate, 2)
    segments = [
        (code, t0, tn)
        for code, (t0, tn) in zip(codes, itertools.pairwise(times))
        if code not in silence
    ]
    return len([code for code, _, _ in segments if code in sonorant]) / sum(
        [tn - t0 for _, t0, tn in segments]
    )


class RhythmModelFineGrained:
    """Rhythm modeling block (Fine-Grained). Estimates the duration distribution of each sound type."""

    def __init__(self, hop_length: int = 320, sample_rate: int = 16000):
        """
        Args:
            hop_length (int): hop length between the frames of speech units.
            sample_rate (int): the sample rate of the audio waveforms.
        """
        self.hop_rate = hop_length / sample_rate
        self.source = None
        self.target = None

    def _tally_durations(
        self, utterances: List[Tuple[List[SoundType], List[int]]]
    ) -> Dict[SoundType, np.ndarray]:
        durations_dict = {}
        for clusters, boundaries in utterances:
            durations = np.diff(boundaries)
            for cluster, duration in zip(clusters, durations):
                # NOTE(unilight): the original implementation is
                # cluster in SILENCE, but it somehow does not work,
                # so I changed to comparison using value
                if (
                    cluster.value == SILENCE.value and duration <= 3
                ):  # ignore silences that are too short
                    continue
                durations_dict.setdefault(cluster, []).append(self.hop_rate * duration)
        return {
            cluster: np.array(durations)
            for cluster, durations in durations_dict.items()
        }

    def state_dict(self) -> Mapping[str, Mapping[SoundType, Tuple[float, ...]]]:
        state_dict = {}
        if self.source:
            state_dict["source"] = {
                cluster: (dist.args[0], dist.kwds["scale"])
                for cluster, dist in self.source.items()
            }
        if self.target:
            state_dict["target"] = {
                cluster: (dist.args[0], dist.kwds["scale"])
                for cluster, dist in self.target.items()
            }
        return state_dict

    def load_state_dict(
        self, state_dict: Mapping[str, Mapping[SoundType, Tuple[float, ...]]]
    ):
        if "source" in state_dict:
            self.source = {
                cluster.value: stats.gamma(a, scale=scale)
                for cluster, (a, _, scale) in state_dict["source"].items()
            }
        if "target" in state_dict:
            self.target = {
                cluster.value: stats.gamma(a, scale=scale)
                for cluster, (a, _, scale) in state_dict["target"].items()
            }

    def _fit(
        self, utterances: List[Tuple[List[SoundType], List[int]]]
    ) -> Mapping[SoundType, Tuple[float, ...]]:
        duration_tally = self._tally_durations(utterances)
        dists = {
            cluster: stats.gamma.fit(durations, floc=0)
            for cluster, durations in duration_tally.items()
        }
        return dists

    def fit_source(self, utterances: List[Tuple[List[SoundType], List[int]]]):
        """Fit the duration model for the source speaker.

        Args:
            utterances (List[Tuple[List[SoundType], List[int]]]): list of segemented utterances.
        """
        source = self._fit(utterances)
        self.source = {
            cluster: stats.gamma(a, scale=scale)
            for cluster, (a, _, scale) in source.items()
        }

    def fit_target(self, utterances: List[Tuple[List[SoundType], List[int]]]):
        """Fit the duration model for the target speaker.

        Args:
            utterances (List[Tuple[List[SoundType], List[int]]]): list of segemented utterances.
        """

        target = self._fit(utterances)
        self.target = {
            cluster: stats.gamma(a, scale=scale)
            for cluster, (a, _, scale) in target.items()
        }

    def __call__(self, clusters: List[SoundType], boundaries: List[int]) -> List[int]:
        """Transforms the source durations to match the target rhythm.

        Args:
            clusters (List[SoundType]): list of segmented sound types of shape (N,).
            boundaries (List[int]): list of segment boundaries of shape (N+1,).

        Returns:
            List[int]: list of target durations of shape (N,)
        """
        durations = self.hop_rate * np.diff(boundaries)
        durations = [
            transform(self.source[cluster.value], self.target[cluster.value], duration)
            for cluster, duration in zip(clusters, durations)
            # NOTE(unilight): the original implementation is
            # cluster not in SILENCE, but it somehow does not work,
            # so I changed to comparison using value
            if not cluster.value == SILENCE.value
            or duration > 3 * self.hop_rate  # ignore silences that are too short
        ]
        durations = [round(duration / self.hop_rate) for duration in durations]
        return durations
