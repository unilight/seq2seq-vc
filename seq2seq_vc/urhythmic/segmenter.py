# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Urhythmic segmenter module

   Based on Urhythmic: https://github.com/bshall/urhythmic
"""

import itertools
from collections import Counter
from typing import Any, List, Mapping, Tuple

import numba
import numpy as np
import torch
from seq2seq_vc.urhythmic.utils import OBSTRUENT, SILENCE, SONORANT, SoundType
from sklearn.cluster import AgglomerativeClustering


class Segmenter:
    """Segmentation and clustering block. Groups similar speech units into short segments.
    The segments are then combined into coarser groups approximating sonorants, obstruents, and silences.
    """

    def __init__(self, num_clusters: int = 3, gamma: float = 2):
        """
        Args:
            num_clusters (int): number of clusters used for agglomerative clustering.
            gamma (float): regularizer weight encouraging longer segments
        """
        self.gamma = gamma
        self.clustering = AgglomerativeClustering(n_clusters=num_clusters)
        self.sound_types = dict()

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "n_clusters_": self.clustering.n_clusters_,
            "labels_": torch.from_numpy(self.clustering.labels_),
            "n_leaves_": self.clustering.n_leaves_,
            "n_features_in_": self.clustering.n_features_in_,
            "children_": torch.from_numpy(self.clustering.children_),
            "sound_types": self.sound_types,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        if self.clustering.n_clusters != state_dict["n_clusters_"]:
            raise RuntimeError(
                "Error in loading state_dict for {}", self.__class__.__name__
            )
        self.clustering.labels_ = state_dict["labels_"].numpy()
        self.clustering.n_leaves_ = state_dict["n_leaves_"]
        self.clustering.n_features_in_ = state_dict["n_features_in_"]
        self.clustering.children_ = state_dict["children_"].numpy()
        self.sound_types = state_dict["sound_types"]

    def cluster(self, codebook: np.ndarray):
        """Fit the hierarchical clustering from the codebook of discrete units.

        Args:
            codebook (NDArray): codebook of discrete units of shape (K, D)
                where K is the number of units and D is the unit dimension.
        """
        self.clustering.fit(codebook)

    def identify(
        self,
        utterances: List[Tuple[np.ndarray, ...]],
    ) -> Mapping[int, SoundType]:
        """Identify which clusters correspond to sonorants, obstruents, and silences.
        Only implemented for num_clusters = 3.

        Args:
            utterances (List[Tuple[np.ndarray, ...]]): list of segmented utterances along with marked silences and voiced frames.

        Returns:
            Mapping[int, SoundType]: mapping of cluster id to sonorant, obstruent, or silence.
        """
        if self.clustering.n_clusters_ != 3:
            raise ValueError(
                "Cluster identification is only implemented for num_clusters = 3."
            )

        silence_overlap = Counter()
        voiced_overlap = Counter()
        total = Counter()

        for segments, boundaries, silences, voiced_flags in utterances:
            for code, (a, b) in zip(segments, itertools.pairwise(boundaries)):
                silence_overlap[code] += np.count_nonzero(silences[a : b + 1])
                voiced_overlap[code] += np.count_nonzero(voiced_flags[a : b + 1])
                total[code] += b - a + 1

        clusters = {0, 1, 2}

        silence, _ = max(
            [(k, v / total[k]) for k, v in silence_overlap.items()], key=lambda x: x[1]
        )
        clusters.remove(silence)

        sonorant, _ = max(
            [(k, v / total[k]) for k, v in voiced_overlap.items() if k in clusters],
            key=lambda x: x[1],
        )
        clusters.remove(sonorant)

        obstruent = clusters.pop()

        self.sound_types = {
            silence: SILENCE,
            sonorant: SONORANT,
            obstruent: OBSTRUENT,
        }
        return self.sound_types

    def _segment(self, log_probs: np.ndarray) -> Tuple[List[int], List[int]]:
        codes, boundaries = segment(log_probs, self.gamma)
        segments = codes[boundaries[:-1]]
        segments, boundaries = cluster_merge(self.clustering, segments, boundaries)
        return list(segments), list(boundaries)

    def __call__(self, log_probs: np.ndarray) -> Tuple[List[SoundType], List[int]]:
        """Segment the soft speech units into groups approximating the different sound types.

        Args:
            log_probs (NDArray): log probabilities of each discrete unit of shape (T, K) where T is the number of frames and K is the number of discrete units

        Returns:
            List[SoundType]: list of segmented sound types of shape (N,).
            List[int]: list of segment boundaries of shape (N+1,).
        """
        segments, boundaries = self._segment(log_probs)
        segments = [self.sound_types[cluster] for cluster in segments]
        return segments, boundaries


def segment(log_probs: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    alpha, P = _segment(log_probs, gamma)
    return _backtrack(alpha, P)


@numba.njit()
def _backtrack(alpha, P):
    rhs = len(alpha) - 1
    segments = np.zeros(len(alpha) - 1, dtype=np.int32)
    boundaries = [rhs]
    while rhs != 0:
        lhs, code = P[rhs, :]
        boundaries.append(lhs)
        segments[lhs:rhs] = code
        rhs = lhs
    boundaries.reverse()
    return segments, np.array(boundaries)


@numba.njit()
def _segment(log_probs, gamma):
    T, K = log_probs.shape

    alpha = np.zeros(T + 1, dtype=np.float32)
    P = np.zeros((T + 1, 2), dtype=np.int32)
    D = np.zeros((T, T, K), dtype=np.float32)

    for t in range(T):
        for k in range(K):
            D[t, t, k] = log_probs[t, k]
    for t in range(T):
        for s in range(t + 1, T):
            D[t, s, :] = D[t, s - 1, :] + log_probs[s, :]

    for t in range(T):
        alpha[t + 1] = -np.inf
        for s in range(t + 1):
            k = np.argmax(D[t - s, t, :])
            alpha_max = alpha[t - s] + D[t - s, t, k] + gamma * s
            if alpha_max > alpha[t + 1]:
                P[t + 1, :] = t - s, k
                alpha[t + 1] = alpha_max
    return alpha, P


def cluster_merge(
    clustering: AgglomerativeClustering, segments: np.ndarray, boundaries: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    clusters = clustering.labels_[segments]
    cluster_switches = np.diff(clusters, prepend=-1, append=-1)
    (cluster_boundaries,) = np.nonzero(cluster_switches)
    clusters = clusters[cluster_boundaries[:-1]]
    cluster_boundaries = boundaries[cluster_boundaries]
    return clusters, cluster_boundaries
