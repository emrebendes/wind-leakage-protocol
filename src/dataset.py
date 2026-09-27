# -*- coding: utf-8 -*-
"""
Data layer — reads the causal feature cache.
============================================

The model sees the decomposition components as CHANNELS:

        X : (batch, look_back, K)     components of the trailing window
        y : (batch, H)                RAW wind speed, m/s, at t+1 .. t+H

This replaces the previous design, in which one model was trained per component
and the predictions were summed. Under causal decomposition that arrangement
needs component targets anchored at t+H while the features are anchored at t,
which is both H times more expensive and internally inconsistent. Feeding the
components as channels and predicting the raw signal directly avoids the
problem and trains one model instead of K.

Guarantees enforced here
------------------------
* Normalisation statistics come from the TRAINING split only.
* Every condition uses the same origins for a given W, so no method is scored
  on easier points than another.
* The cache is sliced to `look_back` from a depth of LOOK_BACK_MAX, which is
  exact — `tests/test_causality.py::test_look_back_slicing_is_exact`.
* The cache signature is checked against the requested parameters, so a stale
  cache cannot be used silently.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (DATA_FILE, load_series, CACHE_DIR, TRAIN_RATIO, VAL_RATIO,
                    LOOK_BACK_MAX, MAX_HORIZON, SEED)


# =============================================================================
class CausalFeatureCache:
    """
    Reader for one (method, W, station) cache directory.

    Owns the on-disk layout so that no other module has to know it, and
    verifies that what it loads is what was asked for.
    """

    def __init__(self, method: str, W: int, station: int, arm: str = "causal",
                 root: str = CACHE_DIR):
        self.method = method
        self.arm = arm
        self.W = int(W)
        self.station = int(station)
        self.dir = os.path.join(root, arm, f"W{self.W}", method,
                                f"st{self.station}")
        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"no {arm} cache for {method} W={W} station={station}. "
                f"Run precompute first.\n  expected: {meta_path}")
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.K = int(self.meta["K"])
        self.depth = int(self.meta["depth"])

    # ---------------------------------------------------------------- checks
    def assert_matches(self, params: Optional[dict]) -> None:
        """Refuse a cache built with different decomposition parameters."""
        if params is None:
            return
        stored = self.meta.get("params", {})
        if stored != params:
            raise RuntimeError(
                f"cache for {self.method} W={self.W} st{self.station} was built "
                f"with {stored} but {params} was requested. Rebuild the cache "
                f"or pass the stored parameters.")

    # ------------------------------------------------------------------ data
    def origins(self, split: str) -> np.ndarray:
        return np.asarray(self.meta["origins"][split], dtype=int)

    def features(self, split: str, look_back: int,
                 rows: Optional[np.ndarray] = None) -> np.ndarray:
        """
        (n_selected, look_back, K) float32.

        Chunks are concatenated in order and sliced to `look_back` from the
        tail, which is exactly equal to computing at that depth directly.

        `rows` selects a subset by position within the split, applied while
        reading so that unwanted rows are never materialised. The hyperparameter
        search uses this: reading the full training set for every trial costs
        both the memory and the time, and neither buys anything.
        """
        if look_back > self.depth:
            raise ValueError(
                f"look_back={look_back} exceeds cached depth {self.depth}")
        paths = sorted(glob.glob(os.path.join(self.dir, f"{split}_*.npy")))
        if not paths:
            raise FileNotFoundError(
                f"no {split} chunks in {self.dir} — precompute incomplete")

        expected = len(self.origins(split))

        if rows is None:
            blocks = [np.load(p, mmap_mode="r")[..., -look_back:] for p in paths]
            arr = np.concatenate([np.asarray(b, dtype=np.float32)
                                  for b in blocks])
            if len(arr) != expected:
                raise RuntimeError(
                    f"{self.dir}/{split}: {len(arr)} rows but {expected} "
                    f"origins. The precompute stage is incomplete.")
        else:
            rows = np.asarray(sorted(rows), dtype=int)
            if len(rows) and (rows[0] < 0 or rows[-1] >= expected):
                raise IndexError(
                    f"{self.dir}/{split}: row selection outside "
                    f"[0, {expected})")
            blocks, offset = [], 0
            for p in paths:
                mm = np.load(p, mmap_mode="r")
                n = len(mm)
                local = rows[(rows >= offset) & (rows < offset + n)] - offset
                if len(local):
                    blocks.append(np.asarray(mm[local][..., -look_back:],
                                             dtype=np.float32))
                offset += n
            if offset != expected:
                raise RuntimeError(
                    f"{self.dir}/{split}: {offset} rows but {expected} "
                    f"origins. The precompute stage is incomplete.")
            arr = (np.concatenate(blocks) if blocks
                   else np.empty((0, self.K, look_back), dtype=np.float32))

        return np.transpose(arr, (0, 2, 1))          # -> (n, look_back, K)


# =============================================================================
class WindowDataset(Dataset):
    """Tensor pair for one split. Normalisation is applied by the caller."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# =============================================================================
class ChannelScaler:
    """
    Per-channel min-max scaling, fitted on training data only.

    Kept as an object rather than loose arrays so that the inverse transform
    used at evaluation time cannot accidentally be applied with statistics from
    a different split.
    """

    def __init__(self):
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ChannelScaler":
        flat = X.reshape(-1, X.shape[-1])
        self.min_ = flat.min(axis=0)
        self.max_ = flat.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rng = np.where((self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_)
        return ((X - self.min_) / rng).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def state(self) -> dict:
        return {"min": self.min_.tolist(), "max": self.max_.tolist()}


class TargetScaler:
    """Scalar min-max for the target, with an inverse for reporting in m/s."""

    def __init__(self):
        self.min_ = 0.0
        self.max_ = 1.0

    def fit(self, y: np.ndarray) -> "TargetScaler":
        self.min_, self.max_ = float(y.min()), float(y.max())
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        rng = self.max_ - self.min_ or 1.0
        return ((y - self.min_) / rng).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        rng = self.max_ - self.min_ or 1.0
        return y * rng + self.min_

    def state(self) -> dict:
        return {"min": self.min_, "max": self.max_}


# =============================================================================
class CausalDataModule:
    """
    Assembles train/val/test loaders for one (method, W, look_back) setting,
    pooled across stations.

    Targets are the RAW series, so metrics are reported in m/s without any
    reconstruction step — the component-space versus physical-units gap that
    the companion paper's reviewers flagged cannot arise here.
    """

    def __init__(self, method: str, W: int, look_back: int,
                 stations: Sequence[int], arm: str = "causal",
                 horizon: int = MAX_HORIZON,
                 batch_size: int = 64, cache_root: str = CACHE_DIR,
                 decomp_params: Optional[dict] = None, seed: int = SEED,
                 subsample: Optional[Dict[str, int]] = None):
        self.method = method
        self.arm = arm
        self.W = int(W)
        self.look_back = int(look_back)
        self.stations = list(stations)
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        raw = load_series()
        self.series = {s: np.asarray(raw[s], dtype=np.float64)
                       for s in self.stations}

        self.caches = {}
        for s in self.stations:
            cache = CausalFeatureCache(method, W, s, arm, cache_root)
            cache.assert_matches(decomp_params)
            self.caches[s] = cache

        ks = {c.K for c in self.caches.values()}
        if len(ks) != 1:
            raise RuntimeError(
                f"stations disagree on channel count: {ks}. Rebuild the cache.")
        self.K = ks.pop()

        self.feature_scaler = ChannelScaler()
        self.target_scaler = TargetScaler()
        self._data: Dict[str, tuple] = {}
        self.subsample = dict(subsample or {})
        self._rows: Dict[int, Dict[str, Optional[np.ndarray]]] = {}

    # ------------------------------------------------------------ subsampling
    def _rows_for(self, station: int, split: str) -> Optional[np.ndarray]:
        """
        Which rows of this split to use, or None for all of them.

        Drawn uniformly across the whole split rather than taken as a leading
        block. Wind is strongly seasonal, so the first year of the training
        region is one season cycle under one regime; a search tuned on it is
        tuned on that year. A spread sample of the same size costs the same and
        sees every season.

        The draw depends only on the global seed and the station, never on the
        arm or the decomposition. Both arms and all six methods therefore train
        on exactly the same origins, which is what keeps the comparison between
        them readable.
        """
        cap = self.subsample.get(split)
        if not cap:
            return None
        key = self._rows.setdefault(station, {})
        if split in key:
            return key[split]

        n = len(self.caches[station].origins(split))
        if n <= cap:
            key[split] = None
        else:
            rng = np.random.RandomState(self.seed + 1000 * station)
            key[split] = np.sort(rng.choice(n, size=cap, replace=False))
        return key[split]

    # ----------------------------------------------------------------- build
    def _assemble(self, split: str):
        Xs, ys = [], []
        for s in self.stations:
            cache = self.caches[s]
            rows = self._rows_for(s, split)
            X = cache.features(split, self.look_back, rows=rows)
            origins = cache.origins(split)
            if rows is not None:
                origins = origins[rows]
            x_raw = self.series[s]
            y = np.stack([x_raw[origins + h]
                          for h in range(1, self.horizon + 1)], axis=1)
            Xs.append(X)
            ys.append(y.astype(np.float32))
        return np.concatenate(Xs), np.concatenate(ys)

    def setup(self) -> "CausalDataModule":
        """Load every split and fit scalers on TRAIN only."""
        Xtr, ytr = self._assemble("train")
        self.feature_scaler.fit(Xtr)
        self.target_scaler.fit(ytr)

        self._data["train"] = (self.feature_scaler.transform(Xtr),
                               self.target_scaler.transform(ytr), ytr)
        for split in ("val", "test"):
            X, y = self._assemble(split)
            self._data[split] = (self.feature_scaler.transform(X),
                                 self.target_scaler.transform(y), y)
        return self

    # --------------------------------------------------------------- loaders
    def loader(self, split: str, shuffle: Optional[bool] = None) -> DataLoader:
        if not self._data:
            self.setup()
        X, y_scaled, _ = self._data[split]
        if shuffle is None:
            shuffle = (split == "train")
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(WindowDataset(X, y_scaled),
                          batch_size=self.batch_size, shuffle=shuffle,
                          generator=g if shuffle else None)

    def raw_targets(self, split: str) -> np.ndarray:
        """Unscaled targets in m/s, for metric computation."""
        if not self._data:
            self.setup()
        return self._data[split][2]

    def persistence(self, split: str) -> np.ndarray:
        """
        Naive baseline: the forecast for every horizon step is the value
        observed at the origin. Required as a reference in every table.
        """
        preds = []
        for s in self.stations:
            origins = self.caches[s].origins(split)
            last = self.series[s][origins]
            preds.append(np.repeat(last[:, None], self.horizon, axis=1))
        return np.concatenate(preds).astype(np.float32)

    def describe(self) -> dict:
        if not self._data:
            self.setup()
        return {
            "arm": self.arm,
            "method": self.method, "W": self.W, "look_back": self.look_back,
            "K": self.K, "horizon": self.horizon,
            "stations": self.stations,
            "n_train": len(self._data["train"][0]),
            "n_val": len(self._data["val"][0]),
            "n_test": len(self._data["test"][0]),
            "feature_scaler": self.feature_scaler.state(),
            "target_scaler": self.target_scaler.state(),
        }
