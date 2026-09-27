# -*- coding: utf-8 -*-
"""
Causal feature precompute — a resumable Stage.
==============================================

Builds, for every (method, W, station, split), the tensor

        (n_origins, K, LOOK_BACK_MAX)   float32

where row i holds the components of the trailing window ending at origin i.
Training reads this cache and slices `[..., -look_back:]`, so a smaller
look_back costs nothing extra.

One ledger unit is one chunk of origins:

    key = "{method}|W{W}|st{station}|{split}|{chunk:05d}"

The claim/resume loop lives in `pipeline.Stage`; this file supplies only the
plan and the work. A chunk writes its .npy first and is marked DONE second, so
a job killed mid-chunk loses at most that chunk.

Usage
-----
    python precompute_causal.py                    # everything
    python precompute_causal.py --methods dwt vmd  # subset
    python precompute_causal.py --W 512
    python precompute_causal.py --status --failures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (ALL_ARMS, ARMS, DATA_FILE, load_series, CACHE_DIR, PARAMS_DIR, LEDGER_DB,
                    LOOK_BACK_MAX, W_CHOICES, DECOMPOSITION_METHODS,
                    MAX_TRAIN_ORIGINS, NUM_STATIONS, SEED)
from atomicio import write_json, write_npy
from decompositions import get_decomposer, default_params
from causal_features import get_builder, valid_origins
from pipeline import Stage

CHUNK = 2000

# Length of the series handed to the contract guard. The guard permutes and
# rebuilds, so nothing it computes can be cached; on the full record that is
# hours per group for the EMD family. `block_of` and `valid_origins` derive
# their boundaries from len(x), so a shorter series gives proportionally
# shorter blocks and tests the same property in seconds.
#   4096 -> train 2867, val 614, test 614; enough history for W=1024 and
#   enough future for the within-block leak check.
VERIFY_SERIES_LEN = 4096
SPLITS = ("train", "val", "test")


class PrecomputeStage(Stage):
    """Builds the causal feature cache."""

    name = "precompute"
    unit_name = "chunk"          # 2000 origins of one split
    description = "per-origin causal decomposition features"

    def __init__(self, methods, ws, stations, arms=ARMS,
                 ledger_db=LEDGER_DB, **kw):
        super().__init__(ledger_db, **kw)
        self.arms = list(arms)
        self.methods = list(methods)
        self.ws = list(ws)
        self.stations = list(stations)
        data = load_series()
        self.series = {s: np.asarray(data[s], dtype=np.float64)
                       for s in self.stations}

    # ----------------------------------------------------------- parameters
    @staticmethod
    def load_params(method: str, W: int) -> dict:
        """Tuned parameters for this (method, W); falls back to defaults."""
        path = os.path.join(PARAMS_DIR, f"W{W}", f"{method}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default_params(method)

    # --------------------------------------------------------------- layout
    @staticmethod
    def cache_dir(arm: str, method: str, W: int, station: int) -> str:
        return os.path.join(CACHE_DIR, arm, f"W{W}", method, f"st{station}")

    @classmethod
    def chunk_path(cls, arm, method, W, station, split, ci) -> str:
        return os.path.join(cls.cache_dir(arm, method, W, station),
                            f"{split}_{ci:05d}.npy")

    @classmethod
    def meta_path(cls, arm, method, W, station) -> str:
        return os.path.join(cls.cache_dir(arm, method, W, station), "meta.json")

    def origins(self, station: int, W: int, split: str, method: str):
        """
        Origins for one split. The TRAINING set may be sub-sampled for the
        expensive EMD-family methods; validation and test never are, so every
        method is scored on identical evaluation points.
        """
        o = valid_origins(len(self.series[station]), W, split)
        cap = MAX_TRAIN_ORIGINS.get(method)
        if split == "train" and cap is not None and len(o) > cap:
            rng = np.random.RandomState(SEED)
            o = np.sort(rng.choice(o, size=cap, replace=False))
        return o

    # ------------------------------------------------------------ Stage API
    def plan(self):
        units = {}
        for arm in self.arms:
            for m in self.methods:
                for W in self.ws:
                    for s in self.stations:
                        for split in SPLITS:
                            o = self.origins(s, W, split, m)
                            n_chunks = max(1, int(np.ceil(len(o) / CHUNK)))
                            for ci in range(n_chunks):
                                key = (f"{arm}|{m}|W{W}|st{s}|{split}|"
                                       f"{ci:05d}")
                                units[key] = dict(arm=arm, method=m, W=W,
                                                  station=s, split=split,
                                                  chunk=ci)
        return units

    def preflight(self):
        """
        Reject impossible (method, W) combinations before any work starts,
        rather than discovering them chunk by chunk. FeatureBuilder raises if
        W is below the method's minimum window.
        """
        bad = []
        for arm in self.arms:
            for m in self.methods:
                for W in self.ws:
                    try:
                        get_builder(arm, get_decomposer(
                            m, self.load_params(m, W)), W, LOOK_BACK_MAX)
                    except ValueError as e:
                        bad.append(f"  {arm}/{m} @ W={W}: {e}")
        if bad:
            raise RuntimeError(
                "incompatible method/window combinations:\n" + "\n".join(bad))
        self.log(f"[{self.name}] preflight ok: {len(self.arms)} arms x "
                 f"{len(self.methods)} methods x {len(self.ws)} windows")

    def group_key(self, spec):
        """Builder and metadata are shared by every chunk of a station."""
        return (spec["arm"], spec["method"], spec["W"], spec["station"])

    def setup_group(self, gkey, spec):
        arm, method, W, station = gkey
        params = self.load_params(method, W)
        builder = get_builder(arm, get_decomposer(method, params), W,
                              LOOK_BACK_MAX)

        # Each builder is checked against its own contract before any chunk is
        # written: the causal one must ignore the future, the leaky one must
        # not — a leaky arm that accidentally became causal would make the
        # whole comparison vacuous.
        #
        # ONCE PER (arm, method, W), AND ON A SHORT SERIES
        #     The contract is a property of the builder class and the
        #     decomposition. It does not vary by station, and the guard is
        #     expensive in a way that is easy to miss: it permutes the series
        #     and rebuilds, so every trial is a decomposition of a block that
        #     has never been seen and cannot be cached.
        #
        #     Only half of it is expensive, which is why it was easy to
        #     misjudge. assert_block_isolated permutes OUTSIDE the block, so
        #     the block's contents are unchanged, the hash matches and the
        #     decomposition is a cache hit. assert_leaks_within permutes
        #     INSIDE it, so every trial is a genuine recomputation.
        #
        #     Measured on job 6288323: ceemdan over a 35502-sample partition
        #     block takes ~1266 s, and with two origins that is ~40 minutes of
        #     verification per group before a single chunk is written. There
        #     are 72 groups and 24 workers each did their own. With
        #     VERIFY_SERIES_LEN the
        #     same assertions run on proportionally smaller blocks — same
        #     data, same decomposer, same code path — in seconds, and the
        #     receipt stops the other seven stations from repeating them.
        self._verify_once(builder, arm, method, W, station)

        meta = self._ensure_meta(builder, arm, method, W, station)
        return builder, meta

    def output_exists(self, key, spec):
        """The chunk file is the output; a deleted cache must be rebuilt."""
        return os.path.exists(self.chunk_path(
            spec["arm"], spec["method"], spec["W"], spec["station"],
            spec["split"], spec["chunk"]))

    def execute(self, key, spec):
        builder, meta = self.group(spec)
        split, ci = spec["split"], spec["chunk"]
        path = self.chunk_path(spec["arm"], spec["method"], spec["W"],
                               spec["station"], split, ci)
        if os.path.exists(path):
            return {"skipped": True}

        origins = np.asarray(meta["origins"][split], dtype=int)
        sub = origins[ci * CHUNK: (ci + 1) * CHUNK]
        block = builder.for_origins(self.series[spec["station"]], sub,
                                    k_target=meta["K"])

        write_npy(path, block)       # write to a private temp, then rename
        return {"n": int(len(sub)), "shape": list(block.shape)}

    # --------------------------------------------------------------- helpers
    def _global_k(self, builder, arm, method, W) -> int:
        """
        Channel count for (arm, method, W), the SAME for every station.

        This used to be decided per station. For dwt, vmd and none that made
        no difference — the decomposer knows K in advance and every station
        agreed. The EMD family returns as many IMFs as the signal supports, so
        one station settled on 5 channels and another on 1, and the loader,
        which stacks stations into one tensor, refused:

            RuntimeError: stations disagree on channel count: {1, 5}

        Probing every station and taking the minimum keeps the existing
        convention — harmonisation only ever folds components together, never
        invents them — and makes the result independent of which station a
        worker happened to start with.

        The probe is expensive for the EMD family, so the answer is written
        next to the cache and reused. Several workers may compute it at once;
        the probe is seeded and deterministic, so they all agree and whichever
        file lands last is correct.
        """
        k = builder.decomposer.expected_k()
        if k is not None:
            return int(k)

        shared = os.path.join(CACHE_DIR, arm, f"W{W}", method, "K.json")
        if os.path.exists(shared):
            try:
                with open(shared, encoding="utf-8") as f:
                    return int(json.load(f)["K"])
            except (json.JSONDecodeError, UnicodeDecodeError, OSError, KeyError):
                pass                      # unreadable: fall through and redo

        per_station = {}
        for st in self.stations:
            per_station[int(st)] = int(builder.component_count(
                self.series[st], self.origins(st, W, "train", method)))
        k = min(per_station.values())
        self.log(f"  {arm} {method} W={W}: K={k} "
                 f"(per station {per_station})")

        os.makedirs(os.path.dirname(shared), exist_ok=True)
        write_json(shared, {"arm": arm, "method": method, "W": int(W),
                            "K": int(k), "per_station": per_station})
        return int(k)

    def _verify_once(self, builder, arm, method, W, station) -> None:
        """Run the contract check for this (arm, method, W), or note it ran."""
        receipt = os.path.join(self.cache_dir(arm, method, W, station), "..",
                               "verified.json")
        receipt = os.path.normpath(receipt)
        if os.path.exists(receipt):
            return

        t0 = time.time()
        builder.verify_sample(self.series[station],
                              self.origins(station, W, "train", method),
                              n=2, max_len=VERIFY_SERIES_LEN)
        elapsed = time.time() - t0
        self.log(f"  {arm} builder verified: {method} W={W} st{station} "
                 f"({elapsed:.0f}s, {VERIFY_SERIES_LEN}-sample series)")

        os.makedirs(os.path.dirname(receipt), exist_ok=True)
        write_json(receipt, {
            "arm": arm, "method": method, "W": int(W),
            "verified_on_station": int(station),
            "series_len": VERIFY_SERIES_LEN,
            "origins": 2, "seconds": round(elapsed, 1),
            "note": ("contract check only; the assertions are structural and "
                     "do not depend on the block length. Delete this file to "
                     "force a re-check."),
            "created": time.time(),
        })

    def _ensure_meta(self, builder, arm, method, W, station) -> dict:
        """Fix K once per (arm, method, W, station) so all chunks agree."""
        mp = self.meta_path(arm, method, W, station)
        if os.path.exists(mp):
            try:
                with open(mp, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                # An unreadable meta file is treated as absent and rebuilt.
                # It is cheap to regenerate, and refusing to run because of a
                # corrupt one turns a recoverable state into a manual cleanup —
                # which is what happened here, when files left behind by an
                # earlier bug failed every unit that touched them.
                self.log(f"  rebuilding unreadable meta: "
                         f"{os.path.relpath(mp)} ({type(exc).__name__})")

        os.makedirs(os.path.dirname(mp), exist_ok=True)
        k = self._global_k(builder, arm, method, W)

        meta = {
            "arm": arm, "method": method, "W": W, "station": station,
            "K": int(k), "depth": int(LOOK_BACK_MAX),
            "params": builder.decomposer.params,
            "signature": builder.signature(),
            "builder": builder.describe(),
            "origins": {sp: self.origins(station, W, sp, method).tolist()
                        for sp in SPLITS},
            "created": time.time(),
        }
        # Several workers reach this at once for the same group and each
        # writes an identical file. The content is deterministic, so whichever
        # commits last is correct; only the temporary name has to be private.
        write_json(mp, meta)
        return meta


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=PrecomputeStage.description)
    ap.add_argument("--methods", nargs="+", default=DECOMPOSITION_METHODS)
    ap.add_argument("--W", type=int, nargs="+", default=W_CHOICES)
    ap.add_argument("--stations", type=int, nargs="+",
                    default=list(range(NUM_STATIONS)))
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ALL_ARMS)
    Stage.add_common_arguments(ap)
    args = ap.parse_args()

    stage = PrecomputeStage(args.methods, args.W, args.stations,
                            arms=args.arms)
    stage.main_from_args(args)


if __name__ == "__main__":
    main()
