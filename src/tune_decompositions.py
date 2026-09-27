# -*- coding: utf-8 -*-
"""
Decomposition hyperparameter tuning — a resumable Stage.
========================================================

Selects K, alpha, wavelet, level and so on for each (method, W) pair.

Two things are different from the previous study, and both matter.

1. TRAINING DATA ONLY.
   The old code scored candidates on `signal[-8760:]`, the final year of the
   series. For a 51,144-sample record split 70/15/15 that window is 88% test
   data and contains no training data at all, so the decomposition parameters
   were chosen with the test set in view. Here candidates are scored on windows
   drawn from the training region and nowhere else.

2. THE TUNING REGIME MATCHES THE DEPLOYMENT REGIME.
   The old code scored a single 8,760-sample block, but the pipeline runs on
   W-sample windows. The optimum differs: on 8,760 samples the score falls
   monotonically with K so the search returns the top of the grid (K=8), while
   on 256-sample windows the optimum is interior (K=6). Parameters are
   therefore tuned on windows of exactly the W they will be used with.

Search space validity
---------------------
Every candidate is checked against the method's `min_window()` for this W.
A level-6 sym5 DWT needs 320 samples and is silently dropped at W=256 rather
than raising once per scored window.

One ledger unit is one (method, W) pair.

Usage
-----
    python tune_decompositions.py
    python tune_decompositions.py --methods vmd dwt --W 512
    python tune_decompositions.py --status --failures
"""

from __future__ import annotations

import argparse
import itertools
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atomicio import atomic_path, write_json
from config import (DATA_FILE, load_series, PARAMS_DIR, LEDGER_DB, W_CHOICES, SEED,
                    DECOMPOSITION_METHODS, DECOMP_TUNE_WINDOWS,
                    DECOMP_SCORE_WEIGHTS, NUM_STATIONS)
from decompositions import (get_decomposer, param_space, default_params,
                            composite_score, REGISTRY)
from causal_features import split_bounds
from pipeline import Stage

N_NUMERIC_STEPS = 6          # grid resolution for continuous parameters


# =============================================================================
def expand_space(space: dict, n_steps: int = N_NUMERIC_STEPS):
    """
    Turn a declared search space into a list of concrete parameter dicts.

        (lo, hi)   -> n_steps values (integer ranges stay integer)
        [a, b, c]  -> the listed choices
    """
    if not space:
        return [{}]
    axes = {}
    for key, spec in space.items():
        if isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = spec
            if isinstance(lo, int) and isinstance(hi, int):
                axes[key] = list(range(int(lo), int(hi) + 1))
            else:
                axes[key] = list(np.linspace(float(lo), float(hi), n_steps))
        elif isinstance(spec, (list, tuple)):
            axes[key] = list(spec)
        else:
            axes[key] = [spec]
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes.values())]


class TuneDecompositionStage(Stage):
    """Grid search over decomposition parameters, scored on training windows."""

    name = "tune_decomp"
    unit_name = "search"         # one (method, W) grid
    description = "decomposition hyperparameters, per W, on training windows"

    def __init__(self, methods, ws, stations, n_windows=DECOMP_TUNE_WINDOWS,
                 ledger_db=LEDGER_DB, **kw):
        super().__init__(ledger_db, **kw)
        self.methods = [m for m in methods if param_space(m)]   # skip "none"
        self.ws = list(ws)
        self.stations = list(stations)
        self.n_windows = int(n_windows)
        data = load_series()
        self.series = {s: np.asarray(data[s], dtype=np.float64)
                       for s in self.stations}

    # ------------------------------------------------------------ Stage API
    def plan(self):
        """
        One unit per CANDIDATE, not per (method, W).

        The candidate loop is embarrassingly parallel and used to run inside a
        single unit, serially. That capped the whole stage at the runtime of
        one unit: six remaining units kept six cores busy and left nine idle,
        and adding machines would not have helped because every machine would
        have taken one unit and waited the same wall clock. CEEMDAN at 126
        candidates x 200 windows is a day of that.

        Splitting per candidate turns the same work into ~1050 small units, so
        every available core is used — 15 locally, 52 on a hamsi node.
        """
        units = {}
        for m in self.methods:
            for W in self.ws:
                for i, params in enumerate(self.feasible_candidates(m, W)):
                    units[f"{m}|W{W}|c{i:04d}"] = dict(method=m, W=W,
                                                       index=i, params=params)
        return units

    def preflight(self):
        """Report how much of each search space survives the window constraint."""
        for m in self.methods:
            for W in self.ws:
                n_all = len(expand_space(param_space(m)))
                n_ok = len(self.feasible_candidates(m, W))
                if n_ok == 0:
                    raise RuntimeError(
                        f"{m} @ W={W}: no candidate satisfies the minimum "
                        f"window constraint. Raise W or widen the space.")
                self.log(f"  {m:8s} W={W:5d}: {n_ok}/{n_all} candidates feasible")

    def output_exists(self, key, spec):
        """
        The parameter file is the output.

        This matters when the project is copied to another machine: if the
        ledger travels with it, every unit arrives marked DONE and the stage
        reports complete without producing anything. Checking the file closes
        that hole — but do not copy `state/` between machines regardless.
        """
        return os.path.exists(os.path.join(
            PARAMS_DIR, f"W{spec['W']}", f"{spec['method']}.json"))

    def group_key(self, spec):
        """Windows are sampled once per W, not per candidate."""
        return spec["W"]

    def setup_group(self, gkey, spec):
        return self.training_windows(gkey)

    def output_exists(self, key, spec):
        return os.path.exists(self._partial_path(spec["method"], spec["W"],
                                                 spec["index"]))

    @staticmethod
    def _partial_path(method: str, W: int, index: int) -> str:
        return os.path.join(PARAMS_DIR, f"W{W}", "_partial", method,
                            f"{index:04d}.json")

    def execute(self, key, spec):
        """Score one candidate and record it. The winner is picked later."""
        method, W, index = spec["method"], spec["W"], spec["index"]
        windows = self.group(spec)

        t0 = time.time()
        score = self.score_candidate(method, spec["params"], windows)
        write_json(self._partial_path(method, W, index),
                   {"params": spec["params"], "median_score": score})

        self.log(f"  {method} W={W} candidate {index}: score "
                 f"{score:.5f}  ({time.time() - t0:.0f}s)  {spec['params']}")
        return {"score": score}

    def run(self, max_seconds=None):
        """Score the candidates, then decide the winners."""
        super().run(max_seconds=max_seconds)
        self.collect()

    def collect(self) -> None:
        """
        Turn completed candidate scores into one parameter file per (method, W).

        Only groups whose candidates are all present are written, so a partial
        run leaves no half-informed choice behind. Safe to call repeatedly and
        from several workers: the write is atomic and the inputs are fixed.
        """
        for m in self.methods:
            for W in self.ws:
                expected = len(self.feasible_candidates(m, W))
                paths = sorted(glob.glob(
                    self._partial_path(m, W, 0).replace("0000.json", "*.json")))
                if len(paths) < expected:
                    continue

                results = []
                for p in paths:
                    with open(p, encoding="utf-8") as f:
                        results.append(json.load(f))
                results = [r for r in results
                           if np.isfinite(r["median_score"])]
                if not results:
                    self.warn(f"{m} @ W={W}: every candidate failed")
                    continue

                results.sort(key=lambda r: r["median_score"])
                best = results[0]
                self.write_params(m, W, best["params"], results)

                spread = ((results[-1]["median_score"] - best["median_score"])
                          / best["median_score"] * 100)
                self.log(f"[{self.name}] {m} W={W}: best "
                         f"{best['median_score']:.5f} of {len(results)} "
                         f"candidates, spread {spread:.2f}%  {best['params']}")

                edge = self.boundary_report(m, best["params"])
                if edge:
                    self.warn(f"{m} W={W}: best parameters sit on the search "
                              f"boundary: {edge}. Widen the range, or justify "
                              f"the limit — see the spread above before "
                              f"deciding it matters.")

    # --------------------------------------------------------------- scoring
    def feasible_candidates(self, method: str, W: int):
        """Candidates whose minimum window fits inside W."""
        out = []
        for params in expand_space(param_space(method)):
            merged = {**default_params(method), **params}
            try:
                if REGISTRY[method](merged).min_window() <= W:
                    out.append(merged)
            except Exception:
                continue
        return out

    def training_windows(self, W: int):
        """
        Sample windows from the TRAINING region of every station.
        Never touches validation or test.
        """
        rng = np.random.RandomState(SEED)
        per_station = max(1, self.n_windows // len(self.stations))
        windows = []
        for s in self.stations:
            x = self.series[s]
            train_end, _ = split_bounds(len(x))
            hi = train_end - 1
            if hi <= W:
                continue
            ends = rng.randint(W - 1, hi, size=per_station)
            windows.extend(x[e - W + 1: e + 1] for e in ends)
        return windows

    def score_candidate(self, method: str, params: dict, windows) -> float:
        """
        Median composite score over the sampled windows. The median makes the
        result robust to a handful of pathological windows, which matters for
        VMD: its per-window optimisation occasionally fails to converge.
        """
        try:
            decomp = get_decomposer(method, params)
        except Exception:
            return float("inf")

        scores = []
        for w in windows:
            try:
                comps = decomp(w)
                s = composite_score(comps, w, DECOMP_SCORE_WEIGHTS)
                if np.isfinite(s):
                    scores.append(s)
            except Exception:
                continue
        if len(scores) < max(3, len(windows) // 4):
            return float("inf")          # too unstable to trust
        return float(np.median(scores))

    # ---------------------------------------------------------------- output
    @staticmethod
    def boundary_report(method: str, best: dict):
        """
        Flag parameters that landed on the edge of their search range. A
        reviewer of the companion paper asked exactly this ("why does the
        search keep choosing K=3?"), so it is checked automatically.
        """
        edge = []
        for k, spec in param_space(method).items():
            if isinstance(spec, tuple) and len(spec) == 2 and k in best:
                lo, hi = spec
                if np.isclose(float(best[k]), float(lo)):
                    edge.append(f"{k}=lower({lo})")
                elif np.isclose(float(best[k]), float(hi)):
                    edge.append(f"{k}=upper({hi})")
        return ", ".join(edge)

    @staticmethod
    def write_params(method: str, W: int, best: dict, results):
        out_dir = os.path.join(PARAMS_DIR, f"W{W}")
        os.makedirs(out_dir, exist_ok=True)

        write_json(os.path.join(out_dir, f"{method}.json"), best, indent=2)

        with atomic_path(os.path.join(out_dir,
                                      f"{method}_search.json")) as tmp:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=float)


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=TuneDecompositionStage.description)
    ap.add_argument("--methods", nargs="+", default=DECOMPOSITION_METHODS)
    ap.add_argument("--W", type=int, nargs="+", default=W_CHOICES)
    ap.add_argument("--stations", type=int, nargs="+",
                    default=list(range(NUM_STATIONS)))
    ap.add_argument("--n-windows", type=int, default=DECOMP_TUNE_WINDOWS)
    Stage.add_common_arguments(ap)
    args = ap.parse_args()

    stage = TuneDecompositionStage(args.methods, args.W, args.stations,
                                   n_windows=args.n_windows)
    stage.main_from_args(args)


if __name__ == "__main__":
    main()
