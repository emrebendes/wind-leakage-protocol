# -*- coding: utf-8 -*-
"""
Final training — a resumable Stage.
===================================

One ledger unit is one (decomposition, model, seed) run. Three seeds per
configuration give the replicates that the significance testing needs; the
previous study had none and a reviewer raised it.

Two layers of resume protect a long job:

  * the ledger recovers the unit a killed worker was holding, and
  * `Trainer` checkpoints every epoch, so the unit restarts near where it
    stopped rather than from scratch.

Results go through `ResultStore`, which refuses to overwrite an existing run.
With 6 x 7 x 3 = 126 units writing concurrently that guarantee is doing real
work: a re-submitted array job skips what is finished instead of destroying it.

Every result carries the persistence baseline and per-station metrics, so no
table in the paper can quote an improvement without its reference.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (ALL_ARMS, ARMS, DECOMPOSITION_METHODS, DEVICE,
                    FINAL_EPOCHS, FINAL_PATIENCE, PARTITION_ARM,
                    FINAL_SEEDS, FORECAST_HORIZONS, LEDGER_DB, MAX_HORIZON,
                    MODELS, MODELS_DIR, NUM_STATIONS, RESULTS_DIR)
from dataset import CausalDataModule
from metrics import MetricSuite, PersistenceBaseline, SignificanceTest
from models import build_model
from optimize import OptunaStage
from pipeline import Stage
from results import ResultStore, RunSpec
from trainer import Trainer


class FinalTrainingStage(Stage):
    """Trains the tuned configurations and stores evaluated results."""

    name = "train_final"
    unit_name = "model"          # one (method, model, seed)
    description = "final training with replicate seeds, causal arm"

    def __init__(self, methods, models, seeds, stations, arm="causal",
                 epochs=FINAL_EPOCHS, ledger_db=LEDGER_DB, **kw):
        super().__init__(ledger_db, **kw)
        self.methods = list(methods)
        self.models = list(models)
        self.seeds = list(seeds)
        self.stations = list(stations)
        self.arm = arm
        self.epochs = int(epochs)
        self.store = ResultStore(RESULTS_DIR)
        self.suite = MetricSuite(FORECAST_HORIZONS)

    # ------------------------------------------------------------ Stage API
    def plan(self):
        return {f"{self.arm}|{d}|{m}|seed{s}":
                dict(arm=self.arm, decomposition=d, model=m, seed=s)
                for d in self.methods for m in self.models for s in self.seeds}

    def preflight(self):
        """Fail early if a configuration has not been tuned yet."""
        missing = []
        for d in self.methods:
            for m in self.models:
                try:
                    OptunaStage.load_best(self.arm, d, m)
                except FileNotFoundError:
                    missing.append(f"{d}/{m}")
        if missing:
            hint = ""
            if self.arm == PARTITION_ARM and any(k.startswith("none/")
                                                 for k in missing):
                # The partition arm has no 'none' cell by design: with the
                # identity decomposition it is bit-identical to the causal arm
                # (tests/test_causality.py proves it for every origin). Without
                # this hint the message sends the user to run an Optuna search
                # for a cell that will never have a cache.
                hint = ("\n'none' is intentionally absent from the partition "
                        "arm — it cannot differ from causal. Name the methods:"
                        "\n  python run.py train --arm partition "
                        "--methods dwt vmd emd eemd ceemdan")
            raise RuntimeError(
                f"no tuned hyperparameters ({self.arm} arm) for: "
                + ", ".join(missing) +
                f"\nRun `python run.py optuna --arm {self.arm}` first." + hint)
        self.log(f"[{self.name}] all {len(self.methods) * len(self.models)} "
                 f"configurations have tuned parameters")

    def group_key(self, spec):
        """Seeds of one configuration share the data module."""
        return (spec["decomposition"], spec["model"])

    def setup_group(self, gkey, spec):
        d, m = gkey
        best = OptunaStage.load_best(self.arm, d, m)
        W = int(best["W"])
        look_back = min(int(best["look_back"]), W)
        batch = int(best.get("batch_size", 64))

        dm = CausalDataModule(d, W, look_back, self.stations, arm=self.arm,
                              horizon=MAX_HORIZON, batch_size=batch).setup()
        self.log(f"  data ready [{self.arm}]: {d}/{m} "
                 f"{dm.describe()['n_train']} train windows, K={dm.K}, "
                 f"W={W}, look_back={look_back}")
        return best, dm

    @staticmethod
    def run_spec(arm: str, spec: dict, best: dict) -> RunSpec:
        """
        The run's identity, including the tuned hyperparameters.

        Without them, re-running the search and then re-training produces a
        spec that fingerprints identically to the previous result whenever W
        and look_back happen to match — and `exists()` then reports "already
        stored" and keeps the OLD numbers, trained with the OLD architecture.
        Nothing errors; the stale result simply survives. RunSpec's own
        docstring states the rule this was breaking: every field that could
        change the numbers belongs in the identity.
        """
        W = int(best["W"])
        return RunSpec(arm=arm, decomposition=spec["decomposition"],
                       model=spec["model"], W=W,
                       look_back=min(int(best["look_back"]), W),
                       seed=spec["seed"],
                       horizon_set="-".join(map(str, FORECAST_HORIZONS)),
                       extra={"tuned": dict(sorted(best.items()))})

    def output_exists(self, key, spec):
        """A unit is only finished if its result is still in the store."""
        try:
            best = OptunaStage.load_best(self.arm, spec["decomposition"],
                                         spec["model"])
        except FileNotFoundError:
            return None                      # cannot tell; leave it alone
        return self.store.exists(self.run_spec(self.arm, spec, best))

    def execute(self, key, spec):
        d, m, seed = spec["decomposition"], spec["model"], spec["seed"]
        best, dm = self.group(spec)
        W = int(best["W"])
        look_back = min(int(best["look_back"]), W)
        run = self.run_spec(self.arm, spec, best)

        if self.store.exists(run):
            self.log(f"  already stored: {run.label()}")
            return {"skipped": True}

        arch = {k: v for k, v in best.items()
                if k not in ("W", "look_back", "lr", "batch_size")}
        model = build_model(m, dm.K, MAX_HORIZON, arch)

        ckpt = os.path.join(MODELS_DIR, self.arm, d, m,
                            f"seed{seed}_{run.fingerprint()}.ckpt")
        trainer = Trainer(model, DEVICE, checkpoint_path=ckpt, seed=seed,
                          verbose=self.verbose)

        # Name the device. The search runs on CPU and training on GPU, so a
        # silent fallback to CPU here — a driver problem, a node without a
        # card, hamsi instead of a CUDA queue — would not announce itself and
        # would only show up as a job that takes forty times too long.
        #
        # The physical card is named too. Every worker was handed one card
        # through CUDA_VISIBLE_DEVICES, so torch calls it "cuda" in all four
        # of them and the log alone cannot show whether the four workers are
        # spread over four cards or queueing on one. The environment variable
        # is the ground truth and it costs nothing to print.
        card = os.environ.get("CUDA_VISIBLE_DEVICES")
        where = f"{DEVICE}" + (f" (kart {card})" if card else "")
        self.log(f"[{self.name}] {run.label()}  "
                 f"({model.n_parameters:,} parameters)  on {where}")
        fit = trainer.fit(dm.loader("train"), dm.loader("val"),
                          epochs=self.epochs, patience=FINAL_PATIENCE,
                          lr=float(best.get("lr", 1e-3)))

        metrics = self._evaluate(trainer, dm, model, fit, run)
        self.store.save(run, metrics,
                        arrays={"y_pred": metrics.pop("_y_pred"),
                                "y_true": metrics.pop("_y_true")})
        return {"mean_RMSE": metrics["test"]["mean_RMSE"],
                "beats_persistence": metrics["test"]["beats_persistence"]}

    # ------------------------------------------------------------ evaluation
    def _evaluate(self, trainer, dm, model, fit, run: RunSpec) -> dict:
        """Test-set metrics in m/s, against persistence, with per-station detail."""
        y_pred_scaled = trainer.predict(dm.loader("test", shuffle=False))
        y_pred = dm.target_scaler.inverse(y_pred_scaled)
        y_true = dm.raw_targets("test")
        n = min(len(y_pred), len(y_true))
        y_pred, y_true = y_pred[:n], y_true[:n]
        last = dm.persistence("test")[:n, 0]

        station_ids = np.concatenate([
            np.full(len(dm.caches[s].origins("test")), s) for s in dm.stations
        ])[:n]

        summary = self.suite.summary(y_true, y_pred, last)
        baseline = PersistenceBaseline.evaluate(y_true, last, self.suite)

        # `last` is (n,) while the forecasts are (n, H); broadcast explicitly
        # rather than relying on numpy, which would raise here.
        pers_pred = PersistenceBaseline.predict(last, y_true.shape[1])
        dm_test = SignificanceTest.diebold_mariano(
            (y_true - pers_pred).ravel(), (y_true - y_pred).ravel())

        return {
            "test": summary,
            "persistence": baseline,
            "vs_persistence_DM": dm_test,
            "per_station": self.suite.by_station(y_true, y_pred, last,
                                                 station_ids),
            "training": {k: v for k, v in fit.items() if k != "history"},
            "history": fit["history"],
            "model": model.describe(),
            "data": dm.describe(),
            "_y_pred": y_pred.astype(np.float32),
            "_y_true": y_true.astype(np.float32),
        }


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=FinalTrainingStage.description)
    ap.add_argument("--methods", nargs="+", default=DECOMPOSITION_METHODS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=FINAL_SEEDS)
    ap.add_argument("--stations", type=int, nargs="+",
                    default=list(range(NUM_STATIONS)))
    ap.add_argument("--arm", default="causal", choices=ALL_ARMS)
    ap.add_argument("--epochs", type=int, default=FINAL_EPOCHS)
    Stage.add_common_arguments(ap)
    args = ap.parse_args()

    stage = FinalTrainingStage(args.methods, args.models, args.seeds,
                               args.stations, arm=args.arm, epochs=args.epochs)
    stage.main_from_args(args)


if __name__ == "__main__":
    main()
