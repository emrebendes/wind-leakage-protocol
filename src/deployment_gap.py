# -*- coding: utf-8 -*-
"""
Deployment gap — a resumable Stage. No training.
================================================

The leaky arm trains on features built from a full-series decomposition, and
reports a test score computed from the same kind of features. This stage asks
the operational question about those trained models:

    what do they deliver when given only the information an operator has at
    the forecast origin?

Nothing is retrained. A leaky-arm model is loaded and evaluated on the CAUSAL
test features — the same origins, the same channel count, the same target, and
the same architecture. Only the information content of the input changes.

Why the models come from the new leaky arm rather than the previous study
------------------------------------------------------------------------
The previous models were trained one-per-component under a different
architecture, a different tuning protocol and a decomposition whose
hyperparameters had themselves seen the test set. Comparing them with anything
would confound at least four differences at once. Both arms of this study share
every line of code except the moment the decomposition is computed, so a
difference here is attributable to that alone.

In the pilot this produced RMSE 0.1327 reported against 1.6265 deployed, with
persistence at 0.5418 — the model was three times worse than doing nothing.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (ALL_ARMS, DECOMPOSITION_METHODS, DEVICE, FINAL_SEEDS,
                    FORECAST_HORIZONS, LEDGER_DB, MAX_HORIZON, MODELS,
                    MODELS_DIR, NUM_STATIONS, RESULTS_DIR)
from dataset import CausalDataModule
from metrics import MetricSuite, PersistenceBaseline, SignificanceTest
from models import build_model
from optimize import OptunaStage
from pipeline import Stage
from results import ResultStore, RunSpec
from trainer import Trainer


class DeploymentGapStage(Stage):
    """Evaluates leaky-trained models under causal inputs."""

    name = "deployment_gap"
    unit_name = "evaluation"     # inference only
    description = "what a leaky-trained model delivers on causal inputs"

    SOURCE_ARM = "leaky"
    TARGET_ARM = "deployment_gap"

    def __init__(self, methods, models, seeds, stations,
                 ledger_db=LEDGER_DB, source_arm=SOURCE_ARM, **kw):
        super().__init__(ledger_db, **kw)
        # The same question is worth asking of the partition arm: the remedy
        # removes train->test contamination, but does a model trained under it
        # deliver anything on causally available inputs? Everything below is
        # written against self.SOURCE_ARM, so pointing it at another arm is
        # the whole change.
        #
        # The ledger stage and the store arm are renamed with it. Sharing them
        # would make the ledger count partition units as already done, and
        # would mix two different questions into one directory.
        self.SOURCE_ARM = source_arm
        if source_arm != type(self).SOURCE_ARM:
            self.TARGET_ARM = f"deployment_gap_{source_arm}"
            self.name = f"deployment_gap_{source_arm}"
        self.methods = [m for m in methods if m != "none"]
        self.models = list(models)
        self.seeds = list(seeds)
        self.stations = list(stations)
        self.store = ResultStore(RESULTS_DIR)
        self.suite = MetricSuite(FORECAST_HORIZONS)

    # ------------------------------------------------------------ Stage API
    def plan(self):
        return {f"{d}|{m}|seed{s}": dict(decomposition=d, model=m, seed=s)
                for d in self.methods for m in self.models for s in self.seeds}

    def preflight(self):
        """Every unit needs a trained leaky model and a causal cache."""
        missing = []
        for d in self.methods:
            for m in self.models:
                try:
                    OptunaStage.load_best(self.SOURCE_ARM, d, m)
                except FileNotFoundError:
                    missing.append(f"{d}/{m}")
        if missing:
            raise RuntimeError(
                f"the {self.SOURCE_ARM} arm has not been tuned for: "
                + ", ".join(missing) +
                f"\nRun `python run.py optuna --arm {self.SOURCE_ARM}` and "
                f"`python run.py train --arm {self.SOURCE_ARM}` first.")
        self.log(f"[{self.name}] {len(self.plan())} {self.SOURCE_ARM} "
                 f"runs to replay on causal inputs")

    def group_key(self, spec):
        return (spec["decomposition"], spec["model"])

    def setup_group(self, gkey, spec):
        """
        Both data modules for this configuration: the leaky one the model was
        trained on, and the causal one it will be evaluated on. They use the
        same W, the same look_back and the same origins.
        """
        d, m = gkey
        best = OptunaStage.load_best(self.SOURCE_ARM, d, m)
        W = int(best["W"])
        look_back = min(int(best["look_back"]), W)
        batch = int(best.get("batch_size", 64))

        leaky = CausalDataModule(d, W, look_back, self.stations,
                                 arm=self.SOURCE_ARM, horizon=MAX_HORIZON,
                                 batch_size=batch).setup()
        causal = CausalDataModule(d, W, look_back, self.stations, arm="causal",
                                  horizon=MAX_HORIZON, batch_size=batch).setup()

        if leaky.K != causal.K:
            raise RuntimeError(
                f"{d}/{m}: {self.SOURCE_ARM} cache has K={leaky.K} but causal has "
                f"K={causal.K}. The arms must agree on channel count for the "
                f"weights to transfer.")

        self.log(f"  {d}/{m}: W={W} look_back={look_back} K={leaky.K}")
        return best, leaky, causal

    def output_exists(self, key, spec):
        """A unit is only finished if its result is still in the store."""
        try:
            best = OptunaStage.load_best(self.SOURCE_ARM,
                                         spec["decomposition"], spec["model"])
        except FileNotFoundError:
            return None
        W = int(best["W"])
        target = RunSpec(
            arm=self.TARGET_ARM, decomposition=spec["decomposition"],
            model=spec["model"], W=W,
            look_back=min(int(best["look_back"]), W), seed=spec["seed"],
            horizon_set="-".join(map(str, FORECAST_HORIZONS)),
            extra={"source_arm": self.SOURCE_ARM,
                   "tuned": dict(sorted(best.items()))})
        return self.store.exists(target)

    def execute(self, key, spec):
        d, m, seed = spec["decomposition"], spec["model"], spec["seed"]
        best, leaky, causal = self.group(spec)
        W = int(best["W"])
        look_back = min(int(best["look_back"]), W)

        # `source` must fingerprint exactly as train_final did, or the
        # checkpoint path below will not be found. Both therefore carry the
        # tuned hyperparameters in the identity.
        tuned = {"tuned": dict(sorted(best.items()))}
        source = RunSpec(arm=self.SOURCE_ARM, decomposition=d, model=m, W=W,
                         look_back=look_back, seed=seed,
                         horizon_set="-".join(map(str, FORECAST_HORIZONS)),
                         extra=tuned)
        target = RunSpec(arm=self.TARGET_ARM, decomposition=d, model=m, W=W,
                         look_back=look_back, seed=seed,
                         horizon_set="-".join(map(str, FORECAST_HORIZONS)),
                         extra={"source_arm": self.SOURCE_ARM, **tuned})

        if self.store.exists(target):
            return {"skipped": True}

        ckpt = os.path.join(MODELS_DIR, self.SOURCE_ARM, d, m,
                            f"seed{seed}_{source.fingerprint()}.ckpt")
        best_ckpt = ckpt.replace(".ckpt", ".best.ckpt")
        if not os.path.exists(best_ckpt) and not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"no trained {self.SOURCE_ARM} model for {source.label()}\n"
                f"  {best_ckpt}\n"
                f"Run `python run.py train --arm {self.SOURCE_ARM}` first.")

        arch = {k: v for k, v in best.items()
                if k not in ("W", "look_back", "lr", "batch_size")}
        model = build_model(m, leaky.K, MAX_HORIZON, arch)
        trainer = Trainer(model, DEVICE, checkpoint_path=ckpt, seed=seed,
                          verbose=False)
        trainer.load_checkpoint()
        trainer._load_best()          # evaluate the early-stopped weights

        reported = self._score(trainer, leaky)      # what the leaky arm claims
        deployed = self._score(trainer, causal)     # what it actually delivers

        gap = {}
        for h in FORECAST_HORIZONS:
            r = reported["summary"]["per_horizon"].get(f"H{h}", {}).get("RMSE")
            dv = deployed["summary"]["per_horizon"].get(f"H{h}", {}).get("RMSE")
            p = deployed["persistence"]["per_horizon"].get(f"H{h}", {}).get("RMSE")
            if r and dv and p:
                gap[f"H{h}"] = {"reported": r, "deployed": dv,
                                "persistence": p, "ratio": dv / r,
                                "beats_persistence": dv < p}

        metrics = {
            "reported": reported["summary"],
            "deployed": deployed["summary"],
            "persistence": deployed["persistence"],
            "gap": gap,
            "vs_persistence_DM": deployed["dm"],
            "per_station": deployed["per_station"],
            "source": source.to_dict(),
            "model": model.describe(),
        }
        self.store.save(target, metrics,
                        arrays={"y_pred_deployed": deployed["y_pred"],
                                "y_pred_reported": reported["y_pred"],
                                "y_true": deployed["y_true"]})

        h1 = gap.get(f"H{FORECAST_HORIZONS[0]}", {})
        return {"H1_reported": h1.get("reported"),
                "H1_deployed": h1.get("deployed"),
                "H1_persistence": h1.get("persistence"),
                "beats_persistence": h1.get("beats_persistence")}

    # ------------------------------------------------------------- scoring
    def _score(self, trainer: Trainer, dm: CausalDataModule) -> dict:
        """Evaluate the loaded model on one data module, in m/s."""
        pred = dm.target_scaler.inverse(
            trainer.predict(dm.loader("test", shuffle=False)))
        y_true = dm.raw_targets("test")
        n = min(len(pred), len(y_true))
        pred, y_true = pred[:n], y_true[:n]
        last = dm.persistence("test")[:n, 0]

        station_ids = np.concatenate([
            np.full(len(dm.caches[s].origins("test")), s) for s in dm.stations
        ])[:n]

        pers = PersistenceBaseline.predict(last, y_true.shape[1])
        return {
            "summary": self.suite.summary(y_true, pred, last),
            "persistence": PersistenceBaseline.evaluate(y_true, last, self.suite),
            "per_station": self.suite.by_station(y_true, pred, last, station_ids),
            "dm": SignificanceTest.diebold_mariano(
                (y_true - pers).ravel(), (y_true - pred).ravel()),
            "y_pred": pred.astype(np.float32),
            "y_true": y_true.astype(np.float32),
        }


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=DeploymentGapStage.description)
    ap.add_argument("--methods", nargs="+", default=DECOMPOSITION_METHODS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=FINAL_SEEDS)
    ap.add_argument("--stations", type=int, nargs="+",
                    default=list(range(NUM_STATIONS)))
    ap.add_argument("--source-arm", default=DeploymentGapStage.SOURCE_ARM,
                    choices=[a for a in ALL_ARMS if a != "causal"],
                    help="which arm's trained models to replay on causal "
                         "inputs (default: leaky)")
    Stage.add_common_arguments(ap)
    args = ap.parse_args()

    stage = DeploymentGapStage(args.methods, args.models, args.seeds,
                               args.stations, source_arm=args.source_arm)
    stage.main_from_args(args)


if __name__ == "__main__":
    main()
