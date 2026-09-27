# -*- coding: utf-8 -*-
"""
Hyperparameter search — a resumable Stage.
==========================================

One ledger unit is one (decomposition, model) study. Optuna keeps its own
trial history in a SQLite storage, so a study interrupted mid-search resumes
from the trial it reached: the ledger recovers the unit, and Optuna recovers
the trials inside it.

W IS SEARCHED HERE, AND IT MEANS DIFFERENT THINGS IN DIFFERENT ARMS.
--------------------------------------------------------------------
W is the trailing history handed to the decomposition at each forecast origin.
That quantity only exists in the causal arm. The leaky arm decomposes the whole
record and the partition arm decomposes a whole split, so neither of them reads
W when building a feature: for a fixed origin they return the same numbers at
every W. Measured, not assumed:

    same origin, W = 256 / 512 / 1024, max abs difference in the features
        causal      0.00e+00   1.88e-01   3.05e-01     <- W changes the features
        leaky       0.00e+00   0.00e+00   0.00e+00     <- identical
        partition   0.00e+00   0.00e+00   0.00e+00     <- identical

W is still searched in all three arms, and deliberately so: the arms must
receive the same search space and the same budget, or a difference in results
becomes partly a difference in tuning. Its residual effect outside the causal
arm is the origin floor -- valid_origins starts at W-1, so W=1024 trains on
34455 origins per station where W=256 trains on 35223, a 2% difference.

This belongs in the paper (Section 4.6) rather than in a comment alone: a
reviewer who notices that W is tuned in an arm whose decomposition ignores it
should find the answer already written down.

The pilot found the causal-arm effect modest but not negligible (at H=8,
W=512 beat W=256 by 0.81 points, paired 95% interval [+0.08, +1.54]), which is
why it is searched rather than fixed by fiat.

Because the feature cache is built per W, the search only offers the W values
that have actually been precomputed — asking for a missing one would fail
halfway through a trial.

Surrogate fidelity
------------------
Trials run for OPTUNA_EPOCHS rather than the full budget. That assumption is
not free: a reviewer of the companion paper asked for evidence that the short
run preserves the ranking. `--rank-check` re-runs the top trials at full length
and reports the Spearman correlation, which belongs in the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import optuna
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atomicio import write_json
from config import (ALL_ARMS, ARMS, CACHE_DIR, DECOMPOSITION_METHODS,
                    DEVICE_OPTUNA, LEDGER_DB, PARTITION_ARM,
                    LOOK_BACK_RANGE, MODELS, OPTUNA_EPOCHS, OPTUNA_PATIENCE,
                    OPTUNA_REPORT_EVERY, OPTUNA_SELECTION_WINDOW, OPTUNA_TRIALS,
                    OPTUNA_TRAIN_ORIGINS, OPTUNA_VAL_ORIGINS,
                    PARAMS_DIR, STATE_DIR, W_CHOICES,
                    MAX_HORIZON, NUM_STATIONS, SEED)
from dataset import CausalDataModule
from models import build_model, model_param_space
from pipeline import Stage
from trainer import Trainer

optuna.logging.set_verbosity(optuna.logging.WARNING)



def _storage_urls() -> list:
    """
    Every optuna storage file in STATE_DIR.

    The storage is now split per scope (see OptunaStage._storage_tag), so the
    progress display can no longer assume a single optuna.db — it would report
    an empty search while the trials were landing in optuna_causal_1da991fa.db.
    """
    import glob as _glob
    out = []
    for path in sorted(_glob.glob(os.path.join(STATE_DIR, "optuna*.db"))):
        out.append(f"sqlite:///{path}")
    return out


def _trial_states(storage: str, study_name: str) -> dict:
    """Trial counts by state for one study. Read-only, safe while others write."""
    study = optuna.load_study(study_name=study_name, storage=storage)
    out: dict = {}
    for t in study.get_trials(deepcopy=False):
        out[t.state] = out.get(t.state, 0) + 1
    return out


class OptunaStage(Stage):
    """Bayesian search over model and window hyperparameters."""

    name = "optuna"
    unit_name = "study"          # one (decomposition, model) search
    description = "hyperparameter search, one study per (decomposition, model)"

    # Several workers may share one study. Optuna's storage assigns trials, so
    # putting 52 processes on 42 studies keeps every core busy instead of
    # leaving ten idle once the units run out.
    allow_concurrent = True

    # Every worker sharing a study counts as a claim, so the default ceiling of
    # three would idle 49 of 52 workers. Optuna's own storage decides when a
    # study is finished; the ledger only records that it was worked on.
    max_attempts = 10_000

    # Target for this invocation; set from --n-trials in __init__.
    TARGET_TRIALS = OPTUNA_TRIALS

    def __init__(self, methods, models, stations, arm="causal",
                 n_trials=OPTUNA_TRIALS, ledger_db=LEDGER_DB, **kw):
        super().__init__(ledger_db, **kw)
        self.arm = arm
        self.methods = list(methods)
        self.models = list(models)
        self.stations = list(stations)
        self.n_trials = int(n_trials)
        # The roster is printed by a classmethod in the parent process, which
        # has no instance to ask. Without this it fell back to the config
        # default and reported "4/50" for a run launched with --n-trials 5.
        OptunaStage.TARGET_TRIALS = self.n_trials
        self._home = None      # study_name -> (db path, n complete)
        self.storage = self._make_storage(self._storage_tag())

    def _storage_tag(self) -> str:
        """
        A storage file per SCOPE, not per project.

        One shared optuna.db does not survive several hundred workers. Every
        epoch each worker calls should_prune(), and MedianPruner answers it by
        reading every intermediate value of every trial in the study, so the
        read grows with the study and happens once per epoch per worker.
        Measured on Lustre: 40 workers gave zero lock errors, 160 gave ~400,
        576 gave ~4000 and CPU efficiency fell to 0.4% — the workers were
        queueing for SQLite, not training.

        Splitting the file by scope removes the contention, but only if the
        scopes are disjoint: two jobs sharing a study must share its trial
        history, or each would build its own 50 trials and the budget would
        double. So partition the work by --methods/--models and let each
        partition own its storage.

        The tag is derived from the scope rather than the SLURM job id on
        purpose — resubmitting the same partition must reuse the same file, or
        a job that hits the wall would lose every trial it had completed.
        """
        scope = (self.arm, tuple(sorted(self.methods)), tuple(sorted(self.models)))
        if (set(self.methods) == set(DECOMPOSITION_METHODS)
                and set(self.models) == set(MODELS)):
            return ""                      # whole arm: keep the historic name
        digest = hashlib.sha1(repr(scope).encode()).hexdigest()[:8]
        return f"_{self.arm}_{digest}"

    @staticmethod
    def _tag_of_path(path: str) -> str:
        """`.../optuna_leaky_ab12cd34.db` -> `_leaky_ab12cd34`."""
        base = os.path.basename(path)
        return base[len("optuna"):-len(".db")]

    def _storage_for(self, study_name: str):
        """
        The storage that already holds this study, or this scope's own.

        The tag is a hash of (arm, methods, models), which makes resubmitting
        the same partition reuse its file — but it also means a *narrower*
        resubmission is a different partition and gets an empty one. That
        happened: `--methods dwt` had 39 completed trials for dwt/bilstm, and
        resubmitting the single stuck study as `--methods dwt --models bilstm`
        opened a fresh database and restarted the search from zero. Nothing
        was lost, but the work was about to be done twice and the status
        display showed whichever file the glob happened to read last.

        Looking the study up by name removes the coupling entirely: wherever
        it lives, that is where the next worker writes. When the same name
        somehow exists in more than one file, the one with the most completed
        trials wins — an empty duplicate must never displace real history.
        """
        if self._home is None:
            self._home = {}
            for url in _storage_urls():
                path = url[len("sqlite:///"):]
                try:
                    for sm in optuna.get_all_study_summaries(storage=url):
                        n = _trial_states(url, sm.study_name).get(
                            optuna.trial.TrialState.COMPLETE, 0)
                        prev = self._home.get(sm.study_name)
                        if prev is None or n > prev[1]:
                            self._home[sm.study_name] = (path, n)
                except Exception:                             # noqa: BLE001
                    continue
        hit = self._home.get(study_name)
        if hit is None:
            return self.storage
        path, n = hit
        tag = self._tag_of_path(path)
        if tag != self._storage_tag():
            self.log(f"  {study_name}: mevcut {n} tamamlanmis deneme "
                     f"{os.path.basename(path)} icinde — bu kapsamin dosyasi "
                     f"yerine o kullanilacak")
        return self._make_storage(tag)

    @staticmethod
    def _make_storage(tag: str = ""):
        """
        Optuna's trial storage, tuned for many concurrent workers.

        The ledger and this file are two different databases with two very
        different access patterns. A worker touches the ledger twice per unit;
        it touches this file on every trial — create, report, complete. With a
        few hundred workers spread over several jobs that is constant write
        contention, and SQLite's default busy timeout of five seconds is far
        too short: the loser of a race raises "database is locked" and kills
        the trial instead of waiting a moment. Our own ledger waits three
        minutes (jobstate.open_sqlite); ask Optuna for the same.

        Everything here is best-effort. A storage-tuning detail must never be
        the reason a search fails to start, so each step falls back.
        """
        path = os.path.join(STATE_DIR, f"optuna{tag}.db")
        url = f"sqlite:///{path}"

        # WAL is a persistent property of the file, so setting it once with
        # our own connection is enough — Optuna's connections inherit it.
        try:
            from jobstate import open_sqlite
            open_sqlite(path).close()
        except Exception:                                     # noqa: BLE001
            pass

        try:
            return optuna.storages.RDBStorage(
                url=url,
                engine_kwargs={"connect_args": {"timeout": 180.0}},
            )
        except Exception as exc:                              # noqa: BLE001
            print(f"[optuna] tuned storage unavailable ({type(exc).__name__}: "
                  f"{exc}); falling back to the plain URL. Expect "
                  f"'database is locked' if you run many jobs at once.")
            return url

    # ------------------------------------------------------------ Stage API
    def plan(self):
        return {f"{self.arm}|{d}|{m}": dict(arm=self.arm, decomposition=d,
                                            model=m)
                for d in self.methods for m in self.models}

    def preflight(self):
        """Only offer window sizes whose cache actually exists."""
        self.available_W = {}
        for d in self.methods:
            ws = [w for w in W_CHOICES if self._cache_ready(d, w)]
            if not ws:
                if self.arm == PARTITION_ARM and d == "none":
                    # 'none' is deliberately never built in the partition arm:
                    # with the identity decomposition a block and a window give
                    # the same numbers, so the arm cannot differ from causal.
                    # tests/test_causality.py asserts it for every origin,
                    # which is stronger than a run would be. Telling the user
                    # to "run precompute first" would send them off to build a
                    # cache we chose not to build.
                    raise RuntimeError(
                        "the partition arm has no 'none' cache, on purpose: "
                        "with the identity decomposition it is bit-identical "
                        "to the causal arm (see "
                        "test_partition_none_is_identical_to_causal).\n"
                        "Name the methods explicitly, e.g.\n"
                        "  python run.py optuna --arm partition "
                        "--methods dwt vmd emd eemd ceemdan")
                raise RuntimeError(
                    f"no {self.arm} cache for '{d}'. Run "
                    f"`python run.py precompute --arms {self.arm}` first.")
            self.available_W[d] = ws
            self.log(f"  {d:8s}: W options {ws}")

    def _cache_ready(self, method: str, W: int) -> bool:
        return all(os.path.exists(os.path.join(
            CACHE_DIR, self.arm, f"W{W}", method, f"st{s}", "meta.json"))
            for s in self.stations)

    def execute(self, key, spec):
        d, m = spec["decomposition"], spec["model"]
        study_name = f"{self.arm}__{d}__{m}"
        storage = self._storage_for(study_name)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,          # resume an interrupted search
            sampler=optuna.samplers.TPESampler(seed=self._sampler_seed()),
            # n_warmup_steps must be expressed in the same units as the
            # reported step, and reporting now happens every
            # OPTUNA_REPORT_EVERY epochs. With warmup=3 and the first report
            # at epoch 4 the warmup window was already over, so a trial could
            # be cut at its very first checkpoint: 191 of 269 prunes landed on
            # epoch 4, ten in a row in one study, and the stall detector then
            # failed the whole study. Skipping the first two checkpoints gives
            # a trial a third of its budget before it can be judged.
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2 * OPTUNA_REPORT_EVERY),
        )

        # Several workers share this study, so the target must be re-checked
        # against the storage rather than computed once. Running
        # `n_trials=remaining` in every worker would multiply the budget by the
        # number of workers; instead each takes a small batch and looks again.
        self._optimize_until(study, d, m)

        # Not study.best_params: that ranks every completed trial, including
        # the ones that overshot the target while workers drained (see
        # budget_trials). Every study must be judged on the same number of
        # draws or the comparison rewards whichever search happened to get
        # extra workers at the end.
        winner = self.best_within_budget(study, self.n_trials)
        if winner is None:
            raise RuntimeError(
                f"{d}/{m}: no completed trial to select from "
                f"({self._last_failure(study)})")
        pool = len(self.budget_trials(study, self.n_trials))
        overall = self._best_value(study)
        if overall is not None and winner.value > overall:
            self.log(f"  {d}/{m}: ilk {pool} denemeye gore secildi "
                     f"({winner.value:.5f}); butce disinda daha iyisi vardi "
                     f"({overall:.5f}) ve kullanilmadi")

        best = dict(winner.params)
        self._warn_on_boundary(d, m, best)
        self._write_best(self.arm, d, m, best, winner.value)
        return {"best_params": best, "best_value": float(winner.value),
                "budget": pool, "trial": winner.number,
                "n_trials": len(study.trials)}

    # ---------------------------------------------------------------- seeding
    def _sampler_seed(self) -> int:
        """
        A sampler seed that differs per worker.

        With `allow_concurrent`, two workers can end up inside one study. Given
        the same seed and the same trial history they are deterministic in the
        same way, so they propose the *same point* and one of the two trials is
        wasted. Optuna warns about exactly this for distributed runs, and with
        52 workers over 84 studies the overlap is not hypothetical — it starts
        as soon as the first worker finishes and re-claims.

        Offsetting by the worker index keeps each worker reproducible on its own
        while making them explore different points. Note what this does and does
        not buy: the *search* is only reproducible for a given worker count,
        which is inherent to asynchronous distributed optimisation. What the
        paper needs is reproducibility of the *result*, and that is intact —
        the selected hyperparameters are recorded, and final training is seeded
        separately and deterministically.
        """
        idx = os.environ.get("_STAGE_WORKER_ID")
        try:
            return SEED + int(idx) if idx is not None else SEED
        except ValueError:
            return SEED

    # ------------------------------------------------------------ scheduling
    BATCH = 2           # trials claimed per look at the storage
    STALL_LIMIT = 5     # consecutive batches allowed to complete nothing

    def _completed(self, study) -> int:
        return sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE)

    @staticmethod
    def budget_trials(study, n_trials: int) -> list:
        """
        The first `n_trials` COMPLETE trials, in the order they were created.

        Studies do not stop at exactly the target. Several workers share one
        study, and once the others finish they all pile onto whatever is left:
        `causal|none|bilstm` was seen with 26 trials in flight at the same
        time. The count crosses the target while those trials are still
        training, and each of them lands as COMPLETE afterwards. The result was
        a search budget that ranged from 50 to 93 trials depending on how many
        workers happened to be free — and a study given 93 draws has a better
        expected minimum than one given 50, so the comparison between
        decompositions was no longer at equal cost.

        Truncating to the first N restores that, using only trials that were
        already recorded. `trial.number` is assigned when a trial is created,
        so the order is fixed in the storage and the selection is reproducible
        by anyone who opens the database.
        """
        done = [t for t in study.get_trials(deepcopy=False)
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None]
        done.sort(key=lambda t: t.number)
        return done[:n_trials]

    @classmethod
    def best_within_budget(cls, study, n_trials: int):
        """The best trial of the first `n_trials` completed, or None."""
        pool = cls.budget_trials(study, n_trials)
        return min(pool, key=lambda t: t.value) if pool else None

    @staticmethod
    def _best_value(study):
        """Best completed value so far, or None if nothing has finished."""
        vals = [t.value for t in study.get_trials(deepcopy=False)
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None]
        return min(vals) if vals else None

    @staticmethod
    def _last_failure(study) -> str:
        """The most recent failed trial's parameters, for the error message."""
        failed = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.FAIL]
        if not failed:
            return "no failed trials recorded; all were pruned"
        t = failed[-1]
        return f"trial {t.number} with params {t.params}"

    def _stop_when_done(self, study, trial) -> None:
        """
        Stop the study the moment it reaches its target.

        Without this a worker finishes its batch regardless. In the pilot that
        cost twenty minutes: one worker took `none/gru` after the target was
        already met and ran a full extra trial, ending at "6/5 trials
        complete". Harmless at two workers; with 52 sharing 84 studies the
        overshoot lands on every study that finishes while others are still
        running, and it is all wasted core-hours.
        """
        if self._completed(study) >= self.n_trials:
            study.stop()

    def _optimize_until(self, study, d: str, m: str) -> None:
        """
        Run trials in small batches until the study reaches its target.

        This is the distributed pattern Optuna recommends: every worker talks
        to the same storage, takes a couple of trials, and re-reads the count.
        The alternative — one `study.optimize(n_trials=remaining)` per worker —
        would run `workers x remaining` trials in total.

        The previous study instead used `study.optimize(n_jobs=10)`, which is
        thread-based: ten trials share one interpreter and contend for the GIL
        during the Python-heavy parts of the data pipeline, and the option is
        deprecated in current Optuna. Separate processes avoid both problems.
        """
        objective = self._objective(d, m)
        stalled = 0
        while True:
            done = self._completed(study)
            if done >= self.n_trials:
                self.log(f"[{self.name}] {d}/{m}: {done}/{self.n_trials} "
                         f"trials complete")
                return
            self.log(f"[{self.name}] {d}/{m}: {done}/{self.n_trials}, "
                     f"taking {self.BATCH}")
            study.optimize(objective, n_trials=self.BATCH,
                           gc_after_trial=True, catch=(RuntimeError,),
                           callbacks=[self._stop_when_done])

            # Only COMPLETE trials count towards the target, so a fault that
            # kills every trial leaves the count at zero and the loop spinning
            # forever. That is what a bad device object did here: three minutes
            # of a worker reporting progress while finishing nothing.
            #
            # Pruned trials are a legitimate outcome and would also fail to
            # advance the count, hence a tolerance rather than a single strike.
            if self._completed(study) > done:
                stalled = 0
                continue

            # Pruning is success, not failure: it means the pruner is doing
            # its job. Treating a run of prunes as a stall killed studies that
            # were working — emd/lstm lost all seven of its models that way.
            # Only a study that has never completed anything is genuinely
            # stuck; once even one trial has finished, the search is viable
            # and a long pruned streak is information, not a fault.
            if self._completed(study) > 0:
                stalled = 0
                continue

            stalled += 1
            if stalled >= self.STALL_LIMIT:
                last = self._last_failure(study)
                raise RuntimeError(
                    f"{d}/{m}: {self.STALL_LIMIT * self.BATCH} trials and not "
                    f"one completed. Last failure: {last}")

    # --------------------------------------------------------------- progress
    @classmethod
    def unit_detail(cls, key: str) -> str:
        """`arm|decomposition|model` -> that study's trial counts."""
        try:
            arm, d, m = key.split("|")
            states = {}
            for storage in _storage_urls():
                try:
                    states = _trial_states(storage, f"{arm}__{d}__{m}")
                    break                 # the study lives in exactly one file
                except Exception:         # noqa: BLE001  not in this one
                    continue
            done = states.get(optuna.trial.TrialState.COMPLETE, 0)
            pruned = states.get(optuna.trial.TrialState.PRUNED, 0)
            failed = states.get(optuna.trial.TrialState.FAIL, 0)
            out = f"trials {done}/{cls.TARGET_TRIALS} complete"
            if pruned:
                out += f", {pruned} pruned"
            if failed:
                out += f", {failed} failed"
            return out
        except Exception:                                     # noqa: BLE001
            return ""

    @classmethod
    def extra_progress(cls) -> str:
        """
        Trials completed across every study in the storage.

        Without this the monitor reports units, and a unit here is a whole
        study — so the display sat unchanged for half an hour while trials were
        running underneath it.
        """
        try:
            done = running = failed = 0
            pairs = [(u, s) for u in _storage_urls()
                     for s in optuna.get_all_study_summaries(storage=u)]
            for storage, s in pairs:
                for state, count in (s.n_trials and
                                     _trial_states(storage, s.study_name)
                                     or {}).items():
                    if state == optuna.trial.TrialState.COMPLETE:
                        done += count
                    elif state == optuna.trial.TrialState.RUNNING:
                        running += count
                    elif state == optuna.trial.TrialState.FAIL:
                        failed += count
            return f"trials: {done} done, {running} running, {failed} failed"
        except Exception:                                     # noqa: BLE001
            return ""

    # ----------------------------------------------------------------- status
    def status(self, show_failures: bool = False) -> None:
        """
        One line per study: how far it got, and what it found.

        The inherited version prints a single "12/35 done" counter, which
        answers how much is left but not which searches are finished, which
        are close, or whether any of them is producing a usable score. That
        had to be read out of the SLURM logs instead, and a log only shows the
        studies its own job touched.

        Read-only: it opens the Optuna files and the ledger and writes to
        neither, so it is safe on the login node while jobs are running.
        """
        plan = self.plan()
        self.ledger.register_many(plan.keys())
        states = dict(self.ledger.states()) if hasattr(self.ledger, "states") \
            else {}

        # study_name -> storage url. When a name appears in several files
        # (see _storage_for), the richest copy wins; taking the last one the
        # glob returned would have reported 9 completed trials for a study
        # that had 39 in another file.
        found, dupes = {}, {}
        for url in _storage_urls():
            try:
                summaries = optuna.get_all_study_summaries(storage=url)
            except Exception:                                 # noqa: BLE001
                continue
            for sm in summaries:
                n = _trial_states(url, sm.study_name).get(
                    optuna.trial.TrialState.COMPLETE, 0)
                dupes.setdefault(sm.study_name, []).append(
                    (os.path.basename(url), n))
                if sm.study_name not in found or n > found[sm.study_name][1]:
                    found[sm.study_name] = (url, n)
        found = {k: v[0] for k, v in found.items()}
        for name, where in sorted(dupes.items()):
            if len(where) > 1:
                print(f"  UYARI: {name} birden fazla dosyada var — "
                      + ", ".join(f"{f}({n})" for f, n in where)
                      + "; en cok denemesi olan kullaniliyor")

        print(f"\n{'calisma':34s} {'tamam':>6s} {'budanan':>8s} "
              f"{'hatali':>7s} {'kosan':>6s}  {'en iyi':>9s}  durum")
        print("-" * 92)
        done_n = 0
        for key in sorted(plan):
            arm, d, m = key.split("|")
            name = f"{arm}__{d}__{m}"
            url = found.get(name)
            if url is None:
                print(f"{key:34s} {'-':>6s} {'-':>8s} {'-':>7s} {'-':>6s}  "
                      f"{'-':>9s}  baslamamis")
                continue
            try:
                study = optuna.load_study(study_name=name, storage=url)
                counts: dict = {}
                for t in study.get_trials(deepcopy=False):
                    counts[t.state] = counts.get(t.state, 0) + 1
                c = counts.get(optuna.trial.TrialState.COMPLETE, 0)
                p = counts.get(optuna.trial.TrialState.PRUNED, 0)
                f = counts.get(optuna.trial.TrialState.FAIL, 0)
                r = counts.get(optuna.trial.TrialState.RUNNING, 0)
                best = f"{study.best_value:.5f}" if c else "-"
            except Exception as exc:                          # noqa: BLE001
                print(f"{key:34s}  okunamadi: {type(exc).__name__}")
                continue

            if c >= self.n_trials:
                note, ok = "BITTI", True
            elif states.get(key) == "DONE":
                note, ok = "defterde DONE", True
            else:
                note, ok = f"{self.n_trials - c} deneme kaldi", False
            done_n += ok
            print(f"{key:34s} {c:6d} {p:8d} {f:7d} {r:6d}  {best:>9s}  {note}")

        print("-" * 92)
        print(f"{done_n}/{len(plan)} calisma tamam")
        if show_failures:
            for key, attempts, err in self.ledger.failures():
                print(f"  FAILED {key} (attempts={attempts}): {err}")

    # ------------------------------------------------------------------ reset
    def reset(self, key=None):
        """
        Clear the ledger *and* the Optuna storage.

        These are two separate memories and resetting only the first leaves the
        second holding every trial from the run being abandoned. After the
        device fault above, the studies carried 107 failed trials each; a fresh
        ledger would have resumed straight into them and the search would have
        been seeded by a broken run.
        """
        super().reset(key)

        targets = ([key] if key and key != "__ALL__"
                   else [f"{self.arm}|{d}|{m}"
                         for d in self.methods for m in self.models])
        removed = 0
        for t in targets:
            arm, d, m = t.split("|")
            try:
                optuna.delete_study(study_name=f"{arm}__{d}__{m}",
                                    storage=self.storage)
                removed += 1
            except KeyError:
                pass                      # never created; nothing to remove
        print(f"[{self.name}] removed {removed} Optuna study/studies")

    # ------------------------------------------------------------- objective
    def _objective(self, decomposition: str, model_name: str):
        space = model_param_space(model_name)
        w_options = self.available_W[decomposition]

        def objective(trial: optuna.Trial) -> float:
            # Refuse before doing any work if the study is already finished.
            #
            # The stop-callback only runs *between* trials, so it can prevent a
            # worker starting another one but not stop a trial already in
            # flight. With several workers inside one study the target can
            # therefore be passed by up to (workers - 1) trials: the pilot
            # ended at 6/5 with two workers, and 52 workers would waste far
            # more. Checking here costs a database read instead of a trial.
            if self._completed(trial.study) >= self.n_trials:
                raise optuna.TrialPruned()

            params = self._suggest(trial, space)
            W = trial.suggest_categorical("W", w_options)
            look_back = trial.suggest_int("look_back", *LOOK_BACK_RANGE)
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            batch = trial.suggest_categorical("batch_size", [32, 64, 128])

            # look_back can never exceed the window the features came from.
            look_back = min(look_back, W)

            t0 = time.time()
            # The study name goes on every line. With 52 workers interleaved in
            # one log, "trial 0" alone is ambiguous — there are 84 studies and
            # each has a trial 0.
            # Position against the target, not just Optuna's trial number.
            # The number counts everything ever attempted in this study,
            # including pruned and failed trials; only COMPLETE ones count
            # towards n_trials, so the two diverge and the number alone does
            # not answer "how much is left".
            place = self._completed(trial.study) + 1
            tag = (f"{decomposition}/{model_name} "
                   f"trial {place}/{self.n_trials} (id {trial.number})")
            self.log(f"  {tag}: W={W} lb={look_back} bs={batch} "
                     f"lr={lr:.2e} — loading data")

            dm = CausalDataModule(
                decomposition, W, look_back, self.stations,
                arm=self.arm, horizon=MAX_HORIZON, batch_size=batch,
                # Search on a subsample; the final models see everything.
                subsample={"train": OPTUNA_TRAIN_ORIGINS,
                           "val": OPTUNA_VAL_ORIGINS},
            ).setup()
            t_load = time.time() - t0
            self.log(f"  {tag}: {len(dm.loader('train'))} train batches "
                     f"({len(dm.loader('val'))} val), K={dm.K} channels, "
                     f"data ready in {t_load:.0f}s")

            model = build_model(model_name, dm.K, MAX_HORIZON, params)
            trainer = Trainer(model, DEVICE_OPTUNA, checkpoint_path=None,
                              seed=SEED, verbose=False,
                              selection_window=OPTUNA_SELECTION_WINDOW)

            best_so_far = self._best_value(trial.study)

            def report(epoch, tr, va):
                # An epoch line per trial. Without it a slow trial and a hung
                # one look identical from outside. The study best is repeated
                # so a trial that is clearly going nowhere is visible as such
                # without cross-referencing earlier lines.
                mark = ""
                if best_so_far is not None:
                    mark = (f"  study best {best_so_far:.5f}"
                            + ("  <-- beating it" if va < best_so_far else ""))
                self.log(f"    {tag}  epoch {epoch:3d}/{OPTUNA_EPOCHS}  "
                         f"train {tr:.5f}  val {va:.5f}"
                         f"  [{time.time() - t0:.0f}s]{mark}")
                # Only touch the storage every OPTUNA_REPORT_EVERY epochs.
                # should_prune() reads the whole study, so at several hundred
                # workers a per-epoch check turns the shared SQLite file into
                # the bottleneck. See config.OPTUNA_REPORT_EVERY.
                if epoch % OPTUNA_REPORT_EVERY and epoch != OPTUNA_EPOCHS:
                    return False

                # Abandon a trial whose study no longer needs it.
                #
                # The guard at the top of objective() stops a worker starting
                # a trial after the target is met, and the stop-callback runs
                # between trials — neither can touch a trial already in
                # flight. So when the last study of a job finished, the
                # workers inside a trial kept training for up to an hour and a
                # half against a study that was already complete. The job sat
                # at 6 live workers out of 56 and TRUBA flagged it at 12%
                # efficiency, with a warning that the core allocation would be
                # cut if it continued.
                #
                # This is checked on the report grid, where the storage is
                # being read anyway, so it costs nothing extra and bounds the
                # waste at OPTUNA_REPORT_EVERY epochs instead of a full trial.
                if self._completed(trial.study) >= self.n_trials:
                    self.log(f"  {tag}: study already at {self.n_trials} "
                             f"complete — bu deneme birakiliyor")
                    return True

                trial.report(va, epoch)
                if trial.should_prune():
                    self.log(f"  {tag}: PRUNED at epoch {epoch} "
                             f"(val {va:.5f} above the running median)")
                    return True
                return False

            result = trainer.fit(dm.loader("train"), dm.loader("val"),
                                 epochs=OPTUNA_EPOCHS,
                                 patience=OPTUNA_PATIENCE,
                                 lr=lr, on_epoch=report)
            self.log(f"  {tag}: {result['stopped']}, "
                     f"score {result['selection_val']:.5f} "
                     f"(raw best {result['best_val']:.5f} at epoch "
                     f"{result['best_epoch']}), "
                     f"{(time.time() - t0) / 60:.1f} min")
            if result["stopped"] == "pruned":
                raise optuna.TrialPruned()

            # A configuration that diverged has no finite score. Returning NaN
            # makes Optuna raise "The value nan is not acceptable" and record
            # the trial as FAILED — and TPE excludes failed trials from its
            # model, so the search learns nothing and may propose the same
            # region again. Pruned trials ARE modelled, so reporting it that
            # way keeps the information and costs the same compute.
            if not math.isfinite(result["selection_val"]):
                self.warn(f"  {tag}: diverged (non-finite loss); "
                          f"recorded as pruned so the sampler avoids this "
                          f"region instead of discarding it")
                raise optuna.TrialPruned()

            # The smoothed criterion, not the raw minimum. See
            # trainer.smoothed_validation_minimum for why.
            return float(result["selection_val"])

        return objective

    @staticmethod
    def _suggest(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a declared param_space into Optuna suggestions."""
        out = {}
        for key, spec in space.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                lo, hi = spec
                if isinstance(lo, int) and isinstance(hi, int):
                    out[key] = trial.suggest_int(key, lo, hi)
                else:
                    out[key] = trial.suggest_float(key, float(lo), float(hi))
            elif isinstance(spec, (list, tuple)):
                out[key] = trial.suggest_categorical(key, list(spec))
        return out

    # -------------------------------------------------------------- boundary
    def _warn_on_boundary(self, decomposition: str, model: str,
                          best: dict) -> None:
        """
        Warn when a selected value sits on the edge of its search range.

        A bound that binds is not a choice, it is an artefact: the search
        wanted to go further and was not allowed to. The previous study was
        criticised for exactly this — VMD's K came back at 8, which was the top
        of its grid — so the decomposition tuner already checks it and this
        stage did not, which left the model hyperparameters unchecked.

        Categorical parameters are excluded: picking the first or last item of
        an unordered list means nothing.
        """
        space = dict(model_param_space(model))
        space["look_back"] = LOOK_BACK_RANGE

        edges = []
        for key, spec in space.items():
            if key not in best or not (isinstance(spec, tuple)
                                       and len(spec) == 2):
                continue
            lo, hi = spec
            value = best[key]
            tol = (hi - lo) * 0.02
            if value <= lo + tol:
                edges.append(f"{key}={value} at the lower bound {lo}")
            elif value >= hi - tol:
                edges.append(f"{key}={value} at the upper bound {hi}")

        if edges:
            self.warn(f"{decomposition}/{model}: best parameters sit on the "
                      f"search boundary — " + "; ".join(edges)
                      + ". Widen the range in models.py param_space(), or "
                        "justify the limit in the paper.")

    # ---------------------------------------------------------------- output
    @staticmethod
    def _write_best(arm: str, decomposition: str, model: str,
                    params: dict, value: float):
        out_dir = os.path.join(PARAMS_DIR, "models", arm)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{decomposition}__{model}.json")
        # Several workers share one study and each writes this on the way out.
        # The content is whatever the storage says is best, so a private temp
        # name is all that is needed to keep the commits from colliding.
        write_json(path, {"arm": arm, "decomposition": decomposition,
                          "model": model, "best_params": params,
                          "best_value": float(value)}, indent=2)

    @staticmethod
    def load_best(arm: str, decomposition: str, model: str) -> dict:
        """
        Each arm is tuned on its own validation split. Sharing one set of
        hyperparameters across arms would hand the leaky arm settings chosen
        under a different feature distribution, and any difference in results
        would then be partly a tuning artefact.
        """
        path = os.path.join(PARAMS_DIR, "models", arm,
                            f"{decomposition}__{model}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no tuned parameters for {arm}/{decomposition}/{model}. "
                f"Run `python run.py optuna --arm {arm}` first.")
        with open(path) as f:
            return json.load(f)["best_params"]


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=OptunaStage.description)
    ap.add_argument("--methods", nargs="+", default=DECOMPOSITION_METHODS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--stations", type=int, nargs="+",
                    default=list(range(NUM_STATIONS)))
    ap.add_argument("--arm", default="causal", choices=ALL_ARMS)
    ap.add_argument("--n-trials", type=int, default=OPTUNA_TRIALS)
    Stage.add_common_arguments(ap)
    args = ap.parse_args()

    stage = OptunaStage(args.methods, args.models, args.stations,
                        arm=args.arm, n_trials=args.n_trials)
    stage.main_from_args(args)


if __name__ == "__main__":
    main()
