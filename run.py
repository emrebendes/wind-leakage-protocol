#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project entry point. Run everything from here.
==============================================

    cd <repository root>
    python run.py <command> [options]

Commands
--------
    check        causality suite — run this before anything else
    tune         decomposition hyperparameters, per W, training windows only
    precompute   causal feature cache
    optuna       model hyperparameter search (W is searched here)
    train        final training with replicate seeds
    deploy       deployment gap: leaky-trained models on causal inputs
    analyze      tables and figures for the paper
    status       progress of every stage, in one table
    results      what has been produced so far
    doctor       environment and prerequisite check

Every long command accepts the same options:

    --status --failures        progress and failed units
    --reset [KEY]              clear one unit, or the whole stage
    --max-seconds N            stop claiming new work after N seconds

Examples
--------
    python run.py check
    python run.py tune
    python run.py precompute --arms causal leaky
    python run.py optuna --arm leaky

  the third regime (split first, then decompose) is always asked for by
  name, so a parameterless run keeps producing exactly the two arms the
  existing results were built from:

    python run.py precompute --arms partition --methods dwt vmd
    python run.py optuna     --arm  partition --methods dwt vmd
    python run.py train      --arm  partition --methods dwt vmd
    python run.py deploy     --source-arm partition
    python run.py status
    python run.py train --status --failures
    python run.py train --reset "dwt|bilstm|seed0"

Parallelism
-----------
Every long command takes --workers N. It defaults to the SLURM allocation
(SLURM_CPUS_PER_TASK) or the local CPU count, so a 52-core hamsi node runs 52
worker processes that share the job ledger. Use --workers 1 to stay in one
process while debugging.

    python run.py precompute --workers 52
    sbatch slurm/hamsi.sh precompute
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


# =============================================================================
class Command:
    """One runnable pipeline command."""

    def __init__(self, key: str, module: str, summary: str,
                 stage_name: str | None = None):
        self.key = key
        self.module = module
        self.summary = summary
        self.stage_name = stage_name

    def run(self, argv: list) -> int:
        script = os.path.join(SRC, f"{self.module}.py")
        return subprocess.call([sys.executable, script] + argv, cwd=ROOT)

    def __repr__(self) -> str:
        return f"Command({self.key})"


COMMANDS = [
    Command("tune", "tune_decompositions",
            "decomposition hyperparameters, per W", "tune_decomp"),
    Command("precompute", "precompute_causal",
            "feature cache, per arm", "precompute"),
    Command("optuna", "optimize",
            "model hyperparameter search (per arm)", "optuna"),
    Command("train", "train_final",
            "final training, replicate seeds (per arm)", "train_final"),
    Command("deploy", "deployment_gap",
            "leakage-trained models on causal inputs", "deployment_gap"),
    Command("analyze", "analyze",
            "tables and figures for the paper", None),
]
BY_KEY = {c.key: c for c in COMMANDS}


# =============================================================================
class Dashboard:
    """Single view over every stage and the result store."""

    def __init__(self):
        from config import LEDGER_DB, RESULTS_DIR, CACHE_DIR
        self.ledger_db = LEDGER_DB
        self.results_dir = RESULTS_DIR
        self.cache_dir = CACHE_DIR

    def status(self) -> None:
        from jobstate import JobLedger

        print(f"\nproject: {ROOT}")
        print(f"ledger : {self.ledger_db}\n")
        header = f"{'stage':16s}{'done':>8s}{'run':>6s}{'pend':>7s}{'fail':>6s}   state"
        print(header)
        print("-" * len(header))

        if not os.path.exists(self.ledger_db):
            print("  (no stage has been started yet)")
            return

        for cmd in COMMANDS:
            if not cmd.stage_name:
                continue
            led = JobLedger(self.ledger_db, cmd.stage_name)
            c = led.counts()
            if c["TOTAL"] == 0:
                state = "not started"
            elif c["DONE"] == c["TOTAL"]:
                state = "complete"
            elif c["FAILED"]:
                state = f"{c['FAILED']} failed — see --failures"
            else:
                state = "in progress"
            print(f"{cmd.stage_name:16s}{c['DONE']:>8d}{c['RUNNING']:>6d}"
                  f"{c['PENDING']:>7d}{c['FAILED']:>6d}   {state}")
        print()

    def results(self) -> None:
        from results import ResultStore

        if not os.path.exists(self.results_dir):
            print("no results yet")
            return
        store = ResultStore(self.results_dir)
        summary = store.summary()
        print(f"\nresults: {self.results_dir}")
        if not summary:
            print("  (empty)")
            return
        for arm, n in sorted(summary.items()):
            print(f"  {arm:20s} {n:4d} runs")

        problems = store.verify()
        print(f"\nintegrity: {len(problems)} problem(s)")
        for p in problems[:10]:
            print("   ", p)

    def cache(self) -> None:
        if not os.path.exists(self.cache_dir):
            print("no cache yet")
            return
        # Layout is causal_cache/{arm}/W{W}/{method}/st{n}/. This walked only
        # two levels, from before the arm was added to the path, so it read
        # "causal" as a window and "W1024" as a method, looked for meta.json
        # one level too high, and reported an empty cache for a full one —
        # alarming and wrong at exactly the moment you check whether a copy
        # arrived intact.
        print(f"\ncache: {self.cache_dir}")
        if not os.path.isdir(self.cache_dir):
            print("  (missing)")
            return
        total = 0
        for arm in sorted(os.listdir(self.cache_dir)):
            ad = os.path.join(self.cache_dir, arm)
            if not os.path.isdir(ad):
                continue
            for w in sorted(os.listdir(ad)):
                wd = os.path.join(ad, w)
                if not os.path.isdir(wd):
                    continue
                for method in sorted(os.listdir(wd)):
                    md = os.path.join(wd, method)
                    if not os.path.isdir(md):
                        continue
                    stations = [s for s in sorted(os.listdir(md))
                                if os.path.exists(
                                    os.path.join(md, s, "meta.json"))]
                    chunks = sum(
                        len([f for f in os.listdir(os.path.join(md, s))
                             if f.endswith(".npy")]) for s in stations)
                    total += chunks
                    print(f"  {arm:7s} {w:6s} {method:8s} "
                          f"{len(stations)} stations, {chunks} chunks")
        print(f"  {'':7s} {'':6s} {'TOPLAM':8s} {total} chunks")


# =============================================================================
class Doctor:
    """Checks the environment before a long job wastes an allocation."""

    PACKAGES = ["numpy", "torch", "optuna", "pywt", "vmdpy", "PyEMD", "pandas"]

    def run(self) -> int:
        print(f"\nproject root : {ROOT}")
        print(f"python       : {sys.version.split()[0]}")
        print(f"executable   : {sys.executable}\n")

        problems = 0

        print("packages")
        for p in self.PACKAGES:
            # Print BEFORE importing. On Windows, torch can stall for a long
            # time loading its DLLs, and a silent hang is impossible to
            # diagnose — this way the last line printed names the culprit.
            print(f"  ...     {p:10s}", end="\r", flush=True)
            try:
                mod = __import__(p)
                print(f"  ok      {p:10s} {getattr(mod, '__version__', '')}")
            except ImportError:
                print(f"  MISSING {p:10s}")
                problems += 1
            except KeyboardInterrupt:
                print(f"  SKIPPED {p:10s} (interrupted — import was hanging)")
                problems += 1
            except Exception as exc:            # noqa: BLE001
                # A broken install raises something other than ImportError;
                # torch on Windows is the usual offender.
                print(f"  BROKEN  {p:10s} {type(exc).__name__}: "
                      f"{str(exc)[:60]}")
                problems += 1

        print("\ndata")
        from config import DATA_FILE, SERIES_END
        if os.path.exists(DATA_FILE):
            import numpy as np
            d = np.load(DATA_FILE, allow_pickle=True)
            raw = len(d[0])
            print(f"  ok      {len(d)} stations x {raw} hours on disk")

            # Report the length the pipeline actually uses, not the length of
            # the file. The last 426 hours of every station are gap fill and
            # are cut by config.SERIES_END; a machine whose config lacks that
            # cut computes different split boundaries, so a cache copied from
            # a machine that has it would be read at the wrong offsets. This
            # line used to print the raw length on both, which is exactly the
            # question you ask when checking whether a copy is compatible.
            from causal_features import split_bounds
            eff = SERIES_END or raw
            tr, va = split_bounds(eff)
            print(f"  ok      {eff} hours used"
                  f"{'' if SERIES_END is None else f' (SERIES_END={SERIES_END})'}"
                  f"  splits {tr}/{va}/{eff}")
            if SERIES_END is None:
                print("  WARNING SERIES_END is not set. If this project shares "
                      "a cache with a machine that sets it, the splits differ "
                      "and the cache will be read at the wrong offsets.")
                problems += 1
        else:
            print(f"  MISSING {DATA_FILE}")
            problems += 1

        print("\ngpu")
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                total = p.total_memory / 1e9
                print(f"  ok      {p.name}  {total:.1f} GB")
                # Worst case in the search space: transformer, d_model=256,
                # nhead=8, look_back=168, batch=128. Measured at roughly 2 GB
                # of activations. Below ~4 GB the trainer will be chunking
                # batches often, which works but is slow.
                if total < 4:
                    print("  WARNING small card: expect batch chunking on the "
                          "larger configurations")
            else:
                print("  none    cpu only (fine for tune/precompute/optuna)")
        except Exception as exc:                # noqa: BLE001
            print(f"  unknown torch unavailable ({type(exc).__name__})")

        print("\ncpu")
        import multiprocessing
        n = multiprocessing.cpu_count()
        slurm = os.environ.get("SLURM_CPUS_PER_TASK")
        print(f"  ok      {n} cores"
              + (f", SLURM allocation {slurm}" if slurm else ""))
        print(f"          default workers: "
              f"{max(1, int(slurm) if slurm else n - 1)}")

        print("\ndirectories")
        from config import CACHE_DIR, RESULTS_DIR, STATE_DIR, PARAMS_DIR
        for d in (CACHE_DIR, RESULTS_DIR, STATE_DIR, PARAMS_DIR):
            print(f"  {'ok     ' if os.path.isdir(d) else 'created'} "
                  f"{os.path.relpath(d, ROOT)}")

        print(f"\n{problems} problem(s)\n")
        return 1 if problems else 0


# =============================================================================
def run_checks(argv: list) -> int:
    """
    Causality suite. Nothing else should run until this passes.

    Runs under pytest when it is available and under the standalone runner
    when it is not; the two cover the same checks. pytest is therefore a
    convenience, not a dependency — nothing in the pipeline imports it, so a
    cluster environment without it is fine.

        python run.py check            fast methods
        python run.py check --slow     also the EMD family (minutes per case)
    """
    tests = os.path.join(ROOT, "tests")
    test = os.path.join(tests, "test_causality.py")
    slow = "--slow" in argv

    try:
        import pytest  # noqa: F401
    except ImportError:
        # Say so explicitly: a missing package should never look like a
        # silently different result.
        print("pytest not installed — using the standalone runner "
              "(same checks).\nInstall it for per-test output:\n"
              "    pip install pytest\n", flush=True)
        # Without pytest only the causality file can run itself; test_ledger.py
        # needs fixtures. Say so rather than reporting a clean run.
        print("note: the ledger tests need pytest and will be skipped.\n",
              flush=True)
        return subprocess.call(
            [sys.executable, test] + (["--slow"] if slow else []), cwd=ROOT)

    # One pytest process per file, not one for the directory.
    #
    # vmdpy and torch both drag in an Intel OpenMP runtime, and loading both
    # into one Windows process aborts the interpreter outright — no traceback,
    # no failed test, just "Fatal Python error: Aborted" partway through the
    # run. Nothing in the pipeline ever puts them together (tune and precompute
    # use vmdpy and never import torch; optuna and train import torch and only
    # read the cache), so the conflict was an artefact of collecting every test
    # into one process. Separate processes restore the production arrangement
    # and keep one file's imports out of another's way.
    marker = [] if slow else ["-m", "not slow"]
    files = sorted(f for f in os.listdir(tests)
                   if f.startswith("test_") and f.endswith(".py"))

    failed = []
    for name in files:
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}", flush=True)
        rc = subprocess.call(
            [sys.executable, "-m", "pytest", os.path.join(tests, name), "-v"]
            + marker, cwd=ROOT)
        if rc != 0:
            failed.append(name)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    else:
        print(f"  all {len(files)} test files passed")
    print(f"{'=' * 70}\n")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run.py",
        description="Leak-free wind speed forecasting benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            ["commands:", "  check        causality suite (run first)"]
            + [f"  {c.key:12s} {c.summary}" for c in COMMANDS]
            + ["  status       progress of every stage",
               "  results      what has been produced",
               "  doctor       environment check",
               "",
               "any long command accepts --status --failures --reset "
               "--max-seconds"]),
    )
    ap.add_argument("command", nargs="?", default="status")
    return ap


def main() -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args()
    cmd = args.command

    if cmd in ("-h", "--help", "help"):
        parser.print_help()
        return 0
    if cmd == "check":
        return run_checks(rest)
    if cmd == "doctor":
        return Doctor().run()
    if cmd == "status":
        d = Dashboard()
        for section in (d.status, d.cache, d.results):
            try:
                section()
            except Exception as exc:            # one broken section must not
                print(f"  [{section.__name__}] unavailable: {exc}")
        return 0
    if cmd == "results":
        Dashboard().results()
        return 0
    if cmd in BY_KEY:
        return BY_KEY[cmd].run(rest)

    print(f"unknown command: {cmd}\n")
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
