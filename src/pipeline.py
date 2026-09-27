# -*- coding: utf-8 -*-
"""
Pipeline stage base class.
==========================

Every long-running step of this project has the same shape:

    1. enumerate the units of work
    2. claim one, do it, record it
    3. survive being killed by the scheduler and resume

Writing that loop four times (precompute, decomposition tuning, Optuna, final
training) would mean four chances to get the resume logic subtly wrong. It is
written once here; a stage subclass supplies only what is specific to it:

    plan()              -> {key: spec}   the units and what each needs
    execute(key, spec)  -> dict | None   the actual work
    preflight()                          optional checks before any work
    group_key(spec)                      optional: units sharing setup
    setup_group(spec)                    optional: run once per group

The base class provides registration, atomic claiming, the time budget, error
capture, progress reporting and the command-line entry point.

Contract for `execute`
----------------------
It must be safe to call twice for the same key. Write outputs to a temporary
path and rename, then let the ledger mark the unit DONE. A crash between the
two costs one redundant unit and never corrupts an output.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import signal
import socket
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from jobstate import JobLedger
from logsetup import (configure, format_foreign, passthrough,
                      worker_id)


class Stage(ABC):
    """A resumable unit-of-work pipeline stage."""

    name: str = "stage"
    description: str = ""

    # What one ledger unit actually is, for the progress line. Without it the
    # monitor prints a bare "done 0/2" next to a stage-specific trial counter,
    # and the two numbers look like they should agree when they measure
    # different things — one whole Optuna study is a single unit containing
    # fifty trials, so the unit count sits at zero for hours while trials
    # complete steadily underneath it.
    unit_name: str = "unit"

    # Set True only when the stage's work is internally coordinated, so that
    # several workers may share one unit. Optuna qualifies: its storage assigns
    # trials, so extra workers on one study just finish it sooner. For every
    # other stage this must stay False or two workers would repeat the same
    # computation.
    allow_concurrent: bool = False

    # How many times a unit may be claimed before the ledger gives up on it.
    # A concurrent stage needs a high ceiling: every worker sharing a unit
    # counts as one claim, so the default of 3 would leave 49 of 52 workers
    # with nothing to do.
    max_attempts: int = 3

    def __init__(self, ledger_db: str, verbose: bool = True,
                 report_every: int = 10, log_level: str = "INFO",
                 logfile: Optional[str] = None):
        self.ledger = JobLedger(ledger_db, self.name,
                                max_attempts=self.max_attempts)
        self.verbose = verbose
        self.report_every = report_every
        self.logger = configure(self.name, level=log_level, logfile=logfile)
        self._groups: Dict[Any, Any] = {}
        self._plan: Dict[str, Any] = {}

    # ------------------------------------------------------------- contract
    @abstractmethod
    def plan(self) -> Dict[str, Any]:
        """Return {unit_key: spec}. Called on every start; must be cheap."""

    @abstractmethod
    def execute(self, key: str, spec: Any) -> Optional[dict]:
        """
        Perform one unit. Must be idempotent. Return metadata to store on the
        ledger row, or None.
        """

    # --------------------------------------------------------- optional hooks
    def preflight(self) -> None:
        """Checks that must pass before any unit runs. Raise to abort."""
        return None

    def output_exists(self, key: str, spec: Any) -> Optional[bool]:
        """
        Does this unit's output actually exist? None when the stage cannot say.

        The ledger records that work was done; it cannot know whether the file
        it produced is still there. Delete a results directory without
        resetting the ledger and every unit stays DONE while its output is
        gone — the stage then reports complete and the missing runs are simply
        absent from the analysis. That happened here: `results_causal` was
        removed, the deploy ledger was not, and one of two units never re-ran.

        A stage that can cheaply check its own output overrides this, and the
        base class reopens whatever has gone missing.
        """
        return None

    def group_key(self, spec: Any) -> Any:
        """
        Units that share expensive setup return the same group key. Used to run
        `setup_group` once instead of per unit.
        """
        return None

    def setup_group(self, gkey: Any, spec: Any) -> Any:
        """Run once per group. The return value is passed back via `group`."""
        return None

    def group(self, spec: Any) -> Any:
        """Cached group object for this spec."""
        gkey = self.group_key(spec)
        if gkey is None:
            return None
        if gkey not in self._groups:
            self._groups[gkey] = self.setup_group(gkey, spec)
        return self._groups[gkey]

    # ------------------------------------------------------------------- run
    def log(self, msg: str) -> None:
        """Progress. See logsetup for the format and why each field is there."""
        if self.verbose:
            self.logger.info(msg)

    def warn(self, msg: str) -> None:
        self.logger.warning(msg)

    def run(self, max_seconds: Optional[float] = None) -> None:
        """
        Claim and execute units until none are left or the time budget runs
        out. Safe to run concurrently from several SLURM array tasks.
        """
        self._plan = self.plan()
        self.ledger.register_many(self._plan.keys())

        # Confine this worker to the units it was actually asked for. The
        # ledger holds every unit ever planned for this stage, including those
        # of other invocations with different --methods or --W.
        self.ledger.set_scope(self._plan.keys())

        self.log(f"[{self.name}] {len(self._plan)} units planned")
        self._reopen_missing_outputs()
        self.log(self.ledger.progress_line())

        self.preflight()

        # Release whatever this worker holds if it is killed.
        #
        # Ctrl+C, a closed terminal, or SLURM's wall-clock SIGTERM all end the
        # process while it owns a unit. Without this the unit stays RUNNING and
        # the ledger will not hand it to anyone for `stale_after` — six hours
        # of a job that looks busy and is not. It happened here: seven tuning
        # units sat claimed by dead workers, and the next run reported
        # "pending 0" and did nothing.
        self._current: Optional[str] = None
        self._install_release_on_signal()

        t0 = time.time()
        n = 0
        while True:
            if max_seconds is not None and time.time() - t0 > max_seconds:
                self.log(f"[{self.name}] time budget reached; stopping cleanly")
                break

            unit = self.ledger.claim(allow_running=self.allow_concurrent)
            if unit is None:
                self.log(f"[{self.name}] nothing left to claim")
                break

            spec = self._plan.get(unit.key)
            if spec is None:
                # With the scope in place this should be unreachable. If it
                # ever happens, hand the unit back rather than claiming it was
                # finished — an unexecuted unit recorded as DONE is a hole in
                # the cache that no later stage can detect.
                self.ledger.release(unit.key)
                self.log(f"[{self.name}] released out-of-scope unit {unit.key}")
                break

            self._current = unit.key
            try:
                meta = self.execute(unit.key, spec)
                self.ledger.done(unit.key, meta)
                self._current = None
                n += 1
                if self.report_every and n % self.report_every == 0:
                    rate = n / max(time.time() - t0, 1e-9) * 3600
                    self.log(f"{self.ledger.progress_line()}  "
                             f"{n} done this worker, {rate:.0f}/hour")
            except Exception as exc:  # noqa: BLE001
                # exception() attaches the traceback to this record, so it is
                # stamped and attributed like everything else instead of being
                # dumped raw into the middle of another worker's output.
                self.logger.exception(f"unit failed: {unit.key}")
                self.ledger.fail(unit.key, f"{type(exc).__name__}: {exc}")
                self._current = None

        self.log(self.ledger.progress_line())

    def _install_release_on_signal(self) -> None:
        """
        Hand the held unit back before dying, instead of orphaning it.

        The handler must refuse to act in a forked child. Libraries below us
        fork worker pools of their own, and a fork inherits both the handler
        and `self._current`. When such a pool shuts down, its children take
        the signal and every one of them releases a unit that it does not own
        and that the real owner is still computing — the unit goes back to
        PENDING, another worker claims it, and the stage makes no progress
        while appearing busy. Pinning the owning PID at install time is what
        keeps `release` meaning "I am giving up my unit".
        """
        owner_pid = os.getpid()

        def handler(signum, frame):
            if os.getpid() != owner_pid:
                raise KeyboardInterrupt
            key = getattr(self, "_current", None)
            if key:
                try:
                    self.ledger.release(key)
                    self.warn(f"interrupted; released {key} back to PENDING")
                except Exception:                             # noqa: BLE001
                    pass
            raise KeyboardInterrupt

        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not the main thread, or unsupported on this platform.
                pass

    def _reopen_missing_outputs(self) -> None:
        """Send DONE units whose output has disappeared back to PENDING."""
        reopened = 0
        for key in self.ledger.done_keys():
            spec = self._plan.get(key)
            if spec is None:
                continue
            try:
                present = self.output_exists(key, spec)
            except Exception:                                 # noqa: BLE001
                present = None                                # cannot tell
            if present is False:
                self.ledger.reopen(key)
                reopened += 1
        if reopened:
            self.warn(f"{reopened} unit(s) were marked DONE but their output "
                      f"is missing; they will be recomputed")

    # ---------------------------------------------------------------- status
    def status(self, show_failures: bool = False) -> None:
        self._plan = self.plan()
        self.ledger.register_many(self._plan.keys())
        print(self.ledger.progress_line())
        if show_failures:
            for key, attempts, err in self.ledger.failures():
                print(f"  FAILED {key} (attempts={attempts}): {err}")

    def reset(self, key: Optional[str] = None) -> None:
        self.ledger.force_reset(key)
        print(f"[{self.name}] reset {'all units' if key is None else key}")

    # --------------------------------------------------------------- workers
    @staticmethod
    def default_workers() -> int:
        """
        How many worker processes to use when none is specified.

        On SLURM this is the allocation itself: a 52-core hamsi node should run
        52 workers, not one. Falling back to the CPU count keeps local runs
        sensible too.
        """
        for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
            if os.environ.get(var):
                try:
                    return max(1, int(os.environ[var]))
                except ValueError:
                    pass
        return max(1, (os.cpu_count() or 2) - 1)

    @classmethod
    def spawn_workers(cls, n: int, argv: list,
                      scope: Optional[list] = None) -> int:
        """
        Launch `n` copies of this script as separate processes.

        They coordinate through the ledger, so no further communication is
        needed: each claims its own units, and a worker that finds nothing left
        exits. Every worker pins itself to one BLAS thread, otherwise 52
        processes would each spawn 52 threads and thrash the node.
        """
        # Divide the allocated cores among the workers instead of pinning
        # every one of them to a single thread.
        #
        # One thread each is right when the workers *are* the parallelism:
        # 56 decomposition workers on 56 cores would otherwise spawn 56
        # threads apiece and thrash the node, which is why this was hardcoded
        # to 1. But on a GPU node the shape is different — four workers hold
        # forty cores, and pinning them to one thread each leaves the data
        # pipeline single-threaded while the card waits for it.
        #
        # The floor division gives the old behaviour wherever it was correct:
        # 56 cores over 56 workers is still 1.
        per_worker = max(1, cls.default_workers() // max(n, 1))
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(per_worker)
        if per_worker > 1:
            print(f"  threads     : {per_worker} per worker "
                  f"({cls.default_workers()} cores / {n} workers)")

        cmd = [sys.executable, sys.argv[0]] + argv

        # On a multi-GPU node, give each worker its own card.
        #
        # config.DEVICE is torch.device("cuda"), which is card 0 for everyone.
        # Four workers on a four-GPU node would therefore queue for one card
        # and leave three idle — and since the CUDA queues here allocate the
        # whole node, that is three quarters of the allocation wasted, which
        # is exactly the pattern the site's efficiency monitor complains
        # about. Handing each worker a different CUDA_VISIBLE_DEVICES makes
        # its card the only one it can see, so "cuda" resolves to a different
        # physical GPU per worker with no change to the training code.
        gpus = _visible_gpu_count()
        if gpus > 1:
            print(f"  gpus        : {gpus}  (worker i -> card i % {gpus})")

        # Each worker writes to its own log. With 52 processes printing to one
        # terminal the output interleaves into noise; this process stays quiet
        # and prints a progress line the reader can follow, while the detail
        # goes to the per-worker files.
        log_dir = os.path.join(os.getcwd(), "logs", "workers")
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        print(f"\n{'=' * 72}")
        print(f"  stage       : {cls.name}")
        print(f"  started     : {_now()}")
        print(f"  host        : {socket.gethostname().split('.')[0]}"
              + (f"   SLURM job {os.environ['SLURM_JOB_ID']}"
                 if os.environ.get("SLURM_JOB_ID") else ""))
        print(f"  workers     : {n}  (w000 .. w{n - 1:03d})")
        print(f"  command     : {' '.join(cmd)}")
        print(f"  worker logs : logs/workers/{cls.name}_{stamp}_NNN.log")
        print(f"{'=' * 72}\n", flush=True)

        procs, handles, readers = [], [], []
        host = socket.gethostname().split(".")[0]
        for i in range(n):
            path = os.path.join(log_dir, f"{cls.name}_{stamp}_{i:03d}.log")
            fh = open(path, "w", buffering=1)
            handles.append(fh)
            # Each child knows its own index, so a ledger row naming w007 maps
            # straight to ..._007.log.
            child_env = dict(env, _STAGE_WORKER_ID=str(i))
            if gpus > 1:
                child_env["CUDA_VISIBLE_DEVICES"] = str(i % gpus)
            p = subprocess.Popen(cmd, env=child_env,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 text=True, bufsize=1,
                                 errors="replace")
            procs.append(p)
            t = threading.Thread(target=cls._tee, daemon=True,
                                 args=(p, fh, f"{host}/w{i:03d}"))
            t.start()
            readers.append(t)

        rc = cls._watch(procs, handles, n, scope=scope)
        for t in readers:
            t.join(timeout=5.0)
        return rc

    # Serialises writes from the tee threads and the monitor, so two lines
    # never interleave halfway through.
    _print_lock = threading.Lock()

    @classmethod
    def _emit(cls, line: str) -> None:
        with cls._print_lock:
            passthrough().info(line)

    @classmethod
    def _tee(cls, proc, fh, worker: str) -> None:
        """
        Copy one worker's output to its log file and to the screen.

        Both destinations, not one: the per-worker file is what you grep when a
        single worker misbehaves, and the combined stream is what SLURM records
        in the job's .out — the only thing you have if the job dies and you are
        not watching.

        Lines the worker already formatted pass through untouched. Anything
        else (a traceback, a library warning) is unattributed, so it is wrapped
        in the same format; otherwise a stack trace in a 52-worker log belongs
        to nobody.
        """
        level = "INFO"
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                fh.write(line + "\n")
                if " | " in line[:40]:
                    # Already stamped by the worker. Remember its level: the
                    # lines that follow may be the continuation of a multi-line
                    # record — a traceback — and marking those INFO would hide
                    # the body of an error behind an ordinary-looking prefix.
                    parts = line.split(" | ")
                    if len(parts) >= 4:
                        level = parts[3].strip() or level
                    cls._emit(line)
                else:
                    cls._emit(format_foreign(worker, cls.name, line, level))
        except (ValueError, OSError):
            pass                                       # pipe closed at exit
        finally:
            try:
                proc.stdout.close()
            except Exception:                          # noqa: BLE001
                pass

    @classmethod
    def _print_roster(cls, ledger, limit: int = 12) -> None:
        """
        Who is working on what, right now.

        With 52 workers the aggregate counters cannot distinguish "everything
        is moving" from "forty workers finished and twelve are wedged". The
        longest-held units come first, because those are the ones worth
        looking at.
        """
        rows = ledger.running_units()
        if not rows:
            return
        rows.sort(key=lambda r: -r[2])
        # cls.name, not Stage.name: the base class attribute is the literal
        # string "stage", so these lines were labelled with it instead of the
        # stage they describe.
        who = f"{_host()}/monitor"
        cls._emit(format_foreign(
            who, cls.name,
            f"---- roster: {len(rows)} unit(s) claimed, longest first ----"))
        for key, worker, held in rows[:limit]:
            detail = cls.unit_detail(key)
            cls._emit(format_foreign(
                who, cls.name,
                f"    {worker:<16s} held {_hms(held):>9s}  {key}"
                + (f"   {detail}" if detail else "")))
        if len(rows) > limit:
            cls._emit(format_foreign(
                who, cls.name, f"    ... and {len(rows) - limit} more"))

    @classmethod
    def unit_detail(cls, key: str) -> str:
        """
        Per-unit progress for the roster, or "".

        A unit that takes hours needs to say how far along it is. "held 0:40:23"
        tells you a worker is busy; it does not tell you whether that unit is
        nearly done or has barely started, and those call for different
        reactions.
        """
        return ""

    @classmethod
    def extra_progress(cls) -> str:
        """
        Stage-specific detail for the progress line, or "".

        Unit counts are the wrong resolution for a stage whose units take an
        hour. An Optuna study is one unit, so the monitor sat on "done 0/2" for
        half an hour while the search was in fact working — indistinguishable
        from a hang. A stage that knows about finer-grained progress reports it
        here.
        """
        return ""

    @classmethod
    def _watch(cls, procs, handles, n: int, interval: float = 20.0,
               scope: Optional[list] = None) -> int:
        """Report ledger progress while the workers run."""
        from config import LEDGER_DB
        ledger = JobLedger(LEDGER_DB, cls.name)
        # Count only what this invocation asked for, or the monitor reports the
        # whole stage and a subset run looks either finished or barely started.
        if scope:
            ledger.set_scope(scope)

        t0 = time.time()
        last = None
        last_roster = time.time()
        roster_every = 300.0            # the full assignment table, every 5 min
        try:
            while any(p.poll() is None for p in procs):
                # Sleep first: at t=0 the workers have not registered their
                # units yet, so an immediate read would report a misleading 0/0.
                time.sleep(min(interval, 5.0) if last is None else interval)
                counts = ledger.counts()
                alive = sum(1 for p in procs if p.poll() is None)
                detail = cls.extra_progress()
                noun = cls.unit_name + ("s" if counts["TOTAL"] != 1 else "")
                body = (f"{noun} {counts['DONE']}/{counts['TOTAL']} done  "
                        f"{counts['RUNNING']} running  "
                        f"{counts['PENDING']} pending  "
                        f"{counts['FAILED']} failed  |  "
                        f"workers {alive}/{n}"
                        + (f"  |  {detail}" if detail else ""))
                # Compare without the clock, or the line would reprint every
                # interval purely because the time moved on.
                if body != last:
                    cls._emit(format_foreign(f"{_host()}/monitor",
                                             cls.name, body))
                    last = body

                # Periodically, who is holding what. The counters say how much
                # is left; this says which unit a worker has been sitting on.
                if time.time() - last_roster >= roster_every:
                    cls._print_roster(ledger)
                    last_roster = time.time()
        except KeyboardInterrupt:
            print("\n  interrupted — terminating workers", flush=True)
            for p in procs:
                p.terminate()

        failed = sum(1 for p in procs if p.wait() != 0)
        for fh in handles:
            fh.close()

        counts = ledger.counts()
        print(f"\n  finished  {_now()}  (ran {_hms(time.time() - t0)})")
        print(f"  done {counts['DONE']}/{counts['TOTAL']}, "
              f"failed {counts['FAILED']}, "
              f"{n - failed}/{n} workers exited cleanly\n", flush=True)
        if counts["FAILED"]:
            print(f"  inspect with: python run.py <stage> "
                  f"--status --failures\n", flush=True)
        return 1 if failed == n else 0

    # ------------------------------------------------------------------- CLI
    @classmethod
    def add_common_arguments(cls, ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--status", action="store_true",
                        help="print progress and exit")
        ap.add_argument("--failures", action="store_true",
                        help="with --status, list failed units")
        ap.add_argument("--reset", metavar="KEY", nargs="?", const="__ALL__",
                        help="reset one unit, or the whole stage if no key")
        ap.add_argument("--max-seconds", type=float, default=None,
                        help="stop claiming new units after this long; leave "
                             "head-room before the SLURM wall clock")
        ap.add_argument("--workers", type=int, default=None,
                        help="parallel worker processes; defaults to the "
                             "SLURM allocation, or CPU count locally. "
                             "Use 1 to stay in this process.")
        ap.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="verbosity; DEBUG adds per-epoch detail")
        ap.add_argument("--log-file", default=None,
                        help="also write this process's log here. Worker "
                             "processes already get logs/workers/*.log")

    def main_from_args(self, args: argparse.Namespace) -> None:
        """Standard dispatch shared by every stage's __main__ block."""
        # The stage was constructed before the flags were parsed, so the
        # logger is rebuilt here with whatever was asked for. configure()
        # replaces handlers rather than adding to them, so this is safe.
        self.logger = configure(self.name,
                                level=getattr(args, "log_level", "INFO"),
                                logfile=getattr(args, "log_file", None))

        if args.reset:
            # Reset is an operation on its own, not a prefix to a run. Falling
            # through used to start the full default grid immediately after
            # clearing it, which is never what someone asking to reset wants.
            self.reset(None if args.reset == "__ALL__" else args.reset)
            print(f"[{self.name}] nothing was run. Re-issue the command "
                  f"without --reset to start work.")
            return
        if args.status:
            self.status(show_failures=getattr(args, "failures", False))
            return

        workers = args.workers if args.workers is not None else self.default_workers()

        # Never start more workers than there is work. Without this, a
        # one-unit stage spawned fifteen processes, fourteen of which existed
        # only to find nothing to claim and exit. Concurrent stages are exempt:
        # there, several workers on one unit is the point.
        #
        # What counts is work that is still *outstanding*, not work that was
        # planned. On a GPU queue the two differ in a way that costs real
        # money: resubmitting a job whose units are nearly all DONE would take
        # four cards and give three of them nothing to do, and the whole node
        # is billed either way. Registering here also means the children find
        # the ledger already populated, which is the read-before-write path
        # that fixed the start-up stampede.
        scope = list(self.plan().keys())
        if workers > 1 and not self.allow_concurrent:
            self.ledger.register_many(scope)
            state = self.ledger.states()
            todo = [k for k in scope if state.get(k) != "DONE"]
            if todo and len(todo) < workers:
                self.log(f"[{self.name}] {len(todo)} unit(s) outstanding "
                         f"({len(scope)} planned); using {len(todo)} "
                         f"worker(s) instead of {workers}")
                workers = len(todo)

        if workers > 1 and not os.environ.get("_STAGE_WORKER"):
            # Re-invoke this script N times with --workers 1; the children do
            # the work and the ledger keeps them from colliding.
            child_argv = _drop_workers_value(sys.argv[1:]) + ["--workers", "1"]
            os.environ["_STAGE_LOG_LEVEL"] = getattr(args, "log_level", "INFO")
            os.environ["_STAGE_WORKER"] = "1"
            sys.exit(self.spawn_workers(workers, child_argv, scope))

        self.run(max_seconds=args.max_seconds)


def _visible_gpu_count() -> int:
    """
    How many CUDA devices this process may use. 0 when there are none.

    Read from the environment before torch is imported, so that a CPU-only
    stage never pays for `import torch` just to find out there is no GPU.
    CUDA_VISIBLE_DEVICES wins when set, because SLURM writes it and it is what
    torch will actually honour; SLURM_GPUS_ON_NODE is the fallback.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return len([p for p in vis.split(",") if p.strip() != ""])
    for var in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"):
        raw = os.environ.get(var, "")
        # "gpu:4" and "4" are both seen depending on how the job asked.
        digits = raw.split(":")[-1]
        if digits.isdigit():
            return int(digits)
    return 0


def _host() -> str:
    return socket.gethostname().split(".")[0]


def _now() -> str:
    """
    Wall-clock stamp, with the date.

    A cluster job can run for three days, so an hour-minute-second stamp is
    ambiguous the moment it crosses midnight — and every long run does.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _drop_workers_value(argv: list) -> list:
    """Remove a `--workers N` pair from an argv list."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--workers":
            skip = True
            continue
        if a.startswith("--workers="):
            continue
        out.append(a)
    return out
