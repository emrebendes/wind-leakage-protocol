# -*- coding: utf-8 -*-
"""
Logging for the whole project.
==============================

One format, everywhere:

    2026-08-08 22:14:03 | hamsi12/w007 | optuna       | INFO    | message

Fields, and why each is there:

    timestamp   with the date. A cluster job runs for days, so a time-only
                stamp becomes ambiguous the first midnight it crosses.
    worker      host and worker index, e.g. `hamsi12/w007`. It names the file
                `logs/workers/<stage>_<stamp>_007.log` and it is what the job
                ledger records against a claimed unit, so a line, a log file
                and a database row all point at each other.
    stage       which pipeline step. Several may appear in one SLURM .out.
    level       INFO for progress, WARNING for something survivable, ERROR
                for a failed unit. Filterable with `--log-level`.

Why `logging` rather than `print`
---------------------------------
Verbosity becomes a flag instead of an edit; a file handler is attached in one
place rather than by shell redirection; and warnings from numpy, torch and
Optuna land in the same stream in the same format instead of arriving
unattributed and out of order.

Two loggers
-----------
`configure` sets up the one everything writes to. `passthrough` is for the
parent process relaying a child's already-formatted lines — running those
through the normal formatter would stamp them twice, with the parent's clock
and the parent's identity, both wrong.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from typing import Optional

LOG_FORMAT = ("%(asctime)s | %(worker)-14s | %(stage)-12s | "
              "%(levelname)-7s | %(message)s")
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOGGER_NAME = "wb"
PASSTHROUGH_NAME = "wb.raw"


def worker_id() -> str:
    """
    This process's identity.

    `_STAGE_WORKER_ID` is set by Stage.spawn_workers, so a child's index maps
    straight to its log file. A process started by hand has no index and falls
    back to its pid.
    """
    host = socket.gethostname().split(".")[0]
    idx = os.environ.get("_STAGE_WORKER_ID")
    if idx is not None:
        try:
            return f"{host}/w{int(idx):03d}"
        except ValueError:
            pass
    return f"{host}/{os.getpid()}"


class MultiLineFormatter(logging.Formatter):
    """
    Prefix every line of a record, not just the first.

    A traceback is one log record with embedded newlines, so the default
    formatter stamps the header and leaves the remaining twenty lines bare. In
    a log with 52 workers writing at once those bare lines are unattributable —
    and they are the ones that say what actually went wrong. Here the prefix is
    repeated, with a marker so a continuation is not mistaken for a new event.
    """

    CONT = "  |"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if "\n" not in text:
            return text
        head, *rest = text.split("\n")
        prefix = head.rsplit(" | ", 1)[0] + " | "
        return "\n".join([head] + [f"{prefix}{self.CONT} {r}" for r in rest])


class _Context(logging.Filter):
    """Adds `worker` and `stage` to every record so the format can use them."""

    def __init__(self, stage: str, worker: str):
        super().__init__()
        self.stage = stage
        self.worker = worker

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "worker"):
            record.worker = self.worker
        if not hasattr(record, "stage"):
            record.stage = self.stage
        return True


def configure(stage: str, level: str = "INFO",
              logfile: Optional[str] = None,
              worker: Optional[str] = None) -> logging.Logger:
    """
    Set up the project logger. Safe to call twice; handlers are not duplicated.

    Everything goes to stdout, including errors. On SLURM that puts the whole
    story in one file in causal order — with stdout and stderr split, a
    traceback in the .err cannot be lined up against the progress in the .out.
    """
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    log.propagate = False

    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()

    ctx = _Context(stage, worker or worker_id())
    fmt = MultiLineFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    stream.addFilter(ctx)
    log.addHandler(stream)

    if logfile:
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(ctx)
        log.addHandler(fh)

    # Library warnings arrive through the warnings module by default and would
    # otherwise bypass all of this.
    logging.captureWarnings(True)
    for name in ("py.warnings", "optuna"):
        lib = logging.getLogger(name)
        lib.handlers = list(log.handlers)
        lib.propagate = False

    return log


def passthrough() -> logging.Logger:
    """
    Emit already-formatted lines verbatim.

    Used by the parent when relaying a worker's output: the child stamped and
    attributed the line when it was written, and re-formatting it here would
    replace both with the parent's, which is a lie about when and where the
    event happened.
    """
    log = logging.getLogger(PASSTHROUGH_NAME)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def format_foreign(worker: str, stage: str, message: str,
                   level: str = "INFO") -> str:
    """
    Wrap an unattributed line — a traceback, a C-library warning — in the
    project format, so that nothing in a 52-worker log belongs to nobody.
    """
    import time

    return LOG_FORMAT % {
        "asctime": time.strftime(DATE_FORMAT),
        "worker": worker,
        "stage": stage,
        "levelname": level,
        "message": message,
    }
