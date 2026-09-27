# -*- coding: utf-8 -*-
"""
Resumable job ledger.
=====================

Every long-running stage of this project (decomposition tuning, causal feature
precompute, Optuna search, final training) is expressed as a set of small,
independently completable UNITS. This module records the state of each unit in
a SQLite database so that a job killed by the scheduler resumes exactly where
it stopped.

Design rules
------------
* One row per unit. A unit is identified by (stage, key).
* Claim-before-work: a worker atomically marks a unit RUNNING before starting.
  Concurrent SLURM array tasks therefore never duplicate work.
* Stale claims expire. If a worker dies without releasing, the claim is
  reclaimable after `stale_after` seconds.
* The database is the single source of truth for progress. Output files are
  written first, the unit is marked DONE second, so a crash between the two
  only costs one redundant unit.

Usage
-----
    ledger = JobLedger(DB_PATH, stage="precompute")
    ledger.register_many(keys)               # idempotent

    while True:
        unit = ledger.claim()                # None -> everything done
        if unit is None:
            break
        try:
            do_work(unit.key)
            ledger.done(unit.key)
        except Exception as e:
            ledger.fail(unit.key, str(e))
"""

from __future__ import annotations

import json
import os
import random
import socket
import sqlite3
import time
from dataclasses import dataclass
from typing import Iterable, Optional

PENDING = "PENDING"
RUNNING = "RUNNING"
DONE = "DONE"
FAILED = "FAILED"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    stage       TEXT NOT NULL,
    key         TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'PENDING',
    worker      TEXT,
    claimed_at  REAL,
    finished_at REAL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    meta        TEXT,
    PRIMARY KEY (stage, key)
);
CREATE INDEX IF NOT EXISTS idx_units_state ON units(stage, state);
"""


def open_sqlite(db_path: str, timeout: float = 180.0) -> sqlite3.Connection:
    """
    Open a SQLite database in a way that survives a network filesystem.

    WAL is the right journal mode for concurrent readers and writers, but it
    needs shared-memory support that Lustre and GPFS — the filesystems TRUBA
    runs on — do not always provide, and the failure surfaces as an opaque
    "disk I/O error". We therefore try WAL, verify it took effect, and fall
    back to TRUNCATE journalling if it did not.

    A long busy timeout matters either way: several array tasks hit the same
    ledger, and the loser of a race must wait rather than crash.
    """
    conn = sqlite3.connect(db_path, timeout=timeout, isolation_level=None)
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    try:
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if not mode or str(mode[0]).lower() != "wal":
            raise sqlite3.OperationalError("WAL not honoured")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        # Shared filesystem without shm support: journal to a plain file.
        conn.execute("PRAGMA journal_mode=TRUNCATE")
        conn.execute("PRAGMA synchronous=FULL")
    return conn


@dataclass
class Unit:
    stage: str
    key: str
    attempts: int
    meta: dict


class JobLedger:
    # How long a claim may stand before another worker may take the unit.
    #
    # This is a guess about death, and it has to be longer than the slowest
    # unit or it becomes a guess about slowness instead. It was 6 hours, and a
    # single ceemdan causal W=1024 chunk measured at 12-14 hours: at 13:11 w030
    # claimed one, at 22:11 w039 declared it stale and recomputed it from
    # scratch, and the job stayed alive for nine further hours doing work that
    # was already on disk. Nothing was corrupted — the seed is derived from the
    # window, so both workers wrote identical bytes — but a whole node-day was
    # spent on it.
    #
    # A worker that is killed does not need this path anyway: the signal
    # handler hands its unit back immediately. Staleness only covers a hard
    # kill (OOM, node failure), which is rare enough to be worth waiting for.
    STALE_AFTER = 30 * 3600.0

    def __init__(self, db_path: str, stage: str,
                 stale_after: float = STALE_AFTER,
                 max_attempts: int = 3):
        self.db_path = db_path
        self.stage = stage
        self.stale_after = stale_after
        self.max_attempts = max_attempts
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        # Worker identity, readable in a log with 52 of them. The index comes
        # from spawn_workers so that a line can be traced to a log file;
        # hostname distinguishes array tasks spread across nodes.
        host = socket.gethostname().split(".")[0]
        idx = os.environ.get("_STAGE_WORKER_ID")
        self._worker = (f"{host}/w{int(idx):03d}" if idx is not None
                        else f"{host}/{os.getpid()}")
        self._scope_conn = None      # persistent connection holding the scope
        with self._conn() as c:
            c.executescript(_SCHEMA)

    # ------------------------------------------------------------------ core
    def _conn(self):
        return open_sqlite(self.db_path)

    # ----------------------------------------------------------------- scope
    def set_scope(self, keys: Iterable[str]) -> None:
        """
        Restrict this ledger to the units the current invocation covers.

        A stage's ledger holds every unit anyone has ever planned for it. Run
        `precompute --methods dwt` and the VMD units are still there, claimable
        and none of this process's business. Without a scope a worker had to
        decide what to do with a unit it could not execute, and the answer it
        gave — mark it DONE — reported work as finished that was never done.

        The scope lives in a TEMP table, so it is private to this connection
        and disappears with it; concurrent workers each carry their own.
        """
        keys = list(keys)
        self._scope_conn = open_sqlite(self.db_path)
        self._scope_conn.executescript(
            "CREATE TEMP TABLE IF NOT EXISTS scope(key TEXT PRIMARY KEY);"
            "DELETE FROM scope;"
        )
        # DEFERRED, not IMMEDIATE. The scope table is TEMP — private to this
        # connection and stored outside the shared database — but a
        # transaction spans every attached database, so BEGIN IMMEDIATE would
        # reserve the write lock on the shared ledger to insert rows nothing
        # else can see. Every worker did this immediately after register_many,
        # which is the second half of the same startup stampede. A deferred
        # transaction takes locks only for what it actually touches.
        self._scope_conn.execute("BEGIN")
        self._scope_conn.executemany(
            "INSERT OR IGNORE INTO scope(key) VALUES (?)", [(k,) for k in keys])
        self._scope_conn.execute("COMMIT")

    @property
    def scoped(self) -> bool:
        return self._scope_conn is not None

    def register_many(self, keys: Iterable[str], meta: Optional[dict] = None):
        """
        Insert units if absent. Safe to call on every restart.

        Read before writing. Every worker of every job calls this with the
        same plan, so the insert is a no-op for all but the first — but
        `BEGIN IMMEDIATE` takes the exclusive write lock regardless, and the
        no-ops queue behind each other just as real writes would. With 640
        workers starting at once that queue outlived the busy timeout and the
        latecomers died at startup:

            sqlite3.OperationalError: database is locked
              ... in register_many -> c.execute("BEGIN IMMEDIATE")

        The jobs survived on whatever workers happened to win the race — one
        was down to 6 of 56 — so the symptom was not a crash but a node
        running at a tenth of its allocation.

        Checking first turns 640 write transactions into 640 cheap reads and
        one write. The write still retries, because two workers can find the
        same keys missing at the same time.
        """
        keys = list(keys)
        payload = json.dumps(meta or {})

        with self._conn() as c:
            existing = {r[0] for r in c.execute(
                "SELECT key FROM units WHERE stage = ?", (self.stage,))}
        missing = [k for k in keys if k not in existing]
        if not missing:
            return

        delay = 0.5
        for attempt in range(8):
            try:
                with self._conn() as c:
                    c.execute("BEGIN IMMEDIATE")
                    c.executemany(
                        "INSERT OR IGNORE INTO units(stage, key, meta) "
                        "VALUES (?,?,?)",
                        [(self.stage, k, payload) for k in missing],
                    )
                    c.execute("COMMIT")
                return
            except sqlite3.OperationalError:
                if attempt == 7:
                    raise
                # Jitter, or the losers retry in step and collide again.
                time.sleep(delay * (1.0 + random.random()))
                delay = min(delay * 2, 30.0)

    def claim(self, allow_running: bool = False) -> Optional[Unit]:
        """
        Atomically take ownership of one unit of work.
        Returns None when nothing is left to do.

        `allow_running` lets several workers share a unit that is already in
        progress. Only stages whose work is internally coordinated may use it —
        Optuna is the case here, because its own storage assigns trials, so
        putting more workers on one study simply finishes it sooner. For every
        other stage this must stay False or two workers would compute the same
        chunk.
        """
        now = time.time()
        cutoff = now - self.stale_after
        running_clause = ("OR state = 'RUNNING'" if allow_running
                          else "OR (state = 'RUNNING' AND claimed_at < ?)")
        params = ([self.stage, self.max_attempts]
                  + ([] if allow_running else [cutoff]))

        # Only ever hand back a unit this invocation is able to execute.
        scope_clause = "AND key IN (SELECT key FROM scope)" if self.scoped else ""

        conn = self._scope_conn if self.scoped else self._conn()
        try:
            c = conn
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                f"""
                SELECT key, attempts, meta FROM units
                 WHERE stage = ?
                   {scope_clause}
                   AND attempts < ?
                   AND ( state = 'PENDING'
                      OR state = 'FAILED'
                      {running_clause} )
                 ORDER BY state = 'PENDING' DESC, attempts ASC
                 LIMIT 1
                """,
                params,
            ).fetchone()
            if row is None:
                c.execute("COMMIT")
                return None
            key, attempts, meta = row
            c.execute(
                """UPDATE units
                      SET state='RUNNING', worker=?, claimed_at=?, attempts=attempts+1
                    WHERE stage=? AND key=?""",
                (self._worker, now, self.stage, key),
            )
            c.execute("COMMIT")
        finally:
            if not self.scoped:
                conn.close()
        return Unit(self.stage, key, attempts + 1, json.loads(meta or "{}"))

    def done(self, key: str, meta: Optional[dict] = None):
        with self._conn() as c:
            if meta is None:
                c.execute(
                    "UPDATE units SET state='DONE', finished_at=?, error=NULL "
                    "WHERE stage=? AND key=?",
                    (time.time(), self.stage, key),
                )
            else:
                c.execute(
                    "UPDATE units SET state='DONE', finished_at=?, error=NULL, meta=? "
                    "WHERE stage=? AND key=?",
                    (time.time(), json.dumps(meta), self.stage, key),
                )

    def release(self, key: str):
        """
        Return a claimed unit to PENDING without counting the attempt.

        Used when a worker claims a unit that belongs to a different
        invocation — say this process was started with `--methods dwt` and the
        unit is a VMD one. The unit is not done and not failed; it simply is
        not this worker's to do, and the next invocation that does cover it
        must still find it waiting.

        Marking such units DONE instead is silently destructive: the stage then
        reports complete while the outputs were never computed, and every later
        stage reads a cache with holes in it. That happened here — 6,656 of
        7,488 units were recorded as finished by a run that only ever computed
        832 of them.
        """
        with self._conn() as c:
            c.execute(
                "UPDATE units SET state='PENDING', worker=NULL, "
                "claimed_at=NULL, attempts=MAX(attempts-1, 0) "
                "WHERE stage=? AND key=? AND state='RUNNING'",
                (self.stage, key),
            )

    def fail(self, key: str, error: str):
        with self._conn() as c:
            c.execute(
                "UPDATE units SET state='FAILED', finished_at=?, error=? "
                "WHERE stage=? AND key=?",
                (time.time(), error[:4000], self.stage, key),
            )

    def force_reset(self, key: Optional[str] = None):
        """Clear state so units run again. Without a key, resets the stage."""
        with self._conn() as c:
            if key is None:
                c.execute(
                    "UPDATE units SET state='PENDING', attempts=0, error=NULL, "
                    "worker=NULL, claimed_at=NULL, finished_at=NULL WHERE stage=?",
                    (self.stage,),
                )
            else:
                c.execute(
                    "UPDATE units SET state='PENDING', attempts=0, error=NULL, "
                    "worker=NULL, claimed_at=NULL, finished_at=NULL "
                    "WHERE stage=? AND key=?",
                    (self.stage, key),
                )

    # ------------------------------------------------------------- reporting
    def states(self) -> dict:
        """
        key -> state, for every unit of this stage.

        counts() aggregates; a status display that names each unit needs the
        rows themselves. Unscoped on purpose: a status command run from the
        login node has no scope and should show the whole stage.
        """
        with self._conn() as c:
            return dict(c.execute(
                "SELECT key, state FROM units WHERE stage=?", (self.stage,)))

    def counts(self) -> dict:
        # Counts follow the scope. Reporting the whole stage while working on a
        # subset is how "7488/7488 done" appeared for a run that computed 832.
        if self.scoped:
            rows = self._scope_conn.execute(
                "SELECT state, COUNT(*) FROM units WHERE stage=? "
                "AND key IN (SELECT key FROM scope) GROUP BY state",
                (self.stage,),
            ).fetchall()
            d = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0}
            d.update({k: v for k, v in rows})
            d["TOTAL"] = sum(v for k, v in d.items() if k != "TOTAL")
            return d

        with self._conn() as c:
            rows = c.execute(
                "SELECT state, COUNT(*) FROM units WHERE stage=? GROUP BY state",
                (self.stage,),
            ).fetchall()
        d = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0}
        d.update({k: v for k, v in rows})
        d["TOTAL"] = sum(v for k, v in d.items() if k != "TOTAL")
        return d

    def done_keys(self):
        """Keys currently marked DONE, within the scope if one is set."""
        clause = ("AND key IN (SELECT key FROM scope)" if self.scoped else "")
        conn = self._scope_conn if self.scoped else self._conn()
        try:
            rows = conn.execute(
                f"SELECT key FROM units WHERE stage=? AND state='DONE' {clause}",
                (self.stage,),
            ).fetchall()
        finally:
            if not self.scoped:
                conn.close()
        return [r[0] for r in rows]

    def reopen(self, key: str):
        """Send a DONE unit back to PENDING and clear its attempts."""
        with self._conn() as c:
            c.execute(
                "UPDATE units SET state='PENDING', attempts=0, worker=NULL, "
                "claimed_at=NULL, finished_at=NULL WHERE stage=? AND key=?",
                (self.stage, key),
            )

    def running_units(self):
        """
        (key, worker, seconds_held) for everything currently claimed.

        On a 52-worker node the aggregate counters say how much is left but not
        who is stuck on what. This is the view that answers "which unit has
        that worker been sitting on for two hours".

        Scoped, like counts(): six optuna jobs share one ledger, and without
        the filter the eemd job's roster listed the none job's studies on
        another node. Reading a job's log then told you about work it was not
        doing.
        """
        now = time.time()
        conn = self._scope_conn if self.scoped else self._conn()
        scope_clause = ("AND key IN (SELECT key FROM scope)"
                        if self.scoped else "")
        try:
            rows = conn.execute(
                "SELECT key, worker, claimed_at FROM units "
                f"WHERE stage=? AND state='RUNNING' {scope_clause} "
                "ORDER BY claimed_at",
                (self.stage,),
            ).fetchall()
        finally:
            if not self.scoped:
                conn.close()
        return [(k, w or "?", now - (t or now)) for k, w, t in rows]

    def failures(self):
        with self._conn() as c:
            return c.execute(
                "SELECT key, attempts, error FROM units "
                "WHERE stage=? AND state='FAILED' ORDER BY key",
                (self.stage,),
            ).fetchall()

    def is_complete(self) -> bool:
        c = self.counts()
        return c["TOTAL"] > 0 and c[DONE] == c["TOTAL"]

    def progress_line(self) -> str:
        c = self.counts()
        return (f"[{self.stage}] done {c[DONE]}/{c['TOTAL']}  "
                f"running {c[RUNNING]}  pending {c[PENDING]}  failed {c[FAILED]}")


# --------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect or reset a job ledger")
    ap.add_argument("db")
    ap.add_argument("stage")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--reset", metavar="KEY", nargs="?", const="__ALL__")
    a = ap.parse_args()

    led = JobLedger(a.db, a.stage)
    if a.reset:
        led.force_reset(None if a.reset == "__ALL__" else a.reset)
        print("reset ->", a.reset)
    print(led.progress_line())
    if a.failures:
        for k, n, e in led.failures():
            print(f"  {k}  (attempts={n})  {e}")
