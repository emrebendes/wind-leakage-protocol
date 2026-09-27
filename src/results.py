# -*- coding: utf-8 -*-
"""
Result storage — collision proof.
=================================

This study runs a large combinatorial grid: arm x decomposition x model x W x
look_back x horizon x seed x station. With that many outputs, "two runs wrote
to the same filename" is not a hypothetical risk, and a silently overwritten
result is worse than a crash because it is invisible.

Three mechanisms prevent it.

1. IDENTITY IS THE CONFIGURATION.
   `RunSpec` holds every field that distinguishes one run from another. The
   storage path is derived from it, and a short hash of the *complete* spec is
   appended to the filename. Two runs that differ in any field — including one
   the directory layout does not encode — get different files.

2. WRITES REFUSE TO CLOBBER.
   `ResultStore.save` raises if the target exists, unless `overwrite=True` is
   passed explicitly. Accidental re-runs stop instead of destroying data.

3. PATH/SPEC AGREEMENT IS CHECKED.
   Every result is written next to its `spec.json`. If a path already holds a
   spec that differs from the one being written, that is a hash collision or a
   layout bug, and it raises `ResultCollision` rather than continuing.

A SQLite index records every completed run so the analysis stage can query the
grid without walking the filesystem.

Usage
-----
    spec = RunSpec(arm="causal", decomposition="dwt", model="bilstm",
                   W=512, look_back=48, seed=0)
    store = ResultStore(RESULTS_DIR)

    if store.exists(spec):
        return                       # already done, skip

    store.save(spec, metrics=..., predictions=...)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomicio import atomic_path, write_npz


class ResultCollision(RuntimeError):
    """Two different configurations resolved to the same storage location."""


class ResultExists(FileExistsError):
    """A result is already stored for this configuration."""


# =============================================================================
@dataclass(frozen=True)
class RunSpec:
    """
    The complete identity of one run.

    Every field that could change the numbers belongs here. Anything omitted
    becomes an invisible confound: two runs differing only in that field would
    collide, and mechanism 3 would raise — which is the intended behaviour, but
    the field should simply be added instead.
    """

    arm: str                       # "causal" | "leaky" | "deployment_gap"
    decomposition: str             # none | emd | eemd | ceemdan | vmd | dwt
    model: str                     # lstm | bilstm | gru | tcn | tcan | ...
    W: Optional[int] = None        # causal decomposition window
    look_back: Optional[int] = None
    seed: int = 0
    station: Optional[int] = None  # None = pooled across stations
    horizon_set: str = "1-8-16-24"
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- identity
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Short deterministic hash of the entire specification."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def relative_dir(self) -> str:
        """Human-navigable directory. Never the sole guarantee of uniqueness."""
        parts = [self.arm, self.decomposition, self.model]
        if self.W is not None:
            parts.append(f"W{self.W}")
        if self.station is not None:
            parts.append(f"st{self.station}")
        parts.append(f"seed{self.seed}")
        return os.path.join(*parts)

    def basename(self) -> str:
        lb = "lbNA" if self.look_back is None else f"lb{self.look_back}"
        return f"{lb}_{self.fingerprint()}"

    def label(self) -> str:
        """Readable one-line identifier for logs and tables."""
        bits = [self.arm, self.decomposition, self.model]
        if self.W is not None:
            bits.append(f"W={self.W}")
        if self.look_back is not None:
            bits.append(f"lb={self.look_back}")
        bits.append(f"seed={self.seed}")
        if self.station is not None:
            bits.append(f"st={self.station}")
        return " | ".join(bits)


# =============================================================================
_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    fingerprint TEXT PRIMARY KEY,
    arm TEXT, decomposition TEXT, model TEXT,
    W INTEGER, look_back INTEGER, seed INTEGER, station INTEGER,
    horizon_set TEXT,
    path TEXT NOT NULL,
    spec TEXT NOT NULL,
    metrics TEXT,
    created REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_arm ON runs(arm, decomposition, model);
"""


class ResultStore:
    """
    Filesystem + index for run outputs.

    Layout:
        <root>/<arm>/<decomp>/<model>/W<W>/st<n>/seed<k>/
            lb<look_back>_<fingerprint>.spec.json
            lb<look_back>_<fingerprint>.metrics.json
            lb<look_back>_<fingerprint>.npz          (optional arrays)
    """

    def __init__(self, root: str, index_db: Optional[str] = None):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.index_db = index_db or os.path.join(self.root, "index.db")
        with self._conn() as c:
            c.executescript(_INDEX_SCHEMA)

    def _conn(self):
        # Shared with the job ledger: WAL where the filesystem supports it,
        # TRUNCATE journalling on Lustre/GPFS where it does not.
        from jobstate import open_sqlite
        return open_sqlite(self.index_db)

    # ---------------------------------------------------------------- paths
    def dir_for(self, spec: RunSpec) -> str:
        return os.path.join(self.root, spec.relative_dir())

    def path_for(self, spec: RunSpec, suffix: str) -> str:
        return os.path.join(self.dir_for(spec), f"{spec.basename()}.{suffix}")

    def exists(self, spec: RunSpec) -> bool:
        return os.path.exists(self.path_for(spec, "metrics.json"))

    # ----------------------------------------------------------- collisions
    def _assert_no_collision(self, spec: RunSpec) -> None:
        """
        If a spec file already sits at this path, it must describe exactly this
        run. Anything else means two different configurations mapped to the
        same location.
        """
        sp = self.path_for(spec, "spec.json")
        if not os.path.exists(sp):
            return
        with open(sp) as f:
            existing = json.load(f)
        if existing != spec.to_dict():
            diff = {k: (existing.get(k), spec.to_dict().get(k))
                    for k in set(existing) | set(spec.to_dict())
                    if existing.get(k) != spec.to_dict().get(k)}
            raise ResultCollision(
                f"path {sp} already describes a different run.\n"
                f"differing fields (stored, incoming): {diff}\n"
                f"This is a hash collision or a layout bug — do not overwrite."
            )

    # ----------------------------------------------------------------- write
    def save(self, spec: RunSpec, metrics: Dict[str, Any],
             arrays: Optional[Dict[str, np.ndarray]] = None,
             overwrite: bool = False) -> str:
        """
        Persist one run. Raises rather than clobbering.

        Files are written to temporary names and renamed, so an interrupted
        write leaves no half-written result behind.
        """
        self._assert_no_collision(spec)

        mpath = self.path_for(spec, "metrics.json")
        if os.path.exists(mpath) and not overwrite:
            raise ResultExists(
                f"result already stored for {spec.label()}\n  {mpath}\n"
                f"Pass overwrite=True only if you intend to replace it."
            )

        os.makedirs(self.dir_for(spec), exist_ok=True)

        self._atomic_json(self.path_for(spec, "spec.json"), spec.to_dict())

        payload = dict(metrics)
        payload["_spec"] = spec.to_dict()
        payload["_written"] = time.time()
        self._atomic_json(mpath, payload)

        if arrays:
            write_npz(self.path_for(spec, "npz"), **arrays)

        self._index(spec, mpath, metrics)
        return mpath

    @staticmethod
    def _atomic_json(path: str, obj: Any) -> None:
        with atomic_path(path) as tmp:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=_jsonable)

    def _index(self, spec: RunSpec, path: str, metrics: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO runs
                   (fingerprint, arm, decomposition, model, W, look_back,
                    seed, station, horizon_set, path, spec, metrics, created)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (spec.fingerprint(), spec.arm, spec.decomposition, spec.model,
                 spec.W, spec.look_back, spec.seed, spec.station,
                 spec.horizon_set, path,
                 json.dumps(spec.to_dict(), default=_jsonable),
                 json.dumps(metrics, default=_jsonable), time.time()),
            )

    # ------------------------------------------------------------------ read
    def load(self, spec: RunSpec) -> Dict[str, Any]:
        with open(self.path_for(spec, "metrics.json")) as f:
            return json.load(f)

    def load_arrays(self, spec: RunSpec) -> Dict[str, np.ndarray]:
        path = self.path_for(spec, "npz")
        if not os.path.exists(path):
            return {}
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    def query(self, **filters) -> list:
        """Rows from the index. Example: query(arm='causal', model='gru')."""
        sql = "SELECT fingerprint, path, spec, metrics FROM runs"
        clauses, params = [], []
        for k, v in filters.items():
            clauses.append(f"{k} = ?")
            params.append(v)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [{"fingerprint": r[0], "path": r[1],
                 "spec": json.loads(r[2]), "metrics": json.loads(r[3] or "{}")}
                for r in rows]

    def summary(self) -> Dict[str, int]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT arm, COUNT(*) FROM runs GROUP BY arm").fetchall()
        return dict(rows)

    def verify(self) -> list:
        """
        Cross-check the index against the filesystem. Returns a list of
        problems: missing files, or spec files that disagree with the index.
        """
        problems = []
        with self._conn() as c:
            rows = c.execute("SELECT fingerprint, path, spec FROM runs").fetchall()
        for fp, path, spec_json in rows:
            if not os.path.exists(path):
                problems.append(f"missing file for {fp}: {path}")
                continue
            sp = path.replace(".metrics.json", ".spec.json")
            if os.path.exists(sp):
                with open(sp) as f:
                    on_disk = json.load(f)
                if on_disk != json.loads(spec_json):
                    problems.append(f"spec mismatch for {fp}: {sp}")
        return problems


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# =============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the result store")
    ap.add_argument("root")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    store = ResultStore(a.root)
    print("runs per arm:", store.summary())
    if a.verify:
        problems = store.verify()
        print(f"{len(problems)} problem(s)")
        for p in problems:
            print("  ", p)
