#!/usr/bin/env python
"""
Rebuild results_causal/index.db from the result files on disk.

WHY THIS IS NEEDED
    `analyze` does not read the result files. It reads `ResultStore.query()`,
    which reads the index — a SQLite table written by `save()` at the moment a
    run finishes. Copying result files from another machine therefore puts
    them on disk without putting them in the index, and every one of them is
    invisible to the analysis.

    The index also stores an ABSOLUTE path per run, so copying index.db itself
    across machines is worse than useless: the rows would point at
    /arf/scratch/<user>/... on a laptop.

    Every run's full identity lives in its own metrics.json (`_spec`), so the
    index is derivable from the files. This walks them and rebuilds it.

    Run it after merging the two arms, before `run.py analyze`.

        python reindex.py                # rebuild and report
        python reindex.py --kontrol      # report only, change nothing
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src"))

from config import RESULTS_DIR                              # noqa: E402
from results import ResultStore, RunSpec                    # noqa: E402


def specs_on_disk(root: str):
    """(RunSpec, metrics, path) for every metrics.json under root."""
    pattern = os.path.join(root, "**", "*.metrics.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ATLANDI {path}: {type(exc).__name__}")
            continue
        spec_d = payload.get("_spec")
        if not spec_d:
            print(f"  ATLANDI {path}: _spec yok")
            continue
        # Everything except the two bookkeeping keys is the metrics dict that
        # save() indexed, so the rebuilt row matches the original byte for
        # byte rather than being a near-miss.
        metrics = {k: v for k, v in payload.items()
                   if k not in ("_spec", "_written")}
        try:
            spec = RunSpec(**spec_d)
        except TypeError as exc:
            print(f"  ATLANDI {path}: spec cozulemedi ({exc})")
            continue
        yield spec, metrics, path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RESULTS_DIR)
    ap.add_argument("--kontrol", action="store_true",
                    help="sadece raporla, indeksi degistirme")
    args = ap.parse_args()

    store = ResultStore(args.root)
    before = store.summary()

    found, by_arm, mismatched = 0, {}, 0
    for spec, metrics, path in specs_on_disk(args.root):
        found += 1
        by_arm[spec.arm] = by_arm.get(spec.arm, 0) + 1
        # The fingerprint is a hash of the spec, so a file whose name does not
        # carry its own fingerprint was written by different code. Worth
        # knowing about; not worth refusing over.
        if spec.fingerprint() not in os.path.basename(path):
            mismatched += 1
        if not args.kontrol:
            store._index(spec, os.path.abspath(path), metrics)

    print(f"\ndiskte bulunan  : {found} sonuc")
    for arm, n in sorted(by_arm.items()):
        print(f"   {arm:16s} {n}")
    if mismatched:
        print(f"   {mismatched} dosyanin adi kendi parmak iziyle uyusmuyor "
              f"(baska bir surumle yazilmis olabilir)")

    # Rows whose file is gone must be dropped, not merely not-re-added.
    # _index() is INSERT OR REPLACE, so a rebuild adds and updates but never
    # deletes: after removing a stale duplicate from disk the index would
    # still serve it, and analyze — which reads the index, not the files —
    # would keep raising on the same collision.
    silinen = 0
    if not args.kontrol:
        with store._conn() as c:
            rows = c.execute("SELECT fingerprint, path FROM runs").fetchall()
            yok = [fp for fp, pth in rows if not os.path.exists(pth)]
            for fp in yok:
                c.execute("DELETE FROM runs WHERE fingerprint = ?", (fp,))
            silinen = len(yok)

    print(f"\nindeks (once)   : {before or 'bos'}")
    if silinen:
        print(f"indekste dosyasi olmayan {silinen} satir silindi")
    if args.kontrol:
        print("indeks degistirilmedi (--kontrol).")
        eksik = found - sum(before.values())
        if eksik > 0:
            print(f"{eksik} sonuc indekste yok — --kontrol olmadan calistir.")
        return 0
    print(f"indeks (sonra)  : {store.summary()}")

    problems = store.verify()
    if problems:
        print(f"\n{len(problems)} tutarsizlik:")
        for p in problems[:10]:
            print(f"   {p}")
    else:
        print("\nindeks ile diskteki dosyalar tutarli.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
