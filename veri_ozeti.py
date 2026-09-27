#!/usr/bin/env python
"""
Table 1 of the paper, computed rather than remembered.

WHY THIS EXISTS
    The skeleton carries descriptive statistics — "mean 1.97 m/s, sd 1.45,
    range 0-10.2" — that nobody can now trace to a computation. They may well
    be right. They may also predate `SERIES_END`, in which case they include
    the 426-hour constant tail and understate the spread. A number in a paper
    whose provenance is a previous draft is a number waiting to be wrong.

    So this recomputes them from the file the pipeline actually reads, through
    `config.load_series()`, which applies the truncation in the one place every
    stage shares.

WHAT IT REPORTS
    Per station and pooled: n, mean, sd, min, median, max, the share of exact
    zeros, and the longest constant run.

    The last two columns are the ones worth reading. The zero share separates
    genuine calm from a dead sensor only when read next to the run length: a
    station can be 30% zeros because the site is sheltered, or because the
    anemometer stopped. 1687 consecutive identical readings is the second.

    `--kesmesiz` reports the untruncated record alongside, which is how the
    426-hour tail was found and is worth having in the same table when
    defending the decision to cut it.

    python veri_ozeti.py
    python veri_ozeti.py --kesmesiz
    python veri_ozeti.py --markdown        # paste straight into the draft
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from config import DATA_FILE, SERIES_END, load_series   # noqa: E402


def longest_constant_run(x: np.ndarray) -> tuple:
    """
    (length, start index) of the longest stretch of identical values.

    Reported rather than the count of zeros because a stuck sensor is defined
    by repetition, not by the value it repeats: station 1 and 3 are stuck at
    zero here, but a sensor frozen at 2.4 m/s would be just as broken and a
    zero count would not see it.
    """
    if len(x) < 2:
        return len(x), 0
    change = np.flatnonzero(np.diff(x)) + 1
    edges = np.concatenate(([0], change, [len(x)]))
    runs = np.diff(edges)
    i = int(np.argmax(runs))
    return int(runs[i]), int(edges[i])


def rows_for(series: np.ndarray) -> list:
    out = []
    for s in range(series.shape[0]):
        x = np.asarray(series[s], dtype=np.float64)
        run, at = longest_constant_run(x)
        out.append({
            "ist": f"st{s}", "n": len(x),
            "ort": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),
            "min": float(np.min(x)), "med": float(np.median(x)),
            "maks": float(np.max(x)),
            "sifir": 100.0 * float(np.mean(x == 0.0)),
            "kosu": run, "kosu_bas": at,
        })
    x = np.asarray(series, dtype=np.float64).ravel()
    out.append({
        "ist": "TOPLAM", "n": int(x.size),
        "ort": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),
        "min": float(np.min(x)), "med": float(np.median(x)),
        "maks": float(np.max(x)),
        "sifir": 100.0 * float(np.mean(x == 0.0)),
        "kosu": max(r["kosu"] for r in out), "kosu_bas": -1,
    })
    return out


def show(rows: list, title: str, markdown: bool) -> None:
    cols = [("ist", "istasyon", "{:s}"), ("n", "n", "{:d}"),
            ("ort", "ortalama", "{:.3f}"), ("sd", "sd", "{:.3f}"),
            ("min", "min", "{:.2f}"), ("med", "medyan", "{:.2f}"),
            ("maks", "maks", "{:.2f}"), ("sifir", "sifir %", "{:.1f}"),
            ("kosu", "en uzun sabit kosu", "{:d}")]
    print(f"\n{title}")
    if markdown:
        print("| " + " | ".join(c[1] for c in cols) + " |")
        print("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            print("| " + " | ".join(f[2].format(r[f[0]]) for f in cols) + " |")
        return
    w = [max(len(c[1]), 9) for c in cols]
    print("  " + "  ".join(f"{c[1]:>{wi}s}" for c, wi in zip(cols, w)))
    print("  " + "  ".join("-" * wi for wi in w))
    for r in rows:
        print("  " + "  ".join(f"{f[2].format(r[f[0]]):>{wi}s}"
                               for f, wi in zip(cols, w)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kesmesiz", action="store_true",
                    help="kirpilmamis kaydi da raporla (426 saatlik kuyruk "
                         "dahil) — kesme kararinin gerekcesi")
    ap.add_argument("--markdown", action="store_true",
                    help="taslağa yapistirilabilir tablo")
    args = ap.parse_args()

    series = load_series()
    print(f"\nveri : {DATA_FILE}")
    print(f"kesme: SERIES_END = {SERIES_END}  ->  "
          f"{series.shape[0]} istasyon x {series.shape[1]} saat = "
          f"{series.size} gozlem")
    show(rows_for(series), "KESME SONRASI  (makaleye giren)", args.markdown)

    if args.kesmesiz:
        raw = np.asarray(np.load(DATA_FILE, allow_pickle=True),
                         dtype=np.float64)
        print(f"\nkirpilmamis: {raw.shape[0]} x {raw.shape[1]} = {raw.size}")
        show(rows_for(raw), "KESME ONCESI  (yalniz karsilastirma icin)",
             args.markdown)
        print("\n  Kuyruk sabit oldugu icin kesme oncesi sd DAHA KUCUK cikar.")
        print("  Taslakta bir sd varsa ve bu sutunla eslesiyorsa, o sayi")
        print("  kirpilmamis kayittan gelmis demektir — degistirilmeli.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
