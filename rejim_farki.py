#!/usr/bin/env python
"""
How different are the leaky and partition arms, actually?

WHY THIS EXISTS
    The partition arm costs 35 Optuna studies and 105 training runs. Before
    spending them it is worth knowing, per method, whether the two arms even
    produce different inputs — because if they do not, the training run can
    only reproduce numbers we already have.

    There is reason to think they often will not. A decomposition with finite
    support reads only a neighbourhood of each position, so a coefficient far
    from a block boundary cannot tell which block it is in. The causality
    suite already asserts this for dwt at origin 30000:

        test_finite_support_makes_partition_coincide_with_leaky

    A global decomposition (vmd, and the EMD family, which sift over the whole
    signal) has no such limit and should differ everywhere.

    This measures which is which, on the real record, with the real tuned
    parameters, at the origins that actually produce the reported score.

WHAT IT REPORTS, per method
    same        fraction of sampled TEST origins where the two arms give
                bit-identical features
    max rel     largest relative difference where they do differ
    boundary    how far into the test block the differences reach

    same = 100%  ->  the partition arm cannot change the reported score.
                     Do not train it; assert the identity instead and say so.
    same = 0%    ->  genuinely different inputs; the training run is informative.

COST
    Two decompositions per method per station — one full series, one test
    block — then the sampled origins are free, because both builders memoise.
    Cheap for dwt and vmd; the EMD family sifts over 50718 samples and takes
    minutes. Start with --methods dwt vmd.

    python rejim_farki.py --methods dwt vmd
    python rejim_farki.py --methods dwt vmd emd eemd ceemdan --stations 0 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src"))

from config import (DECOMPOSITION_METHODS, W_DEFAULT,               # noqa: E402
                    load_series, SERIES_END)
from causal_features import (get_builder, split_bounds,             # noqa: E402
                             valid_origins)
from decompositions import get_decomposer                           # noqa: E402


def tuned_params(method: str, W: int) -> dict:
    """
    The parameters the real run uses, not the library defaults.

    Delegates to PrecomputeStage.load_params rather than re-deriving the path.
    A second copy of "where the tuned parameters live" is a copy that can go
    stale, and the support width — the thing that decides the answer here —
    depends directly on those parameters.
    """
    from precompute_causal import PrecomputeStage
    return PrecomputeStage.load_params(method, W)


def compare(series: np.ndarray, method: str, W: int, n_origins: int,
            atol: float) -> dict:
    params = tuned_params(method, W)
    leaky = get_builder("leaky", get_decomposer(method, params), W=W)
    part = get_builder("partition", get_decomposer(method, params), W=W)

    o = valid_origins(len(series), W, "test")
    step = max(1, len(o) // n_origins)
    sample = o[::step][:n_origins]

    same, diffs, first_diff_offset = 0, [], None
    test_lo = split_bounds(len(series))[1]
    for t in sample:
        a = np.asarray(leaky.at_origin(series, int(t)), dtype=np.float64)
        b = np.asarray(part.at_origin(series, int(t)), dtype=np.float64)
        if a.shape != b.shape:
            diffs.append(np.inf)
            continue
        scale = max(float(np.max(np.abs(a))), 1e-12)
        rel = float(np.max(np.abs(a - b))) / scale
        if rel <= atol:
            same += 1
        else:
            diffs.append(rel)
            # How deep into the test block do differences still appear? For a
            # finite-support method this should stop after the support width.
            off = int(t) - test_lo
            if first_diff_offset is None or off > first_diff_offset:
                first_diff_offset = off
    return {"n": len(sample), "same": same,
            "same_pct": 100.0 * same / max(1, len(sample)),
            "max_rel": max(diffs) if diffs else 0.0,
            "deepest_diff": first_diff_offset}


def alignment_probe(series: np.ndarray, method: str, W: int,
                    n_shifts: int = 40, block: int = 4096) -> None:
    """
    Is the leaky/partition difference INFORMATION, or just grid alignment?

    THE QUESTION
        At an interior test origin both blocks contain the whole neighbourhood
        the decomposition can read, so neither has information the other lacks.
        Yet the features differ. Two mechanisms could explain that and they
        mean opposite things:

        (1) ALIGNMENT. A dwt decimates from index 0 of whatever block it is
            given. Moving the block start by one sample moves the whole dyadic
            grid, and the value at a fixed timestamp changes. This is shift
            variance -- DWTDecomposition already declares
            `is_shift_invariant = False` and the pilot measured 68-106% swings
            in the fine bands. It is an artefact of where the block begins,
            not of what the block contains.

        (2) INFORMATION. The block genuinely excludes data the other block
            used, and the decomposition is global enough to have needed it.
            vmd solves one optimisation over the entire block, so this is the
            expected mechanism there.

    THE TEST
        Hold the block LENGTH fixed and slide its start one sample at a time,
        reading the feature at the same timestamp each time. Information
        content barely changes across a few dozen samples. If the difference
        is periodic with period 2^level, the mechanism is alignment. If it
        drifts smoothly and never returns near zero, it is information.

    Reported per shift: relative difference against shift 0.
    """
    params = tuned_params(method, W)
    dec = get_decomposer(method, params)
    n = len(series)
    t = int(valid_origins(n, W, "test")[len(valid_origins(n, W, "test")) // 2])
    lo0 = t - block // 2

    ref = None
    rows = []
    for d in range(n_shifts):
        lo = lo0 - d
        comps = np.asarray(dec(series[lo: lo + block]), dtype=np.float64)
        col = comps[:, t - lo]                       # same timestamp every time
        if ref is None:
            ref = col
            rows.append((d, 0.0))
            continue
        if col.shape != ref.shape:
            rows.append((d, np.inf))
            continue
        scale = max(float(np.max(np.abs(ref))), 1e-12)
        rows.append((d, float(np.max(np.abs(col - ref))) / scale))

    lvl = int(params.get("level", 0)) if isinstance(params, dict) else 0
    print(f"\n  hizalama sondasi — {method}, origin {t}, blok {block} ornek, "
          f"level={lvl or '?'}")
    print(f"  {'kayma':>6s}{'rel fark':>12s}   (blok basi 1 ornek geri kayiyor)")
    for d, r in rows:
        mark = ""
        if lvl and d and d % (2 ** lvl) == 0:
            mark = f"  <- 2^{lvl}'in kati"
        print(f"  {d:>6d}{r:>12.3e}{mark}")

    if lvl:
        aligned = [r for d, r in rows if d and d % (2 ** lvl) == 0]
        other = [r for d, r in rows if d and d % (2 ** lvl) != 0]
        if aligned and other:
            print(f"\n  2^{lvl}'in katlarinda ortalama fark : "
                  f"{float(np.mean(aligned)):.3e}")
            print(f"  digerlerinde ortalama fark       : "
                  f"{float(np.mean(other)):.3e}")
            if float(np.mean(aligned)) < float(np.mean(other)) / 10:
                print("  -> HIZALAMA. Fark bilgi degil, desimasyon izgarasinin yeri.")
            else:
                print("  -> BILGI. Fark blogun icerdigi veriden geliyor.")


def aligned_control(series: np.ndarray, method: str, W: int) -> None:
    """
    Separate the artefact from the information, exactly.

    The partition block begins `depth - 1` = 167 samples before the split
    boundary. 167 is an implementation choice, not a fact about the data, and
    for a dwt it decides where the decimation grid lands. So part of the
    measured leaky/partition difference is that choice and part of it is the
    block's contents, and the paper should not attribute the first to the
    second.

    This separates them. At an interior test origin, features are built three
    ways:

        A  whole series                    [0, n)          -- the leaky arm
        B  natural partition block         [43110-167, n)  -- the arm we run
        C  same block, start rounded DOWN to a multiple of 2^level

    C contains everything B contains plus a few extra samples at the front, so
    it holds no less information than B and no more than A. If C matches A
    exactly, the A-B difference cannot be information: it is the grid.

    For a global method there is no grid, so C should differ from A by about
    as much as B does — and the difference is real.
    """
    params = tuned_params(method, W)
    dec = get_decomposer(method, params)
    n = len(series)
    o = valid_origins(n, W, "test")
    t = int(o[len(o) // 2])
    from config import LOOK_BACK_MAX
    natural = split_bounds(n)[1] - (LOOK_BACK_MAX - 1)
    lvl = int(params.get("level", 0)) if isinstance(params, dict) else 0
    period = 2 ** lvl if lvl else 1
    aligned = (natural // period) * period

    def feat(lo):
        c = np.asarray(dec(series[lo:n]), dtype=np.float64)
        return c[:, t - lo]

    A, B, C = feat(0), feat(natural), feat(aligned)
    rel = lambda u, v: float(np.max(np.abs(u - v))) / max(float(np.max(np.abs(u))), 1e-12)

    print(f"\n  hizali kontrol — {method}, origin {t}, level={lvl or '-'}, "
          f"periyot {period}")
    print(f"    A tam seri        [0, {n})")
    print(f"    B dogal blok      [{natural}, {n})     (isinma 167)")
    print(f"    C hizali blok     [{aligned}, {n})     (basi {period}'in kati)")
    print(f"    A-B fark : {rel(A, B):.3e}   <- kostugumuz kol")
    print(f"    A-C fark : {rel(A, C):.3e}   <- izgara ayni, icerik hemen ayni")
    print(f"    B-C fark : {rel(B, C):.3e}")
    if rel(A, C) < rel(A, B) / 100:
        print("    -> A-B farkinin TAMAMI izgara hizalamasi. Bilgi farki yok.")
    else:
        print("    -> A-B farki icerikten geliyor. Gercek bilgi farki.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["dwt", "vmd"],
                    choices=[m for m in DECOMPOSITION_METHODS if m != "none"])
    ap.add_argument("--stations", type=int, nargs="+", default=[0])
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    ap.add_argument("--origins", type=int, default=40,
                    help="test origins sampled per method (evenly spaced)")
    ap.add_argument("--atol", type=float, default=1e-9,
                    help="relative tolerance below which features count as identical")
    ap.add_argument("--hizalama", action="store_true",
                    help="farkin hizalama mi bilgi mi oldugunu ayirt et")
    args = ap.parse_args()

    data = load_series()
    n = data.shape[1]
    tr, va = split_bounds(n)
    print(f"\nseri {n} saat  (SERIES_END={SERIES_END})   "
          f"train [0,{tr})  val [{tr},{va})  test [{va},{n})")
    print(f"W={args.W}   istasyon {args.stations}   "
          f"{args.origins} test orijini   tolerans {args.atol:g}\n")

    hdr = (f"{'yontem':10s}{'istasyon':>9s}{'ayni':>10s}{'en buyuk rel':>15s}"
           f"{'en derin fark':>15s}{'sure':>8s}")
    print(hdr)
    print("-" * len(hdr))

    verdict = {}
    for m in args.methods:
        for st in args.stations:
            t0 = time.time()
            try:
                r = compare(np.asarray(data[st], dtype=np.float64), m,
                            args.W, args.origins, args.atol)
            except Exception as exc:                       # noqa: BLE001
                print(f"{m:10s}{st:>9d}{'HATA':>10s}   {type(exc).__name__}: {exc}")
                continue
            deep = ("-" if r["deepest_diff"] is None
                    else f"+{r['deepest_diff']}")
            print(f"{m:10s}{st:>9d}{r['same_pct']:>9.1f}%{r['max_rel']:>15.2e}"
                  f"{deep:>15s}{time.time() - t0:>7.0f}s")
            verdict.setdefault(m, []).append(r["same_pct"])

    print()
    print("YORUM")
    for m, pcts in verdict.items():
        p = float(np.mean(pcts))
        if p >= 99.9:
            print(f"  {m:8s} ozdes ({p:.1f}%). Partition kolu bu yontem icin "
                  f"leaky ile ayni girdiyi uretiyor —")
            print(f"           egitmek yeni bir sayi vermez. Ozdeslik iddia "
                  f"edilip makalede oyle yazilmali.")
        elif p <= 0.1:
            print(f"  {m:8s} tamamen farkli ({p:.1f}%). Egitim bilgi verir; "
                  f"kos.")
        else:
            print(f"  {m:8s} kismen farkli ({p:.1f}% ozdes). Sinir tabakasi "
                  f"etkisi; kos ve farki raporla.")
    if args.hizalama:
        for m in args.methods:
            x = np.asarray(data[args.stations[0]], dtype=np.float64)
            alignment_probe(x, m, args.W)
            aligned_control(x, m, args.W)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
