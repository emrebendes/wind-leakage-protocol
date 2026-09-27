#!/usr/bin/env python
"""
Do the arms agree on the channel count?

WHY IT MATTERS
    K is the number of feature channels a model receives. `deploy` loads a
    model trained in one arm and evaluates it on another arm's features, so a
    disagreement means the weights do not fit and the whole stage fails —
    after the training that produced them, not before.

    It also matters for reading the results at all: if the partition arm hands
    the network six channels where the leaky arm handed five, the two rows of
    the main table describe different models and the comparison is empty.

    K is decided once per (arm, method, W) by `_global_k`, which probes
    W-sized WINDOWS rather than the block a leaky or partition builder
    actually decomposes. That is deliberate — it is what keeps the three arms
    on the same channel count — but it means the agreement is a property to
    verify rather than assume.

WHERE THE NUMBER LIVES
    Not in K.json — that file does not exist for any method. `_global_k`
    returns before writing it whenever `decomposer.expected_k()` knows the
    answer in advance, and every method now does: dwt from `level`, vmd from
    `K`, and the EMD family from `max_imfs + 1`. The EMD classes used to
    return None and let precompute measure K from the data, which is what
    K.json was for; that was changed after a stuck sensor made one station
    answer 1 while the other seven answered 5.

    The per-station `meta.json` is written for every cell regardless, and it
    is what the loader actually reads, so that is the honest source. Reading
    it per station also catches a disagreement BETWEEN stations, which would
    stop CausalDataModule from stacking them:
        RuntimeError: stations disagree on channel count

    python k_kontrol.py
    python k_kontrol.py --methods dwt vmd

"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from config import CACHE_DIR                                       # noqa: E402

ARM_ORDER = ["causal", "leaky", "partition"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=None,
                    help="default: whatever is on disk")
    args = ap.parse_args()

    pattern = os.path.join(CACHE_DIR, "*", "W*", "*", "st*", "meta.json")
    seen: dict = collections.defaultdict(lambda: collections.defaultdict(set))
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if args.methods and d.get("method") not in args.methods:
            continue
        seen[(d["method"], int(d["W"]))][d["arm"]].add(int(d["K"]))

    if not seen:
        print(f"\n{CACHE_DIR} altinda meta.json yok — precompute hic kosmamis.")
        return 1

    # Collapse each arm's per-station set. More than one value in it is itself
    # a fault, reported as e.g. "5/6" so it cannot be mistaken for agreement.
    rows: dict = collections.defaultdict(dict)
    split_arms = []
    for key, per_arm in seen.items():
        for arm, ks in per_arm.items():
            if len(ks) == 1:
                rows[key][arm] = next(iter(ks))
            else:
                rows[key][arm] = "/".join(str(k) for k in sorted(ks))
                split_arms.append((key[0], key[1], arm, sorted(ks)))

    print(f"\nonbellek: {CACHE_DIR}\n")
    hdr = f"{'yontem':10s}{'W':>6s}" + "".join(f"{a:>11s}" for a in ARM_ORDER) + "   uyum"
    print(hdr)
    print("-" * len(hdr))

    bad = []
    for (m, w), per_arm in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        present = {a: k for a, k in per_arm.items() if a in ARM_ORDER}
        distinct = set(present.values())
        # One arm on its own says nothing about agreement between arms, so a
        # single arm passes that half of the check. But a value like "1/5" is
        # a within-arm split, and a row carrying one must never read OK — that
        # was the first version's mistake: it compared the STRING against
        # itself, found one distinct value, and called a broken cell healthy.
        split_here = any(isinstance(v, str) for v in present.values())
        ok = len(distinct) <= 1 and not split_here
        cells = "".join(f"{present.get(a, '-'):>11}" for a in ARM_ORDER)
        mark = "OK" if ok else ("ISTASYONLAR" if split_here else "FARKLI")
        if not ok and not split_here:
            bad.append((m, w, dict(present)))
        print(f"{m:10s}{w:>6d}{cells}   {mark}")

    print()
    if bad:
        print(f"{len(bad)} hucrede kollar K uzerinde anlasmiyor:")
        for m, w, p in bad:
            print(f"  {m} W={w}: {p}")
        print("  -> `deploy` agirlik aktaramaz ve ana tablonun iki satiri")
        print("     farkli modelleri anlatir.\n")

    if split_arms:
        print("Bir kolun istasyonlari kendi aralarinda anlasmiyor:")
        for m, w, arm, ks in split_arms:
            print(f"  {arm}/{m} W={w}: istasyonlar {ks}")
        print("  -> CausalDataModule istasyonlari tek tensore yigiyor; bu")
        print("     haliyle `stations disagree on channel count` ile duser.\n")

    if bad or split_arms:
        print("Egitime baslamadan cozulmeli.")
        return 1

    n_arms = len({a for p in rows.values() for a in p})
    print(f"Butun hucrelerde kollar ayni K'da anlasiyor ({n_arms} kol goruldu).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
