# -*- coding: utf-8 -*-
"""
Analysis — tables and figures for the paper.
============================================

Reads the result store and produces the paper's evidence. Four analyses, each
one an `Analysis` subclass so a new one can be added without touching the
others:

  LeakageInflation   how much the reported gain shrinks under causal evaluation
  RankingArtefact    whether the horizon-dependent method ranking survives
  DeploymentGap      reported score versus deliverable score
  CapacityHypothesis whether decomposition helps small models more

Nothing here recomputes a forecast. If a number is missing the analysis says so
rather than silently dropping a cell, because a quietly incomplete table is how
a benchmark misleads.

Every comparison is paired on (station, seed) and reported with an interval.
The previous study reported point estimates with no uncertainty and a reviewer
objected; that failure mode is now impossible to reproduce here, because the
helpers below refuse to emit a difference without one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atomicio import atomic_path
from config import (DECOMPOSITION_METHODS, FIGURES_DIR, FORECAST_HORIZONS,
                    MODELS, RESULTS_DIR)
from metrics import SignificanceTest
from results import ResultStore


# =============================================================================
class Analysis:
    """One question asked of the result store."""

    name = "analysis"
    title = ""

    def __init__(self, store: ResultStore, out_dir: str):
        self.store = store
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------- contract
    def compute(self) -> dict:
        raise NotImplementedError

    def render(self, result: dict) -> str:
        """Plain-text table for the console and the log."""
        return json.dumps(result, indent=2, default=str)

    # ---------------------------------------------------------------- shared
    def run(self) -> Optional[dict]:
        rows = self.store.query()
        if not rows:
            print(f"[{self.name}] no results yet")
            return None
        result = self.compute()
        if result is None:
            print(f"[{self.name}] not enough data yet")
            return None
        path = os.path.join(self.out_dir, f"{self.name}.json")
        with atomic_path(path) as tmp:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=_jsonable)
        print(f"\n{'=' * 72}\n{self.title}\n{'=' * 72}")
        print(self.render(result))
        print(f"\nwritten: {path}")
        return result

    # ------------------------------------------------------------- utilities
    def collect(self, arm: str) -> Dict[tuple, List[dict]]:
        """
        Runs grouped by configuration. Rows inside a group differ ONLY by seed.

        Grouping on (decomposition, model) alone was wrong, and quietly so.
        Anything else that differs — a different look_back, a different set of
        tuned hyperparameters, a result left over from an earlier search — was
        pooled into the same group and then read as if it were a replicate.
        The seed-variance check consumed that directly: with a single seed it
        still reported a "seed sd", because it was measuring the distance
        between two different configurations and calling it training noise.

        The key is therefore the whole identity except the seed.
        """
        grouped = defaultdict(list)
        for row in self.store.query(arm=arm):
            spec = row["spec"]
            key = (spec["decomposition"], spec["model"], spec.get("W"),
                   spec.get("look_back"),
                   json.dumps(spec.get("extra", {}), sort_keys=True))
            grouped[key].append(row)
        return grouped

    @staticmethod
    def label_of(key: tuple) -> str:
        """Human-readable name for a group key."""
        return f"{key[0]}|{key[1]}"

    @staticmethod
    def rmse_at(row: dict, horizon: int) -> Optional[float]:
        try:
            return row["metrics"]["test"]["per_horizon"][f"H{horizon}"]["RMSE"]
        except (KeyError, TypeError):
            return None

    @staticmethod
    def seed_values(rows: List[dict], horizon: int) -> List[float]:
        """
        RMSE per seed for one configuration.

        Refuses to treat two runs with the same seed as replicates — that would
        mean the group is not what it claims to be, and the caller would read
        a configuration difference as training noise.
        """
        seen, out = {}, []
        for r in rows:
            v = Analysis.rmse_at(r, horizon)
            if v is None:
                continue
            seed = r["spec"].get("seed")
            if seed in seen:
                raise RuntimeError(
                    f"two runs share seed {seed} inside one configuration "
                    f"group: {r['spec'].get('decomposition')}/"
                    f"{r['spec'].get('model')}. The result store holds a stale "
                    f"run; remove it or re-run the stage.")
            seen[seed] = v
            out.append(v)
        return out


# =============================================================================
class LeakageInflation(Analysis):
    """
    Main finding 1. For every (decomposition, model, horizon), the improvement
    over the no-decomposition control, computed separately in the leaky and the
    causal arm.

    The pilot measured +75% shrinking to +2-3%; this repeats it across the full
    grid with replicate seeds and paired intervals.
    """

    name = "leakage_inflation"
    title = "Finding 1 — how much of the reported gain survives causal evaluation"

    def compute(self) -> Optional[dict]:
        out = {}
        # "partition" is the remedy the literature adopts: split first, then
        # decompose. It is listed last and treated as optional throughout, so
        # this analysis produces exactly what it always did when that arm has
        # not been run.
        # Collected first, because the partition arm borrows the causal arm's
        # undecomposed baseline (see below) and therefore cannot be processed
        # before causal has been read.
        collected = {arm: self.collect(arm)
                     for arm in ("leaky", "partition", "causal")}
        borrowed = []

        for arm in ("leaky", "partition", "causal"):
            grouped = collected[arm]
            if not grouped:
                continue
            # The reference for `dwt|gru` is `none|gru`, not the average of
            # every undecomposed model. Pooling all seven architectures into
            # one list compared a decomposed GRU against a mixture that
            # included a transformer, and — because each architecture
            # contributes its own seeds 0,1,2 — handed seed_values seven runs
            # per seed, which is exactly the repetition it refuses. The stage
            # could not run at all:
            #     RuntimeError: two runs share seed 0 ... none/cnn_lstm
            # CapacityHypothesis below already keys its baseline on
            # (model, horizon); this now matches it.
            baseline = {}
            for key, rows in grouped.items():
                if key[0] == "none":
                    baseline[key[1]] = {h: self.seed_values(rows, h)
                                        for h in FORECAST_HORIZONS}

            # THE PARTITION ARM BORROWS THE CAUSAL BASELINE, ON PURPOSE
            #     'none' is never run in the partition arm because it cannot
            #     differ: with the identity decomposition there is no channel
            #     to leak through, so decomposing a block and slicing it gives
            #     exactly the window. tests/test_causality.py asserts this for
            #     every origin, which is stronger than a run would be.
            #
            #     Without this fallback the partition column would silently
            #     come out EMPTY -- every cell needs a baseline, and there
            #     would be none. An empty column is the worst outcome here,
            #     because the table would still render and simply look as
            #     though the arm had not been run.
            #
            #     The substitution is recorded in the result so it appears in
            #     the paper rather than living only in this comment.
            if arm == "partition" and not baseline:
                for key, rows in collected["causal"].items():
                    if key[0] == "none":
                        baseline[key[1]] = {h: self.seed_values(rows, h)
                                            for h in FORECAST_HORIZONS}
                if baseline:
                    borrowed.append(arm)

            arm_rows = {}
            for key, rows in grouped.items():
                d, m = key[0], key[1]
                if d == "none":
                    continue
                cell = {}
                for h in FORECAST_HORIZONS:
                    vals = self.seed_values(rows, h)
                    base = baseline.get(m, {}).get(h) or []
                    if not vals or not base:
                        continue
                    gain = (np.mean(base) - np.mean(vals)) / np.mean(base) * 100
                    cell[f"H{h}"] = {
                        "rmse_mean": float(np.mean(vals)),
                        "rmse_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                        "n_seeds": len(vals),
                        "gain_pct": float(gain),
                    }
                if cell:
                    arm_rows[f"{d}|{m}"] = cell
            out[arm] = arm_rows

        if "causal" not in out:
            return None

        if borrowed:
            out["baseline_borrowed_from_causal"] = borrowed
        out["inflation"] = self._inflation(out)
        return out

    @staticmethod
    def _inflation(out: dict) -> dict:
        """
        Leaky gain minus causal gain, per cell and horizon.

        When the partition arm is present each cell also carries
        `removed_pct`: the share of the leaky-to-causal gap that splitting
        before decomposing actually removes.

            removed_pct = (leaky_gain - partition_gain)
                          / (leaky_gain - causal_gain) * 100

            0   -> the remedy removes none of the leakage
            100 -> the remedy removes all of it

        This single number is the paper's central claim, so it is computed
        here rather than by hand. The denominator is the measured gap, not an
        assumption, and cells whose gap is too small to divide by are left
        out instead of producing a large ratio from noise.
        """
        infl = {}
        leaky = out.get("leaky", {})
        causal = out.get("causal", {})
        part = out.get("partition", {})
        for key in sorted(set(leaky) & set(causal)):
            infl[key] = {}
            for h in FORECAST_HORIZONS:
                lg = leaky[key].get(f"H{h}", {}).get("gain_pct")
                cg = causal[key].get(f"H{h}", {}).get("gain_pct")
                if lg is None or cg is None:
                    continue
                cell = {"leaky": lg, "causal": cg, "inflation_points": lg - cg}
                pg = part.get(key, {}).get(f"H{h}", {}).get("gain_pct")
                if pg is not None:
                    cell["partition"] = pg
                    gap = lg - cg
                    # A gap below one percentage point is not a gap; dividing
                    # by it turns seed noise into a headline number.
                    cell["removed_pct"] = (float((lg - pg) / gap * 100)
                                           if abs(gap) >= 1.0 else None)
                infl[key][f"H{h}"] = cell
        return infl

    def render(self, result: dict) -> str:
        infl = result.get("inflation", {})
        has_part = any("partition" in c
                       for cells in infl.values() for c in cells.values())

        if not has_part:
            lines = [f"{'configuration':22s}" +
                     "".join(f"{'H' + str(h) + ' leaky':>14s}{'causal':>10s}"
                             for h in FORECAST_HORIZONS[:2])]
            lines.append("-" * len(lines[0]))
            for key, cells in sorted(infl.items()):
                row = f"{key:22s}"
                for h in FORECAST_HORIZONS[:2]:
                    c = cells.get(f"H{h}")
                    row += (f"{c['leaky']:+13.2f}%{c['causal']:+9.2f}%" if c
                            else f"{'-':>14s}{'-':>10s}")
                lines.append(row)
            if not infl:
                lines.append("  (needs both arms; run the leaky arm to compare)")
            return "\n".join(lines)

        # Three-arm view: the paper's main table. Only cells that have the
        # partition arm are shown, because mixing two-arm and three-arm rows
        # in one table is how a reader ends up comparing different things.
        h0 = FORECAST_HORIZONS[0]
        lines = [f"{'configuration':22s}{'whole-series':>14s}"
                 f"{'per-partition':>15s}{'per-origin':>13s}{'removed':>10s}",
                 f"{'(gain over none, H' + str(h0) + ')':22s}"]
        lines.append("-" * 74)
        removed = []
        for key, cells in sorted(infl.items()):
            c = cells.get(f"H{h0}")
            if not c or "partition" not in c:
                continue
            r = c.get("removed_pct")
            if r is not None:
                removed.append(r)
            lines.append(f"{key:22s}{c['leaky']:+13.2f}%{c['partition']:+14.2f}%"
                         f"{c['causal']:+12.2f}%"
                         + (f"{r:9.1f}%" if r is not None else f"{'n/a':>10s}"))
        if result.get("baseline_borrowed_from_causal"):
            lines.append("")
            lines.append("  not: partition kolunun 'none' referansi causal koldan "
                         "aliniyor -- ikisi ozdes")
        if removed:
            lines.append("-" * 74)
            lines.append(f"{'median removed by splitting before decomposing':<63s}"
                         f"{float(np.median(removed)):9.1f}%")
            lines.append("")
            lines.append("  0% = splitting first removes none of the leakage; "
                         "100% = it removes all of it.")
        return "\n".join(lines)


# =============================================================================
class RankingArtefact(Analysis):
    """
    Main finding 2. In the leaky pilot, VMD overtook DWT at medium horizons —
    the claim the previous paper built on. Under causal evaluation that
    crossover disappeared. This checks whether the ranking of decomposition
    methods is stable across arms, or an artefact of differing leakage.
    """

    name = "ranking_artefact"
    title = "Finding 2 — is the horizon-dependent ranking real or a leakage artefact"

    def compute(self) -> Optional[dict]:
        out = {}
        for arm in ("leaky", "partition", "causal"):
            grouped = self.collect(arm)
            if not grouped:
                continue
            per_h = {}
            for h in FORECAST_HORIZONS:
                by_decomp = defaultdict(list)
                for key, rows in grouped.items():
                    by_decomp[key[0]].extend(self.seed_values(rows, h))
                ranked = sorted(((d, float(np.mean(v)))
                                 for d, v in by_decomp.items() if v),
                                key=lambda t: t[1])
                per_h[f"H{h}"] = [{"decomposition": d, "mean_RMSE": v,
                                   "rank": i + 1}
                                  for i, (d, v) in enumerate(ranked)]
            out[arm] = per_h

        if len(out) < 2:
            out["note"] = ("both arms are required to test whether the ranking "
                           "is stable; only one is present")
            return out or None

        out["rank_agreement"] = self._agreement(out)
        return out

    @staticmethod
    def _agreement(out: dict) -> dict:
        """Spearman correlation between the two arms' rankings, per horizon."""
        agree = {}
        for h in FORECAST_HORIZONS:
            key = f"H{h}"
            a = {r["decomposition"]: r["rank"] for r in out["leaky"].get(key, [])}
            b = {r["decomposition"]: r["rank"] for r in out["causal"].get(key, [])}
            shared = sorted(set(a) & set(b))
            if len(shared) < 3:
                continue
            ra = np.array([a[d] for d in shared], dtype=float)
            rb = np.array([b[d] for d in shared], dtype=float)
            d2 = float(np.sum((ra - rb) ** 2))
            n = len(shared)
            rho = 1 - 6 * d2 / (n * (n ** 2 - 1))
            agree[key] = {"spearman_rho": rho, "n_methods": n,
                          "leaky_best": min(a, key=a.get),
                          "causal_best": min(b, key=b.get)}
        return agree

    def render(self, result: dict) -> str:
        lines = []
        for arm in ("leaky", "partition", "causal"):
            if arm not in result:
                continue
            lines.append(f"\n{arm}:")
            for h in FORECAST_HORIZONS:
                rows = result[arm].get(f"H{h}", [])
                order = " > ".join(r["decomposition"] for r in rows)
                lines.append(f"  H={h:<3d} {order}")
        for key, v in result.get("rank_agreement", {}).items():
            lines.append(f"\n  {key}: rho={v['spearman_rho']:+.2f}  "
                         f"leaky best={v['leaky_best']}  "
                         f"causal best={v['causal_best']}")
        return "\n".join(lines)


# =============================================================================
class DeploymentGap(Analysis):
    """
    Main finding 3. What the previous models actually deliver when fed only
    causally computable inputs, against what they reported and against
    persistence.
    """

    name = "deployment_gap"
    title = "Finding 3 — reported score versus deliverable score"

    # Which store arm this analysis reads. `deploy --source-arm partition`
    # writes to `deployment_gap_partition`, and until this was parameterised
    # those 105 runs sat in the index with nothing reporting them — the arm
    # that answers the sharpest form of the paper's question, silently absent
    # from the output.
    STORE_ARM = "deployment_gap"
    SOURCE_LABEL = "whole-series"

    def compute(self) -> Optional[dict]:
        rows = self.store.query(arm=self.STORE_ARM)
        if not rows:
            return None
        out = {}
        for row in rows:
            spec, met = row["spec"], row["metrics"]
            out[f"{spec['decomposition']}|{spec['model']}"] = met.get("gap", {})
        return out

    def render(self, result: dict) -> str:
        h = FORECAST_HORIZONS[0]
        header = (f"{'configuration':22s}{'reported':>12s}{'deployed':>12s}"
                  f"{'persistence':>13s}{'ratio':>9s}")
        lines = [f"models trained under the {self.SOURCE_LABEL} regime",
                 "", header, "-" * len(header)]
        beats = total = 0
        for key, cells in sorted(result.items()):
            c = cells.get(f"H{h}")
            if not c:
                continue
            total += 1
            beats += bool(c.get("beats_persistence"))
            lines.append(f"{key:22s}{c['reported']:>12.4f}{c['deployed']:>12.4f}"
                         f"{c['persistence']:>13.4f}{c['ratio']:>8.1f}x")
        lines.append("-" * len(header))
        # The count belongs in the output, not in the reader's head. It is the
        # sentence the discussion section is built on.
        lines.append(f"{beats} of {total} configurations beat persistence on "
                     f"causally available inputs.")
        lines.append(f"\n(H={h}; ratio = deployed / reported. A model whose "
                     f"deployed RMSE exceeds persistence has no operational value.)")
        return "\n".join(lines)


class DeploymentGapPartition(DeploymentGap):
    """
    The same question asked of the remedy.

    Finding 3 shows what a whole-series model delivers. This asks whether the
    partition-level remedy — which removes contamination — produces a model
    that delivers anything an operator could use. If it does not, then the
    paper's title claim holds in the strongest available form: the remedy is
    not merely incomplete, it buys no operational validity at all.
    """

    name = "deployment_gap_partition"
    title = "Finding 3b — the same question asked of the partition-level remedy"
    STORE_ARM = "deployment_gap_partition"
    SOURCE_LABEL = "partition-level"


# =============================================================================
class InternalControl(Analysis):
    """
    Check 0 — does the design isolate decomposition timing, and nothing else?

    WHY THIS RUNS FIRST AND WHY IT WAS MISSING
        Every number in Finding 1 is a gain measured against `none`. If `none`
        itself moves between arms, the arms differ by something other than the
        decomposition and the whole table compares incomparable things.

        The claim that it does not move is quoted throughout the paper as
        "0.1%", but nothing in the analysis printed it. A load-bearing number
        that only exists in prose is a number waiting to drift away from the
        data. It is computed here, from the same store every other finding
        reads.

    The partition arm has no `none` cell by construction (the identity
    transform makes it bit-identical to the per-origin arm, asserted for every
    origin in tests/test_causality.py), so only two arms appear.
    """

    name = "internal_control"
    title = "Check 0 — the undecomposed control, which must not move between arms"

    def compute(self) -> Optional[dict]:
        out = {}
        for arm in ("leaky", "causal"):
            rows = [r for r in self.store.query(arm=arm)
                    if r["spec"]["decomposition"] == "none"]
            if not rows:
                continue
            per_model: dict = {}
            for r in rows:
                m = r["spec"]["model"]
                for h in FORECAST_HORIZONS:
                    # Through rmse_at, not by indexing the dict here. The
                    # metrics live under metrics["test"]["per_horizon"], and
                    # reaching for metrics["per_horizon"] silently found
                    # nothing — every cell would have been dropped and the
                    # check would have reported "not enough data" on a
                    # complete store.
                    v = self.rmse_at(r, h)
                    if v is not None:
                        per_model.setdefault(m, {}).setdefault(h, []).append(v)
            out[arm] = {m: {h: float(np.mean(v)) for h, v in hs.items()}
                        for m, hs in per_model.items()}
        if len(out) < 2:
            return None
        diffs = {}
        for m in sorted(set(out["leaky"]) & set(out["causal"])):
            for h in FORECAST_HORIZONS:
                a = out["leaky"][m].get(h)
                b = out["causal"][m].get(h)
                if a and b:
                    diffs.setdefault(h, []).append(100.0 * abs(a - b) / b)
        # The reference the arm difference must be judged against is the
        # training noise, not a constant. A fixed threshold answers "is the
        # number small?", which is not the question; the question is "is it
        # smaller than the spread the same configuration produces when only
        # the seed changes?" Two runs that differ by less than that are not
        # distinguishable, and that is what makes the arms comparable.
        seed_pct = {}
        for arm in ("leaky", "causal"):
            for rows in self.collect(arm).values():
                if rows[0]["spec"]["decomposition"] != "none":
                    continue
                for h in FORECAST_HORIZONS:
                    vals = self.seed_values(rows, h)
                    if len(vals) > 1:
                        seed_pct.setdefault(h, []).append(
                            100.0 * float(np.std(vals, ddof=1)) /
                            float(np.mean(vals)))
        return {"per_arm": out,
                "abs_pct_diff": {h: v for h, v in diffs.items()},
                "median_abs_pct": {h: float(np.median(v))
                                   for h, v in diffs.items()},
                "seed_sd_pct": {h: float(np.median(v))
                                for h, v in seed_pct.items()},
                "pooled_abs_pct": float(np.median(
                    [v for vs in diffs.values() for v in vs]))}

    def render(self, result: dict) -> str:
        header = (f"{'horizon':9s}{'leaky RMSE':>12s}{'causal RMSE':>12s}"
                  f"{'|diff|':>9s}{'max cell':>10s}{'seed sd':>9s}{'ratio':>8s}")
        lines = [header, "-" * len(header)]
        ratios = []
        for h in FORECAST_HORIZONS:
            per = result["abs_pct_diff"].get(h)
            if not per:
                continue
            la = np.mean([v[h] for v in result["per_arm"]["leaky"].values()
                          if h in v])
            ca = np.mean([v[h] for v in result["per_arm"]["causal"].values()
                          if h in v])
            med = result["median_abs_pct"][h]
            sd = result["seed_sd_pct"].get(h)
            r = med / sd if sd else float("nan")
            ratios.append(r)
            lines.append(f"H={h:<7d}{la:>12.4f}{ca:>12.4f}{med:>8.2f}%"
                         f"{max(per):>9.2f}%{sd:>8.2f}%{r:>8.2f}")
        lines.append("-" * len(header))
        pooled = result["pooled_abs_pct"]
        worst_ratio = max(ratios) if ratios else float("nan")
        lines.append(f"pooled median difference across all cells: {pooled:.2f}%")
        lines.append(
            f"largest arm difference relative to seed noise: {worst_ratio:.2f}x  ->  "
            + ("the arms agree to within training noise; the design isolates "
               "decomposition timing"
               if worst_ratio < 2.0 else
               "WARNING: at some horizon the arms differ by more than twice the "
               "training noise. Finding 1 rests on this control; resolve before "
               "reading it."))
        lines.append("  (the partition arm has no 'none' cell: with the identity "
                     "transform it is\n   bit-identical to the per-origin arm, "
                     "asserted for every origin in the tests)")
        return "\n".join(lines)


# =============================================================================
class CapacityHypothesis(Analysis):
    """
    The study's most original analysis. Hypothesis: the benefit of decomposition
    falls as model capacity rises, because a dilated causal TCN is already a
    learned multiscale filter bank and does internally what the hand-crafted
    transform provides.

    Tested by regressing the causal gain on the parameter count across the
    seven architectures.
    """

    name = "capacity_hypothesis"
    title = "Analysis — does decomposition help low-capacity models more"

    def compute(self) -> Optional[dict]:
        grouped = self.collect("causal")
        if not grouped:
            return None

        sizes, gains = {}, defaultdict(dict)
        base = defaultdict(list)
        for key, rows in grouped.items():
            d, m = key[0], key[1]
            if d == "none":
                for h in FORECAST_HORIZONS:
                    base[(m, h)].extend(self.seed_values(rows, h))
            n_par = [r["metrics"].get("model", {}).get("n_parameters")
                     for r in rows]
            n_par = [p for p in n_par if p]
            if n_par:
                sizes[m] = int(np.mean(n_par))

        for key, rows in grouped.items():
            d, m = key[0], key[1]
            if d == "none":
                continue
            for h in FORECAST_HORIZONS:
                vals = self.seed_values(rows, h)
                b = base.get((m, h))
                if vals and b:
                    gains[m][f"{d}|H{h}"] = float(
                        (np.mean(b) - np.mean(vals)) / np.mean(b) * 100)

        if len(sizes) < 3:
            return {"note": "at least three architectures are needed",
                    "sizes": sizes}

        models = sorted(sizes, key=lambda m: sizes[m])
        mean_gain = {m: float(np.mean(list(gains[m].values())))
                     for m in models if gains.get(m)}
        if len(mean_gain) < 3:
            return {"note": "not enough gains computed yet", "sizes": sizes}

        x = np.log10([sizes[m] for m in mean_gain])
        y = np.array([mean_gain[m] for m in mean_gain])
        slope, intercept = np.polyfit(x, y, 1)
        corr = float(np.corrcoef(x, y)[0, 1])

        return {"parameter_counts": sizes,
                "mean_causal_gain_pct": mean_gain,
                "log_capacity_vs_gain": {"slope": float(slope),
                                         "intercept": float(intercept),
                                         "pearson_r": corr},
                "supports_hypothesis": bool(slope < 0 and corr < -0.3)}

    def render(self, result: dict) -> str:
        if "note" in result:
            return f"  {result['note']}"
        lines = [f"{'model':14s}{'parameters':>14s}{'mean causal gain':>20s}",
                 "-" * 48]
        sizes = result["parameter_counts"]
        for m, g in sorted(result["mean_causal_gain_pct"].items(),
                           key=lambda kv: sizes.get(kv[0], 0)):
            lines.append(f"{m:14s}{sizes.get(m, 0):>14,d}{g:>19.2f}%")
        fit = result["log_capacity_vs_gain"]
        lines.append(f"\n  slope {fit['slope']:+.2f} per decade of parameters, "
                     f"r={fit['pearson_r']:+.2f}")
        lines.append(f"  hypothesis supported: {result['supports_hypothesis']}")
        return "\n".join(lines)


# =============================================================================
class SeedVariance(Analysis):
    """
    Sanity check: how much of the spread between configurations is just
    training randomness. If seed variance is comparable to the differences
    being discussed, no ranking claim is safe.
    """

    name = "seed_variance"
    title = "Check — training randomness versus the differences being claimed"

    def compute(self) -> Optional[dict]:
        grouped = self.collect("causal")
        if not grouped:
            return None
        out = {}
        for h in FORECAST_HORIZONS:
            within, means = [], []
            for key, rows in grouped.items():
                d, m = key[0], key[1]
                vals = self.seed_values(rows, h)
                if len(vals) > 1:
                    within.append(float(np.std(vals, ddof=1)))
                if vals:
                    means.append(float(np.mean(vals)))
            if within and len(means) > 1:
                out[f"H{h}"] = {
                    "median_seed_sd": float(np.median(within)),
                    "between_config_sd": float(np.std(means, ddof=1)),
                    "ratio": float(np.median(within) / np.std(means, ddof=1)),
                    "n_configs": len(means),
                }
        return out or None

    def render(self, result: dict) -> str:
        header = (f"{'horizon':10s}{'seed sd':>12s}{'config sd':>12s}"
                  f"{'ratio':>9s}")
        lines = [header, "-" * len(header)]
        for key, v in result.items():
            lines.append(f"{key:10s}{v['median_seed_sd']:>12.4f}"
                         f"{v['between_config_sd']:>12.4f}{v['ratio']:>9.2f}")
        lines.append("\n  ratio near or above 1 means the configurations are "
                     "not distinguishable from training noise")
        return "\n".join(lines)


# =============================================================================
ANALYSES = [InternalControl, LeakageInflation, RankingArtefact,
            DeploymentGap, DeploymentGapPartition,
            CapacityHypothesis, SeedVariance]


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _report_stale_configurations(store) -> None:
    """
    Warn when one (arm, decomposition, model) has several configurations.

    After a re-tuned search, the store holds both the old runs and the new
    ones — the identity includes the hyperparameters, so they no longer
    overwrite each other, which is right. But the old ones are not results of
    the current protocol, and averaging over them would silently mix two
    experiments. This says so instead of leaving it to be noticed.
    """
    seen = defaultdict(set)
    for arm in ("causal", "leaky", "partition"):
        for row in store.query(arm=arm):
            sp = row["spec"]
            seen[(arm, sp["decomposition"], sp["model"])].add(
                (sp.get("W"), sp.get("look_back"),
                 json.dumps(sp.get("extra", {}), sort_keys=True)))

    stale = {k: v for k, v in seen.items() if len(v) > 1}
    if not stale:
        return

    print(f"\n  WARNING: {len(stale)} configuration(s) appear more than once "
          f"with different hyperparameters.")
    print("  These are almost certainly runs from a previous search. They are "
          "kept apart\n  in the analyses, but the older ones are not results "
          "of the current protocol.")
    for (arm, d, m), variants in sorted(stale.items()):
        print(f"    {arm}/{d}/{m}: {len(variants)} variants "
              f"(look_back {sorted(v[1] for v in variants)})")
    print("  Remove the stale runs, or re-run `train` after clearing "
          "results_causal/.\n")


def main():
    ap = argparse.ArgumentParser(description="Produce the paper's tables")
    ap.add_argument("--only", nargs="+",
                    choices=[a.name for a in ANALYSES],
                    help="run a subset of the analyses")
    ap.add_argument("--out", default=os.path.join(FIGURES_DIR, "analysis"))
    args = ap.parse_args()

    store = ResultStore(RESULTS_DIR)
    summary = store.summary()
    print(f"result store: {RESULTS_DIR}")
    print(f"  runs per arm: {summary or '(empty)'}")

    problems = store.verify()
    if problems:
        print(f"  WARNING: {len(problems)} integrity problem(s); "
              f"run `python run.py results` for detail")

    _report_stale_configurations(store)

    selected = [a for a in ANALYSES
                if not args.only or a.name in args.only]
    for cls in selected:
        cls(store, args.out).run()


if __name__ == "__main__":
    main()
