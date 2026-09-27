#!/usr/bin/env python
"""
Give every study the same search budget, after the fact.

WHY THIS IS NEEDED
    Studies do not stop at exactly OPTUNA_TRIALS. Workers share a study, and
    as the other studies of a job finish, all of them pile onto whatever is
    left — `causal|none|bilstm` was observed with 26 trials in flight at once.
    The completed count crosses the target while those trials are still
    training, and every one of them lands as COMPLETE afterwards.

    Measured on the causal arm: between 50 and 93 completed trials per study
    against a target of 50. That is not a correctness bug — no result is
    wrong — but it breaks the comparison. A search given 93 draws has a better
    expected minimum than one given 50, so part of any difference between two
    decompositions would just be the number of draws each happened to get.

WHAT THIS DOES
    For every study it takes the first OPTUNA_TRIALS COMPLETE trials in
    creation order, picks the best of those, and rewrites
    decomp_params/models/<arm>/<decomposition>__<model>.json from it. Nothing
    is retrained and no trial is deleted; the surplus trials stay in the
    storage and can still be reported.

    `trial.number` is assigned when a trial is created, so the ordering is
    fixed in the database and anyone can reproduce the selection.

    Run it after the searches finish, before `run.py train`.

        python equalize_budget.py                  # report only
        python equalize_budget.py --write          # rewrite the json files
        python equalize_budget.py --arm causal --budget 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src"))

import optuna                                              # noqa: E402

from config import ALL_ARMS, ARMS, OPTUNA_TRIALS, PARAMS_DIR         # noqa: E402
from optimize import OptunaStage, _storage_urls            # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def studies(arms):
    """
    (arm, decomposition, model, study) for every study on disk, once each.

    A study name can appear in more than one storage file: the file is chosen
    by a hash of the submission scope, so resubmitting a narrower selection
    creates a second, empty copy. Yielding both would rewrite the parameter
    file twice and the last one — arbitrarily, whichever the glob read last —
    would win. Keep only the copy with the most completed trials.
    """
    best: dict = {}
    for url in _storage_urls():
        try:
            summaries = optuna.get_all_study_summaries(storage=url)
        except Exception as exc:                            # noqa: BLE001
            print(f"  {url} okunamadi: {type(exc).__name__}")
            continue
        for s in summaries:
            parts = s.study_name.split("__")
            if len(parts) != 3 or parts[0] not in arms:
                continue
            study = optuna.load_study(study_name=s.study_name, storage=url)
            n = sum(1 for t in study.get_trials(deepcopy=False)
                    if t.state == optuna.trial.TrialState.COMPLETE)
            prev = best.get(s.study_name)
            if prev is None or n > prev[0]:
                if prev is not None:
                    print(f"  UYARI: {s.study_name} birden fazla dosyada; "
                          f"{n} denemeli kopya kullaniliyor "
                          f"({prev[0]} denemeli olan atlandi)")
                best[s.study_name] = (n, (*parts, study))
            elif prev is not None:
                print(f"  UYARI: {s.study_name} birden fazla dosyada; "
                      f"{prev[0]} denemeli kopya korundu ({n} atlandi)")
    for _, (_, row) in sorted(best.items()):
        yield row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=ARMS, choices=ALL_ARMS)
    ap.add_argument("--budget", type=int, default=OPTUNA_TRIALS)
    ap.add_argument("--write", action="store_true",
                    help="rewrite the json files; without it nothing changes")
    args = ap.parse_args()

    rows, changed, short = [], 0, []
    for arm, d, m, study in sorted(studies(set(args.arm))):
        pool = OptunaStage.budget_trials(study, args.budget)
        winner = OptunaStage.best_within_budget(study, args.budget)
        total = sum(1 for t in study.get_trials(deepcopy=False)
                    if t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None)
        if winner is None:
            short.append((f"{arm}|{d}|{m}", 0))
            continue
        if len(pool) < args.budget:
            # Still running, or it failed short of the target. Selecting from
            # a partial pool would quietly give this study a smaller budget
            # than the rest, which is the problem this script exists to fix.
            short.append((f"{arm}|{d}|{m}", len(pool)))
            continue

        overall = min(t.value for t in study.get_trials(deepcopy=False)
                      if t.state == optuna.trial.TrialState.COMPLETE
                      and t.value is not None)
        rows.append((arm, d, m, total, winner.value, overall,
                     winner.number, winner.params))
        if winner.value > overall:
            changed += 1

    print(f"\n{'calisma':30s} {'tamam':>6s} {'butce':>6s} "
          f"{'ilk-N en iyi':>13s} {'tum en iyi':>11s}  {'fark':>8s}")
    print("-" * 84)
    for arm, d, m, total, wv, ov, num, _ in rows:
        diff = "" if wv == ov else f"{(wv - ov) / ov * 100:+.2f}%"
        print(f"{arm+'|'+d+'|'+m:30s} {total:6d} {args.budget:6d} "
              f"{wv:13.5f} {ov:11.5f}  {diff:>8s}")
    print("-" * 84)
    fazla = sum(r[3] for r in rows) - args.budget * len(rows)
    print(f"{len(rows)} calisma esitlendi, hedefin {fazla} deneme uzerinde "
          f"kosulmus (ortalama +{fazla / max(len(rows), 1):.1f})")
    print(f"{changed} calismada secim degisiyor: butce disi bir deneme daha "
          f"iyiydi ve artik kullanilmiyor")

    if short:
        print(f"\n{len(short)} calisma henuz {args.budget} denemeye ulasmadi "
              f"— bunlara dokunulmadi:")
        for key, n in short:
            print(f"  {key:30s} {n}/{args.budget}")

    if not args.write:
        print("\nHicbir dosya degismedi. Yazmak icin --write ekle.")
        return 0

    for arm, d, m, _, wv, _, num, params in rows:
        out_dir = os.path.join(PARAMS_DIR, "models", arm)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{d}__{m}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"arm": arm, "decomposition": d, "model": m,
                       "best_params": params, "best_value": float(wv),
                       "budget": args.budget, "trial_number": num,
                       "selection": "first %d completed trials, by trial "
                                    "number" % args.budget},
                      f, indent=2)
    print(f"\n{len(rows)} dosya yazildi: {PARAMS_DIR}/models/<arm>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
