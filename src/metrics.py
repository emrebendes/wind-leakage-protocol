# -*- coding: utf-8 -*-
"""
Evaluation metrics.
===================

All metrics are computed on the RAW series in m/s. There is no component-space
reporting anywhere in this codebase: the companion paper's reviewers showed
that a model can look strong in normalised component space (+0.44 skill) while
failing in physical units (-0.55), and the only defence is to never quote the
former.

Every table carries a persistence reference. A forecast that does not beat
`x[t+h] = x[t]` has no operational value regardless of its R-squared, and the
previous study omitted this baseline entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


# =============================================================================
@dataclass
class HorizonMetrics:
    """Metrics for one forecast horizon, in m/s."""

    horizon: int
    RMSE: float
    MAE: float
    MAPE: float
    R2: float
    DA: float
    nRMSE: float
    persistence_RMSE: Optional[float] = None
    skill: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MetricSuite:
    """
    Computes the full metric set for a multi-horizon forecast.

        y_true : (n, H)   observed wind speed, m/s
        y_pred : (n, H)   forecast, m/s
        last   : (n,)     value at the forecast origin, for DA and persistence

    Held as a class so the reference series and the epsilon policy cannot
    drift between call sites.
    """

    def __init__(self, horizons: Sequence[int], mape_floor: float = 0.5):
        # MAPE is unstable near zero and this dataset averages 1.97 m/s, so it
        # is computed only above a floor and the floor is reported.
        self.horizons = list(horizons)
        self.mape_floor = float(mape_floor)

    # ------------------------------------------------------------ primitives
    @staticmethod
    def rmse(a, b) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    @staticmethod
    def mae(a, b) -> float:
        return float(np.mean(np.abs(a - b)))

    @staticmethod
    def r2(a, b) -> float:
        ss_res = float(np.sum((a - b) ** 2))
        ss_tot = float(np.sum((a - a.mean()) ** 2))
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    def mape(self, a, b) -> float:
        mask = np.abs(a) >= self.mape_floor
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)

    @staticmethod
    def directional_accuracy(a, b, last) -> float:
        """Share of steps whose direction of change relative to the origin is right."""
        true_dir = np.sign(a - last)
        pred_dir = np.sign(b - last)
        return float(np.mean(true_dir == pred_dir) * 100)

    # -------------------------------------------------------------- per step
    def at_horizon(self, y_true: np.ndarray, y_pred: np.ndarray,
                   last: np.ndarray, h: int) -> HorizonMetrics:
        idx = h - 1
        a, b = y_true[:, idx], y_pred[:, idx]
        rmse = self.rmse(a, b)
        pers = self.rmse(a, last)
        rng = float(a.max() - a.min()) or 1.0
        return HorizonMetrics(
            horizon=h,
            RMSE=rmse,
            MAE=self.mae(a, b),
            MAPE=self.mape(a, b),
            R2=self.r2(a, b),
            DA=self.directional_accuracy(a, b, last),
            nRMSE=rmse / rng,
            persistence_RMSE=pers,
            skill=float(1.0 - rmse / pers) if pers > 0 else float("nan"),
        )

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 last: np.ndarray) -> Dict[str, dict]:
        """Metrics at every configured horizon, keyed by 'H1', 'H8', ..."""
        out = {}
        for h in self.horizons:
            if h > y_true.shape[1]:
                continue
            out[f"H{h}"] = self.at_horizon(y_true, y_pred, last, h).to_dict()
        return out

    # ------------------------------------------------------------- aggregate
    def summary(self, y_true: np.ndarray, y_pred: np.ndarray,
                last: np.ndarray) -> dict:
        per_h = self.evaluate(y_true, y_pred, last)
        rmses = [v["RMSE"] for v in per_h.values()]
        skills = [v.get("skill", np.nan) for v in per_h.values()]
        return {
            "per_horizon": per_h,
            "mean_RMSE": float(np.mean(rmses)),
            "mean_skill": float(np.nanmean(skills)),
            "beats_persistence": bool(np.nanmean(skills) > 0),
            "mape_floor": self.mape_floor,
        }

    # ------------------------------------------------------ per-station view
    def by_station(self, y_true, y_pred, last, station_ids) -> dict:
        """
        Station-level breakdown. A reviewer of the previous submission objected
        that eight stations were claimed but only a pooled result was shown.
        """
        out = {}
        station_ids = np.asarray(station_ids)
        for s in np.unique(station_ids):
            m = station_ids == s
            out[f"station_{int(s)}"] = self.summary(y_true[m], y_pred[m], last[m])
        return out


# =============================================================================
class SignificanceTest:
    """
    Paired comparison between two forecasts on the same origins.

    Every claim of the form "method A improves on method B by x%" in the paper
    must be accompanied by one of these. The previous study reported no
    significance testing at all and a reviewer raised it.
    """

    @staticmethod
    def diebold_mariano(e1: np.ndarray, e2: np.ndarray, power: int = 2) -> dict:
        """
        Diebold-Mariano test on paired forecast errors.

        Returns the statistic and a normal-approximation p-value. Positive DM
        means model 1 has the larger loss, i.e. model 2 is better.
        """
        d = np.abs(e1) ** power - np.abs(e2) ** power
        n = len(d)
        mean_d = float(d.mean())
        var_d = float(d.var(ddof=1))
        if var_d <= 0 or n < 8:
            return {"DM": float("nan"), "p_value": float("nan"), "n": n}
        dm = mean_d / np.sqrt(var_d / n)
        from math import erfc, sqrt
        p = erfc(abs(dm) / sqrt(2.0))          # two-sided normal approximation
        return {"DM": float(dm), "p_value": float(p), "n": n,
                "mean_loss_diff": mean_d}

    @staticmethod
    def paired_interval(values_a: Sequence[float], values_b: Sequence[float],
                        conf: float = 0.95) -> dict:
        """
        Paired confidence interval across replicates (station x seed).

        This is the template used in `leakage_pilot/w_justification.py`: pair
        the replicates, difference them, and report the interval. If it spans
        zero the difference is not distinguishable from noise.
        """
        a = np.asarray(values_a, dtype=float)
        b = np.asarray(values_b, dtype=float)
        if a.shape != b.shape or len(a) < 3:
            return {"mean_diff": float("nan"), "ci": [float("nan")] * 2}
        d = a - b
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        # t critical for common replicate counts; 1.96 asymptotically
        t_crit = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365,
                  10: 2.262, 12: 2.201, 16: 2.131, 20: 2.093}.get(n, 1.96)
        half = t_crit * se
        return {"mean_diff": float(d.mean()),
                "ci": [float(d.mean() - half), float(d.mean() + half)],
                "n": n,
                "significant": bool((d.mean() - half) * (d.mean() + half) > 0)}

    @staticmethod
    def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                     n_samples: int = 1000, conf: float = 0.95,
                     seed: int = 0) -> dict:
        """Bootstrap interval for RMSE, resampling forecast origins."""
        rng = np.random.RandomState(seed)
        n = len(y_true)
        stats = []
        for _ in range(n_samples):
            idx = rng.randint(0, n, n)
            stats.append(np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2)))
        lo = float(np.percentile(stats, (1 - conf) / 2 * 100))
        hi = float(np.percentile(stats, (1 + conf) / 2 * 100))
        return {"RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "ci": [lo, hi], "n_bootstrap": n_samples}


# =============================================================================
class PersistenceBaseline:
    """The reference every result is measured against."""

    @staticmethod
    def predict(last: np.ndarray, horizon: int) -> np.ndarray:
        return np.repeat(np.asarray(last)[:, None], horizon, axis=1)

    @classmethod
    def evaluate(cls, y_true: np.ndarray, last: np.ndarray,
                 suite: MetricSuite) -> dict:
        pred = cls.predict(last, y_true.shape[1])
        return suite.summary(y_true, pred, last)
