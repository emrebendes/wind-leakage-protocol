# -*- coding: utf-8 -*-
"""
Decomposition methods — class hierarchy.
========================================

Every method is a subclass of `BaseDecomposition` and must declare, in one
place, everything the rest of the pipeline needs to know about it:

    decompose_window(window)  the transform itself  (K, len(window))
    default_params()          starting parameters
    param_space()             search space for tuning
    min_window(params)        shortest window for which the method is valid
    expected_k(params)        channel count, when it is determined a priori
    is_stochastic             whether repeated calls need seeding

Why a class and not a function
------------------------------
The abstract base enforces the contract, so a method added later cannot
silently omit its search space or its window constraint. Keeping those next to
the implementation is the point: in the previous study the decomposition
parameters and the code that used them lived in different files, and the two
drifted apart.

`min_window` in particular is a safety net that did not exist before. A level-6
DWT on a 128-sample window produces a degenerate coarsest scale; previously
nothing would have complained.

Causality
---------
`decompose_window` receives a window and nothing else. It cannot reach data
outside that window, which is what makes the pipeline causal by construction.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


# =============================================================================
# helpers
# =============================================================================
def _stable_seed(window: np.ndarray) -> int:
    """Deterministic seed derived from the window contents."""
    h = hashlib.sha1(np.ascontiguousarray(window, dtype=np.float64).tobytes())
    return int.from_bytes(h.digest()[:4], "little")


# =============================================================================
# base
# =============================================================================
class BaseDecomposition(ABC):
    """
    Contract for every decomposition used in this study.

    Subclasses implement `decompose_window`. Callers use `__call__`, which
    validates the window, delegates, and guarantees additivity.
    """

    name: str = "base"
    is_stochastic: bool = False
    is_shift_invariant: bool = False

    # Whether decomposing the whole series and slicing can differ from
    # decomposing the window — that is, whether this method is capable of
    # leaking future information at all. True for every real decomposition,
    # because each one mixes neighbouring samples. False only for the identity
    # (see NoDecomposition), where the two are the same operation.
    #
    # This is what makes 'none' the reference condition: it is the one setting
    # that cannot be inflated by decomposition leakage, so the gap between the
    # arms elsewhere is measured against a fixed point.
    can_leak: bool = True

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged = dict(self.default_params())
        merged.update(params or {})
        self.params = merged
        self.validate_params()

    # ------------------------------------------------------------- contract
    @abstractmethod
    def decompose_window(self, window: np.ndarray) -> np.ndarray:
        """Return (K, len(window)); components need not be exactly additive."""

    @staticmethod
    @abstractmethod
    def default_params() -> Dict[str, Any]:
        """Parameters used when nothing has been tuned yet."""

    @staticmethod
    @abstractmethod
    def param_space() -> Dict[str, Any]:
        """
        Search space for tuning. Conventions:
            (lo, hi)          integer or float range
            [a, b, c]         categorical choices
        """

    def min_window(self) -> int:
        """Shortest window for which this method is meaningful."""
        return 32

    def expected_k(self) -> Optional[int]:
        """Channel count when known in advance; None if data dependent."""
        return None

    def validate_params(self) -> None:
        """Raise if the parameter combination is invalid. Override as needed."""
        return None

    # ---------------------------------------------------------------- public
    def __call__(self, window: np.ndarray) -> np.ndarray:
        window = np.asarray(window, dtype=np.float64)
        need = self.min_window()
        if len(window) < need:
            raise ValueError(
                f"{self.name}: window of {len(window)} samples is shorter than "
                f"the minimum {need} required by params {self.params}. "
                f"Increase W or reduce the decomposition depth."
            )
        comps = self.decompose_window(window)
        return self._as_additive(comps, window)

    @staticmethod
    def _as_additive(comps, window: np.ndarray) -> np.ndarray:
        """
        Force the components to sum exactly to the window. Reconstruction in
        the evaluation stage depends on this.
        """
        comps = np.atleast_2d(np.asarray(comps, dtype=np.float64))
        n = len(window)

        # A degenerate window can leave EMD with nothing to extract — a long
        # flat stretch has no extrema, so sifting returns an empty array. That
        # is data, not a bug: the honest decomposition of a constant window is
        # the window itself, in one component. Without this the run died on
        # `comps[-1]` with an empty first axis.
        if comps.size == 0 or comps.shape[0] == 0:
            return window.reshape(1, n).astype(np.float64).copy()

        if comps.shape[1] != n:
            fixed = np.zeros((comps.shape[0], n))
            m = min(comps.shape[1], n)
            fixed[:, :m] = comps[:, :m]
            comps = fixed
        comps[-1] += window - comps.sum(axis=0)
        return comps

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "min_window": self.min_window(),
            "expected_k": self.expected_k(),
            "stochastic": self.is_stochastic,
            "shift_invariant": self.is_shift_invariant,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"


# =============================================================================
# control condition
# =============================================================================
class NoDecomposition(BaseDecomposition):
    """Raw signal as a single channel. The reference every method is scored against."""

    name = "none"
    is_shift_invariant = True
    can_leak = False          # identity: full-series slice == window

    def decompose_window(self, window):
        return window[None, :]

    @staticmethod
    def default_params():
        return {}

    @staticmethod
    def param_space():
        return {}

    def min_window(self):
        return 1

    def expected_k(self):
        return 1


# =============================================================================
# wavelet
# =============================================================================
class DWTDecomposition(BaseDecomposition):
    """
    Additive multiresolution decomposition. Each level is reconstructed on its
    own so components live in the time domain and sum back to the signal.

    Shift variance: moving the window by one sample changes the decimation
    grid, so component values at shared timestamps change (68-106% for the fine
    bands, measured in the pilot). Documented, reported, not fixed here.
    """

    name = "dwt"
    is_shift_invariant = False

    def decompose_window(self, window):
        import pywt
        wav = self.params["wavelet"]
        lv = int(min(self.params["level"],
                     pywt.dwt_max_level(len(window), pywt.Wavelet(wav).dec_len)))
        coeffs = pywt.wavedec(window, wav, level=lv, mode="periodization")
        out = []
        for i in range(len(coeffs)):
            sel = [np.zeros_like(c) for c in coeffs]
            sel[i] = coeffs[i]
            out.append(pywt.waverec(sel, wav, mode="periodization")[:len(window)])
        return np.array(out)

    @staticmethod
    def default_params():
        return {"wavelet": "sym5", "level": 5}

    @staticmethod
    def param_space():
        # `level` starts at 1, not 3. The first tuning run picked 3 — the old
        # lower bound — and the boundary check fired, meaning the search wanted
        # to go coarser than the range allowed. A bound that binds is not a
        # choice, it is an artefact, and the previous study was rejected partly
        # for reporting one (VMD K=8 sat on its ceiling).
        #
        # If the optimum turns out to be level 1 or 2, that is a result worth
        # reporting rather than hiding: it would say the multiresolution split
        # buys little on this signal once the window is causal.
        return {"wavelet": ["db4", "db6", "db8", "sym5", "sym7", "coif3", "coif5"],
                "level": (1, 6)}

    def min_window(self):
        # The coarsest scale needs at least 2^level samples, and the filter
        # support must fit inside it. 2^L * dec_len is the practical floor.
        import pywt
        dec_len = pywt.Wavelet(self.params["wavelet"]).dec_len
        return int((2 ** int(self.params["level"])) * dec_len // 2)

    def expected_k(self):
        return int(self.params["level"]) + 1


# =============================================================================
# variational
# =============================================================================
class VMDDecomposition(BaseDecomposition):
    """
    Variational mode decomposition.

    Modes are returned sorted by centre frequency so that channel k means the
    same thing at every origin; without sorting, adjacent windows can swap
    modes and the model sees an inconsistent feature definition.
    """

    name = "vmd"

    def decompose_window(self, window):
        # A constant window has no oscillatory content to separate, and the
        # library divides by its standard deviation:
        #
        #     scale_s = np.std(S);  S = S / scale_s      (PyEMD/CEEMDAN.py)
        #
        # so std == 0 gives 0/0 and every component comes back NaN. Those NaNs
        # were written into the feature cache and poisoned training from the
        # first epoch — "train nan val nan" on 1036 trials, every affected
        # study failing with "no completed result".
        #
        # These windows are real: station 3 reads a constant for 1687 hours
        # and station 1 for 820 (see BULGURLAR 1.2), which puts 4768 fully
        # constant windows in the record. The honest decomposition of a
        # constant window is the window itself, in one component; _harmonise
        # then zero-pads it to K, which says "no energy at any other scale"
        # and is exactly right.
        if not np.any(np.diff(window)):
            return np.asarray(window, dtype=np.float64).reshape(1, -1)

        from vmdpy import VMD
        p = self.params
        u, _, omega = VMD(window, p["alpha"], p["tau"], int(p["K"]),
                          p["DC"], p["init"], p["tol"])
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        w_last = np.asarray(omega[-1] if np.ndim(omega) == 2 else omega,
                            dtype=np.float64)

        # vmdpy computes each centre frequency as an energy-weighted mean over
        # the positive half-spectrum, and divides by that mode's energy. When a
        # mode collapses to (near) zero the division is 0/0 and the frequency
        # comes back NaN — the "invalid value encountered in scalar divide"
        # warning. argsort still returns an order, so nothing crashes, but the
        # order is no longer by frequency and channel k stops meaning the same
        # thing at every origin. That is exactly what the sort exists to
        # guarantee, so a NaN here is not cosmetic.
        if not np.all(np.isfinite(w_last)) or len(w_last) != len(u):
            w_last = self._centre_frequencies(u)

        return u[np.argsort(w_last, kind="stable")]

    @staticmethod
    def _centre_frequencies(modes: np.ndarray) -> np.ndarray:
        """
        Spectral centroid of each mode, computed from the mode itself.

        Used when vmdpy's own frequencies are unusable. A mode with no energy
        gets +inf so it sorts last deterministically rather than landing in an
        arbitrary position.
        """
        n = modes.shape[1]
        freqs = np.fft.rfftfreq(n)
        out = np.empty(len(modes))
        for i, m in enumerate(modes):
            power = np.abs(np.fft.rfft(m)) ** 2
            total = power.sum()
            out[i] = float(np.dot(freqs, power) / total) if total > 0 else np.inf
        return out

    @staticmethod
    def default_params():
        return {"K": 6, "alpha": 2000.0, "tau": 0.0, "DC": 0,
                "init": 1, "tol": 1e-7}

    @staticmethod
    def param_space():
        return {"K": (3, 10), "alpha": (500.0, 5000.0)}

    def validate_params(self):
        if int(self.params["K"]) < 2:
            raise ValueError("VMD needs at least 2 modes")

    def min_window(self):
        # Enough samples to resolve K bands with some margin.
        return int(max(64, 16 * int(self.params["K"])))

    def expected_k(self):
        return int(self.params["K"])


# =============================================================================
# empirical mode family
# =============================================================================
class EMDDecomposition(BaseDecomposition):
    """Empirical mode decomposition. IMF count is data dependent."""

    name = "emd"

    def decompose_window(self, window):
        from PyEMD import EMD
        return EMD().emd(window, max_imf=int(self.params["max_imfs"]))

    @staticmethod
    def default_params():
        return {"max_imfs": 8}

    def expected_k(self):
        """
        K is known in advance: PyEMD returns at most `max_imfs` modes and
        appends the residue as the last row.

        This used to return None, which sent precompute off to measure K from
        the data — station by station. Where a sensor is stuck (station 3
        reads a constant for 1687 hours) a probe window is flat, sifting finds
        no extrema, and the measurement came back as 1. Stations then
        disagreed and the loader refused to stack them. Declaring the value
        removes the measurement, and with it the possibility of disagreement.

        A window that yields fewer modes is zero-padded by
        `causal_features._harmonise`, which is the honest reading: that window
        carries no energy at those scales.
        """
        return int(self.params["max_imfs"]) + 1


    @staticmethod
    def param_space():
        return {"max_imfs": (4, 10)}

    def min_window(self):
        return 128


class EEMDDecomposition(BaseDecomposition):
    """
    Ensemble EMD. Stochastic: the noise realisations are seeded from the window
    contents so that the same window always yields the same components. Without
    this the causality guard cannot separate a genuine leak from sampling noise,
    and the cache is not reproducible.
    """

    name = "eemd"
    is_stochastic = True

    def decompose_window(self, window):
        # A constant window has no oscillatory content to separate, and the
        # library divides by its standard deviation:
        #
        #     scale_s = np.std(S);  S = S / scale_s      (PyEMD/CEEMDAN.py)
        #
        # so std == 0 gives 0/0 and every component comes back NaN. Those NaNs
        # were written into the feature cache and poisoned training from the
        # first epoch — "train nan val nan" on 1036 trials, every affected
        # study failing with "no completed result".
        #
        # These windows are real: station 3 reads a constant for 1687 hours
        # and station 1 for 820 (see BULGURLAR 1.2), which puts 4768 fully
        # constant windows in the record. The honest decomposition of a
        # constant window is the window itself, in one component; _harmonise
        # then zero-pads it to K, which says "no energy at any other scale"
        # and is exactly right.
        if not np.any(np.diff(window)):
            return np.asarray(window, dtype=np.float64).reshape(1, -1)

        from PyEMD import EEMD
        # parallel=False is NOT a performance compromise, it is the only
        # correct setting here. PyEMD defaults to parallel=True with
        # processes=os.cpu_count(), so every EEMD() would open a Pool of 56
        # on a 56-core node. This stage already runs 56 windows at once, one
        # per core, so the nested pools would ask for 56*56 = 3136 processes
        # and the node would spend its time context-switching. Parallelism
        # belongs at the unit level, where the work is actually independent.
        e = EEMD(trials=int(self.params["ensemble_size"]),
                 noise_width=float(self.params["noise_width"]),
                 parallel=False)
        e.noise_seed(_stable_seed(window))
        return e.eemd(window, max_imf=int(self.params["max_imfs"]))

    @staticmethod
    def default_params():
        return {"max_imfs": 8, "noise_width": 0.2, "ensemble_size": 250}

    def expected_k(self):
        """
        K is known in advance: PyEMD returns at most `max_imfs` modes and
        appends the residue as the last row.

        This used to return None, which sent precompute off to measure K from
        the data — station by station. Where a sensor is stuck (station 3
        reads a constant for 1687 hours) a probe window is flat, sifting finds
        no extrema, and the measurement came back as 1. Stations then
        disagreed and the loader refused to stack them. Declaring the value
        removes the measurement, and with it the possibility of disagreement.

        A window that yields fewer modes is zero-padded by
        `causal_features._harmonise`, which is the honest reading: that window
        carries no energy at those scales.
        """
        return int(self.params["max_imfs"]) + 1


    @staticmethod
    def param_space():
        return {"max_imfs": (4, 10), "noise_width": (0.05, 0.4),
                "ensemble_size": [100, 250, 500]}

    def min_window(self):
        return 128


class CEEMDANDecomposition(BaseDecomposition):
    """Complete EEMD with adaptive noise. Stochastic, seeded as for EEMD."""

    name = "ceemdan"
    is_stochastic = True

    def decompose_window(self, window):
        # A constant window has no oscillatory content to separate, and the
        # library divides by its standard deviation:
        #
        #     scale_s = np.std(S);  S = S / scale_s      (PyEMD/CEEMDAN.py)
        #
        # so std == 0 gives 0/0 and every component comes back NaN. Those NaNs
        # were written into the feature cache and poisoned training from the
        # first epoch — "train nan val nan" on 1036 trials, every affected
        # study failing with "no completed result".
        #
        # These windows are real: station 3 reads a constant for 1687 hours
        # and station 1 for 820 (see BULGURLAR 1.2), which puts 4768 fully
        # constant windows in the record. The honest decomposition of a
        # constant window is the window itself, in one component; _harmonise
        # then zero-pads it to K, which says "no energy at any other scale"
        # and is exactly right.
        if not np.any(np.diff(window)):
            return np.asarray(window, dtype=np.float64).reshape(1, -1)

        from PyEMD import CEEMDAN
        # See EEMDDecomposition for why parallel=False. CEEMDAN is worse than
        # EEMD in this respect: it opens a fresh Pool for the noise
        # decomposition and another for every IMF iteration, and closes them
        # without joining, so the processes accumulate over a single call.
        c = CEEMDAN(trials=int(self.params["trials"]),
                    epsilon=float(self.params["epsilon"]),
                    parallel=False)
        c.noise_seed(_stable_seed(window))
        return c.ceemdan(window, max_imf=int(self.params["max_imfs"]))

    @staticmethod
    def default_params():
        return {"max_imfs": 8, "epsilon": 0.2, "trials": 250}

    def expected_k(self):
        """
        K is known in advance: PyEMD returns at most `max_imfs` modes and
        appends the residue as the last row.

        This used to return None, which sent precompute off to measure K from
        the data — station by station. Where a sensor is stuck (station 3
        reads a constant for 1687 hours) a probe window is flat, sifting finds
        no extrema, and the measurement came back as 1. Stations then
        disagreed and the loader refused to stack them. Declaring the value
        removes the measurement, and with it the possibility of disagreement.

        A window that yields fewer modes is zero-padded by
        `causal_features._harmonise`, which is the honest reading: that window
        carries no energy at those scales.
        """
        return int(self.params["max_imfs"]) + 1


    @staticmethod
    def param_space():
        return {"max_imfs": (4, 10), "epsilon": (0.05, 0.4),
                "trials": [100, 250, 500]}

    def min_window(self):
        return 128


# =============================================================================
# registry
# =============================================================================
REGISTRY: Dict[str, type] = {
    cls.name: cls for cls in (
        NoDecomposition, DWTDecomposition, VMDDecomposition,
        EMDDecomposition, EEMDDecomposition, CEEMDANDecomposition,
    )
}


def get_decomposer(method: str, params: Optional[dict] = None) -> BaseDecomposition:
    """
    Build a decomposer. The returned object is callable as `f(window)`, so it
    drops straight into the feature builder.
    """
    if method not in REGISTRY:
        raise ValueError(f"unknown decomposition method: {method}. "
                         f"Known: {sorted(REGISTRY)}")
    return REGISTRY[method](params)


def default_params(method: str) -> dict:
    return dict(REGISTRY[method].default_params())


def param_space(method: str) -> dict:
    return dict(REGISTRY[method].param_space())


def min_window_for(method: str, params: Optional[dict] = None) -> int:
    return REGISTRY[method](params).min_window()


# =============================================================================
# decomposition quality metrics (method independent)
# =============================================================================
def envelope_entropy(comps: np.ndarray) -> float:
    """Mean Shannon entropy of the Hilbert amplitude envelope per component."""
    vals = []
    for c in np.atleast_2d(comps):
        n = len(c)
        F = np.fft.fft(c)
        h = np.zeros(n)
        h[0] = 1
        if n % 2 == 0:
            h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[1:(n + 1) // 2] = 2
        a = np.abs(np.fft.ifft(F * h))
        p = a / (a.sum() + 1e-12)
        vals.append(-(p * np.log(p + 1e-12)).sum())
    return float(np.mean(vals))


def orthogonality_index(comps: np.ndarray, signal: np.ndarray) -> float:
    """Near zero when components are mutually independent."""
    comps = np.atleast_2d(comps)
    num = 0.0
    for i in range(len(comps)):
        for j in range(len(comps)):
            if i != j:
                num += float(np.sum(comps[i] * comps[j]))
    return abs(num / (float(np.sum(np.asarray(signal) ** 2)) + 1e-12))


def energy_conservation_ratio(comps: np.ndarray, signal: np.ndarray) -> float:
    e_sig = float(np.sum(np.asarray(signal) ** 2)) + 1e-12
    return abs(float(np.sum(np.atleast_2d(comps) ** 2)) - e_sig) / e_sig


def composite_score(comps: np.ndarray, signal: np.ndarray,
                    weights=(1.0, 0.5, 0.3)) -> float:
    """
    Lower is better. Note this scores the DECOMPOSITION, not the forecast;
    a configuration that minimises it need not forecast well. Biomimetics
    reviewers flagged exactly this gap, so the paper reports both.
    """
    w1, w2, w3 = weights
    return (w1 * envelope_entropy(comps)
            + w2 * orthogonality_index(comps, signal)
            + w3 * energy_conservation_ratio(comps, signal))
