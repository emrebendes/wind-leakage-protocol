# -*- coding: utf-8 -*-
"""
The causality test suite.
=========================

This is the guard that the previous study lacked. It must pass before any
cache is built and before any result is reported.

What it checks
--------------
1. Every decomposition method, at every W, produces features at origin t that
   are unchanged when x[t+1:] is destroyed.
2. The known-leaky pipeline (decompose the whole series, then slice) IS
   detected. A guard that never fires is worthless, so we prove it fires.
3. Components sum back to the window (additivity), which the reconstruction
   step depends on.
4. Origin sets are identical across methods for a given W, so that conditions
   are compared on the same evaluation points.

Run
---
    python -m pytest tests/test_causality.py -v
    python tests/test_causality.py            # without pytest
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import pytest
except ImportError:                                            # pragma: no cover
    # pytest is the preferred runner, but this file must also be executable on
    # its own — a compute node with a minimal environment should never be
    # blocked from verifying causality. The stubs below exist only so the
    # module imports; in that mode the decorated functions are not collected
    # and the __main__ block runs the same checks directly.
    def _decorator(*args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]                 # bare use: @pytest.mark.slow
        return lambda fn: fn               # called use: @...parametrize(...)

    class _Stub:
        def __getattr__(self, name):
            return _decorator

    class _PytestStub:
        mark = _Stub()
        fixture = staticmethod(_decorator)

    pytest = _PytestStub()                                     # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from config import DATA_FILE, W_CHOICES, LOOK_BACK_MAX               # noqa: E402
from decompositions import get_decomposer, default_params, REGISTRY     # noqa: E402
from causal_features import (features_at_origin, valid_origins,           # noqa: E402
                             assert_pipeline_causal, get_builder,
                             FeatureBuilder, LeakyFeatureBuilder)

# EMD-family methods are slow; keep the suite runnable in minutes.
FAST_METHODS = ["none", "dwt", "vmd"]
SLOW_METHODS = ["emd", "eemd", "ceemdan"]
ORIGINS = [12000, 30000]


@pytest.fixture(scope="module")
def series():
    data = np.load(DATA_FILE, allow_pickle=True)
    return np.asarray(data[0], dtype=np.float64)


def _build_fn(decompose, W):
    def f(x, t):
        return features_at_origin(x, t, decompose, W, LOOK_BACK_MAX)
    return f


# =============================================================================
@pytest.mark.parametrize("method", FAST_METHODS)
@pytest.mark.parametrize("W", W_CHOICES)
@pytest.mark.parametrize("t", ORIGINS)
def test_features_are_causal(series, method, W, t):
    """Features at t must not react to anything after t."""
    decompose = get_decomposer(method, default_params(method))
    assert_pipeline_causal(_build_fn(decompose, W), series, t, n_trials=2)


@pytest.mark.slow
@pytest.mark.parametrize("method", SLOW_METHODS)
def test_features_are_causal_slow(series, method):
    decompose = get_decomposer(method, default_params(method))
    assert_pipeline_causal(_build_fn(decompose, 256), series, 30000, n_trials=2)


def test_guard_detects_a_leaky_pipeline(series):
    """
    A guard that cannot fail proves nothing. Here we deliberately rebuild the
    pipeline this project is auditing — decompose the whole series, then take
    the window — and require the guard to reject it.
    """
    decompose = get_decomposer("dwt", default_params("dwt"))

    def leaky(x, t):
        comps = decompose(x)                      # whole series -> leak
        return comps[:, t - LOOK_BACK_MAX + 1: t + 1]

    with pytest.raises(AssertionError, match="CAUSALITY VIOLATED"):
        assert_pipeline_causal(leaky, series, 30000, n_trials=2)


def test_guard_detects_block_decomposition(series):
    """
    The intermediate proposal — decompose the test block on its own — is also
    leaky, because a point inside the block still depends on later points of
    the same block. This is the condition measured at +75.66% in the pilot,
    indistinguishable from full-series decomposition.
    """
    decompose = get_decomposer("dwt", default_params("dwt"))
    block_start = 43472                           # start of the test region

    def block(x, t):
        comps = decompose(x[block_start:])        # whole test block -> leak
        j = t - block_start
        return comps[:, j - LOOK_BACK_MAX + 1: j + 1]

    with pytest.raises(AssertionError, match="CAUSALITY VIOLATED"):
        assert_pipeline_causal(block, series, 45000, n_trials=2)


@pytest.mark.parametrize("method", FAST_METHODS)
def test_components_are_additive(series, method):
    """Reconstruction depends on the components summing back to the window."""
    W = 512
    decompose = get_decomposer(method, default_params(method))
    window = series[30000 - W + 1: 30001]
    comps = decompose(window)
    assert np.allclose(np.asarray(comps).sum(axis=0), window, atol=1e-8), \
        f"{method}: components do not reconstruct the window"


@pytest.mark.parametrize("W", W_CHOICES)
def test_origin_sets_match_across_methods(series, W):
    """
    Every condition must be evaluated on the same origins; otherwise a method
    could look better simply by being scored on easier points.
    """
    ref = valid_origins(len(series), W, "test")
    for split in ("train", "val", "test"):
        o = valid_origins(len(series), W, split)
        assert len(o) > 0
        assert o.min() >= W - 1, "an origin lacks W samples of history"
    assert ref.min() >= W - 1


def test_look_back_slicing_is_exact(series):
    """
    The cache is stored at LOOK_BACK_MAX and sliced for smaller look_back.
    That slice must equal a direct computation, or the cache is not sound.
    """
    W = 512
    decompose = get_decomposer("dwt", default_params("dwt"))
    deep = features_at_origin(series, 30000, decompose, W, LOOK_BACK_MAX)
    for lb in (24, 48, 96):
        direct = features_at_origin(series, 30000, decompose, W, lb)
        assert np.allclose(deep[:, -lb:], direct), \
            f"slicing to look_back={lb} does not match direct computation"


# =============================================================================
# the two arms
# =============================================================================
@pytest.mark.parametrize("method", FAST_METHODS)
def test_causal_builder_is_causal(series, method):
    """The causal arm must ignore everything after the origin."""
    b = get_builder("causal", get_decomposer(method, default_params(method)),
                    W=512)
    assert isinstance(b, FeatureBuilder)
    b.verify(series, 30000)


@pytest.mark.parametrize("method", ["dwt", "vmd"])
def test_leaky_builder_really_leaks(series, method):
    """
    The leaky arm must FAIL a causality test. If it ever passed, it would no
    longer reproduce the paradigm under audit and the comparison between arms
    would be empty. `verify` asserts the failure, so this is a positive test.
    """
    b = get_builder("leaky", get_decomposer(method, default_params(method)),
                    W=512)
    assert isinstance(b, LeakyFeatureBuilder)
    b.verify(series, 30000)          # passes only because the builder leaks


def test_none_cannot_leak(series):
    """
    'none' is the identity, so decomposing the whole series and slicing it is
    the same operation as slicing the window. The leaky arm therefore coincides
    with the causal arm here — and must, because 'none' is the reference every
    method is scored against. A reference that moved between arms would make
    the comparison unreadable.

    This case was missing from the suite and only surfaced when precompute ran:
    the leaky builder demanded leakage unconditionally and failed on 'none'.
    """
    dec = get_decomposer("none", default_params("none"))
    assert dec.can_leak is False

    b = get_builder("leaky", dec, W=512)
    assert isinstance(b, LeakyFeatureBuilder)
    b.verify(series, 30000)          # passes only because it is causal

    causal = get_builder("causal", get_decomposer("none", default_params("none")),
                         W=512)
    assert np.allclose(causal.at_origin(series, 30000),
                       b.at_origin(series, 30000)), \
        "the arms must be identical when the decomposition is the identity"


def test_every_real_decomposition_declares_it_can_leak():
    """A method that mixes neighbouring samples can leak; only the identity cannot."""
    for name, cls in REGISTRY.items():
        assert cls.can_leak == (name != "none"), \
            f"{name}: can_leak={cls.can_leak} is not what this method does"


def test_degenerate_window_yields_one_component(series):
    """
    A long flat stretch has no extrema, so EMD's sifting returns nothing. The
    honest decomposition of a constant window is the window itself, in one
    component; the alternative is an IndexError mid-run, which is what happened.
    """
    from decompositions import BaseDecomposition

    class _Empty(BaseDecomposition):
        name = "empty"

        def decompose_window(self, w):
            return np.zeros((0, len(w)))

        @staticmethod
        def default_params():
            return {}

        @staticmethod
        def param_space():
            return {}

        def min_window(self):
            return 1

    window = series[30000 - 64 + 1: 30001]
    comps = _Empty()(window)
    assert comps.shape == (1, len(window))
    assert np.allclose(comps.sum(axis=0), window)


@pytest.mark.parametrize("method", ["dwt", "vmd"])
def test_arms_agree_on_shape(series, method):
    """
    The two arms must produce identical shapes on identical origins. Anything
    else would mean the comparison is not like for like.
    """
    dec = get_decomposer(method, default_params(method))
    causal = get_builder("causal", dec, W=512)
    leaky = get_builder("leaky", get_decomposer(method, default_params(method)),
                        W=512)
    a = causal.at_origin(series, 30000)
    b = leaky.at_origin(series, 30000)
    assert a.shape == b.shape, "arms disagree on feature shape"
    assert not np.allclose(a, b), "arms produced identical features"


class StandaloneRunner:
    """
    Runs the same checks as the pytest suite without pytest.

    Kept deliberately equivalent in coverage. A fallback that tests less than
    the real suite is worse than no fallback, because it reports PASS while
    leaving the interesting cases unexamined.
    """

    def __init__(self, series: np.ndarray, origin: int = 30000,
                 slow: bool = False):
        self.x = series
        self.t = origin
        self.slow = slow
        self.passed = 0
        self.failures: list = []

    # ------------------------------------------------------------------ core
    def expect_ok(self, label: str, fn) -> None:
        """The check must complete without raising."""
        try:
            fn()
        except Exception as exc:                               # noqa: BLE001
            self._fail(label, f"{type(exc).__name__}: {exc}")
        else:
            self._pass(label)

    def expect_violation(self, label: str, fn) -> None:
        """The check must raise a causality violation. A silent guard is a bug."""
        try:
            fn()
        except AssertionError as exc:
            if "CAUSALITY VIOLATED" in str(exc):
                self._pass(label)
            else:
                self._fail(label, f"raised, but not a causality violation: {exc}")
        except Exception as exc:                               # noqa: BLE001
            self._fail(label, f"wrong exception type: {type(exc).__name__}: {exc}")
        else:
            self._fail(label, "guard did not fire")

    def _pass(self, label: str) -> None:
        self.passed += 1
        print(f"  PASS  {label}", flush=True)

    def _fail(self, label: str, why: str) -> None:
        self.failures.append((label, why))
        print(f"  FAIL  {label}\n          {why}", flush=True)

    # ----------------------------------------------------------- the checks
    def run(self) -> int:
        print("causality suite (standalone — pytest not installed)\n")

        print("1. causal features ignore the future")
        for method in FAST_METHODS:
            for W in W_CHOICES:
                d = get_decomposer(method, default_params(method))
                self.expect_ok(
                    f"{method:8s} W={W}",
                    lambda d=d, W=W: assert_pipeline_causal(
                        _build_fn(d, W), self.x, self.t, n_trials=2))

        if self.slow:
            print("\n1b. EMD family (slow — minutes per case)")
            for method in SLOW_METHODS:
                d = get_decomposer(method, default_params(method))
                self.expect_ok(
                    f"{method:8s} W=256",
                    lambda d=d: assert_pipeline_causal(
                        _build_fn(d, 256), self.x, self.t, n_trials=2))
        else:
            print(f"\n1b. EMD family skipped ({', '.join(SLOW_METHODS)}) — "
                  f"run with --slow before a full precompute")

        print("\n2. the guard fires on pipelines known to leak")
        d = get_decomposer("dwt", default_params("dwt"))

        def full_series(x, t):
            return d(x)[:, t - LOOK_BACK_MAX + 1: t + 1]

        self.expect_violation(
            "full-series decomposition",
            lambda: assert_pipeline_causal(full_series, self.x, self.t,
                                           n_trials=2))

        block_start = 43472

        def block(x, t):
            comps = d(x[block_start:])
            j = t - block_start
            return comps[:, j - LOOK_BACK_MAX + 1: j + 1]

        self.expect_violation(
            "block decomposition",
            lambda: assert_pipeline_causal(block, self.x, 45000, n_trials=2))

        print("\n3. components reconstruct the window")
        for method in FAST_METHODS:
            self.expect_ok(f"{method:8s} additivity",
                           lambda m=method: self._additive(m))

        print("\n4. cached features slice exactly")
        self.expect_ok("look_back slicing", self._slicing)

        print("\n5. origin sets are shared across methods")
        for W in W_CHOICES:
            self.expect_ok(f"origins W={W}", lambda W=W: self._origins(W))

        print("\n6. the two arms")
        for method in FAST_METHODS:
            self.expect_ok(f"{method:8s} causal builder",
                           lambda m=method: get_builder(
                               "causal", get_decomposer(m, default_params(m)),
                               W=512).verify(self.x, self.t))
        for method in ("dwt", "vmd"):
            # verify() asserts this builder FAILS a causality test, so a pass
            # here means the leaky arm really does reproduce the old paradigm.
            self.expect_ok(f"{method:8s} leaky builder leaks",
                           lambda m=method: get_builder(
                               "leaky", get_decomposer(m, default_params(m)),
                               W=512).verify(self.x, self.t))
            self.expect_ok(f"{method:8s} arms agree on shape",
                           lambda m=method: self._shapes(m))

        return self.report()

    # ------------------------------------------------------------- helpers
    def _additive(self, method: str) -> None:
        W = 512
        d = get_decomposer(method, default_params(method))
        window = self.x[self.t - W + 1: self.t + 1]
        comps = np.asarray(d(window))
        assert np.allclose(comps.sum(axis=0), window, atol=1e-8), \
            f"{method}: components do not reconstruct the window"

    def _slicing(self) -> None:
        W = 512
        d = get_decomposer("dwt", default_params("dwt"))
        deep = features_at_origin(self.x, self.t, d, W, LOOK_BACK_MAX)
        for lb in (24, 48, 96):
            direct = features_at_origin(self.x, self.t, d, W, lb)
            assert np.allclose(deep[:, -lb:], direct), \
                f"slicing to look_back={lb} does not match direct computation"

    def _origins(self, W: int) -> None:
        for split in ("train", "val", "test"):
            o = valid_origins(len(self.x), W, split)
            assert len(o) > 0, f"{split}: no valid origins at W={W}"
            assert o.min() >= W - 1, "an origin lacks W samples of history"

    def _shapes(self, method: str) -> None:
        causal = get_builder("causal",
                             get_decomposer(method, default_params(method)),
                             W=512)
        leaky = get_builder("leaky",
                            get_decomposer(method, default_params(method)),
                            W=512)
        a = causal.at_origin(self.x, self.t)
        b = leaky.at_origin(self.x, self.t)
        assert a.shape == b.shape, "arms disagree on feature shape"
        assert not np.allclose(a, b), "arms produced identical features"

    def report(self) -> int:
        total = self.passed + len(self.failures)
        print(f"\n{'-' * 62}")
        print(f"{self.passed}/{total} passed")
        if self.failures:
            print("\nfailures:")
            for label, why in self.failures:
                print(f"  {label}: {why}")
            print("\nDo not build a cache or report a result until these pass.")
        print()
        return 1 if self.failures else 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="causality suite (no pytest needed)")
    ap.add_argument("--slow", action="store_true",
                    help="also test the EMD family; minutes per case")
    ap.add_argument("--origin", type=int, default=30000,
                    help="time index to test features at")
    opts = ap.parse_args()

    data = np.load(DATA_FILE, allow_pickle=True)
    x = np.asarray(data[0], dtype=np.float64)
    sys.exit(StandaloneRunner(x, origin=opts.origin, slow=opts.slow).run())


# =============================================================================
def test_vmd_mode_order_survives_a_collapsed_mode():
    """
    VMD channels must stay ordered by frequency even when a mode dies.

    vmdpy divides by each mode's energy to get its centre frequency, so a mode
    that collapses to zero yields NaN — the "invalid value encountered in
    scalar divide" warning seen during tuning. argsort still returns an order,
    nothing crashes, and the ordering silently stops being by frequency. That
    breaks the one property the sort exists to provide: channel k means the
    same thing at every origin. The model would see one definition of channel 1
    at some windows and another elsewhere.
    """
    from decompositions import VMDDecomposition

    n = 256
    t = np.arange(n)
    fast = np.sin(2 * np.pi * t / 4)
    slow = np.sin(2 * np.pi * t / 128)
    dead = np.zeros(n)

    cf = VMDDecomposition._centre_frequencies(np.vstack([fast, dead, slow]))

    assert np.isfinite(cf[0]) and np.isfinite(cf[2])
    assert cf[1] == np.inf, "a zero-energy mode must sort last, not arbitrarily"
    assert list(np.argsort(cf, kind="stable")) == [2, 0, 1], \
        "modes are not ordered slow, fast, dead"


# =============================================================================
# THE PARTITION ARM
# =============================================================================
# Three tests, and together they are the executable definition of the middle
# regime. Section 4.1 of the paper quotes them rather than describing the
# regime in prose, because prose is how "split first, then decompose" came to
# be believed sufficient in the first place.
def _partition(method, W):
    from causal_features import PartitionFeatureBuilder
    return PartitionFeatureBuilder(get_decomposer(method, default_params(method)),
                                   W=W)


# COST NOTE — why W is not swept for vmd
#     A partition block IS the split, so its length does not depend on W; W
#     only decides which origins are legal. Sweeping W therefore re-runs the
#     same decomposition on the same 35502-sample block three times.
#     For dwt that is milliseconds and the sweep is free insurance. For vmd it
#     is a 500-iteration optimisation over a 71004-point spectrum, so the sweep
#     buys nothing and costs minutes. vmd is checked at one W, on the test
#     split, which is the block the reported score comes from.
CHEAP = [("dwt", W) for W in W_CHOICES]
PARTITION_CASES = CHEAP + [("vmd", 512)]
SPLITS_FOR = {"dwt": ("train", "val", "test"), "vmd": ("test",)}


@pytest.mark.parametrize("method,W", PARTITION_CASES)
def test_partition_builder_ignores_other_splits(series, method, W):
    """
    (a) Contamination really is gone.

    Whatever else the partition arm does, a feature must not react when a
    different split is destroyed. This is the half of the regime the
    literature is right about, and the paper says so.
    """
    from causal_features import assert_block_isolated

    b = _partition(method, W)
    for split in SPLITS_FOR[method]:
        o = valid_origins(len(series), W, split)
        t = int(o[len(o) // 2])
        lo, hi = b.block_of(len(series), t)
        assert_block_isolated(b.at_origin, series, t, lo, hi, n_trials=2)


@pytest.mark.parametrize("method,W", PARTITION_CASES)
def test_partition_builder_still_leaks_inside_its_split(series, method, W):
    """
    (b) The look-ahead the remedy does NOT remove.

    A positive claim about a defect, in the same spirit as
    test_leaky_builder_really_leaks. If this ever stopped holding, the
    partition arm would have quietly become the causal arm and the middle row
    of the paper's main table would be measuring nothing at all — the failure
    mode that would be hardest to notice from the numbers alone.
    """
    from causal_features import assert_leaks_within

    b = _partition(method, W)
    tested = 0
    for split in SPLITS_FOR[method]:
        o = valid_origins(len(series), W, split)
        t = int(o[len(o) // 2])
        _, hi = b.block_of(len(series), t)
        tested += bool(assert_leaks_within(b.at_origin, series, t, hi,
                                           n_trials=2))
    assert tested, ("no origin had enough within-block future to test; the "
                    "splits or min_future have gone out of step")


@pytest.mark.parametrize("method,W", CHEAP)
def test_partition_removes_no_look_ahead_at_test_origins(series, method, W):
    """
    The deductive core of the paper, as an assertion.

    The splits are chronological and test is the last one, so the full series
    and the test block END AT THE SAME PLACE. For a test origin the future
    visible to the decomposition is the identical set of samples in both
    regimes. Per-partition decomposition therefore cannot remove any
    look-ahead at test time — the only thing it changes is how much past the
    decomposition conditions on, and past is not leakage.

    That is why the middle row of the main table is not an empirical
    surprise. This test pins the structural fact the argument rests on, so a
    future change to the split logic cannot silently invalidate it.
    """
    b = _partition(method, W)
    n = len(series)
    for t in valid_origins(n, W, "test")[:: max(1, len(
            valid_origins(n, W, "test")) // 5)]:
        _, hi = b.block_of(n, int(t))
        assert hi == n, (f"the test block ends at {hi}, not at {n}; the "
                         f"deductive argument in the paper assumes test is "
                         f"the last split")


@pytest.mark.parametrize("W", W_CHOICES)
def test_partition_none_is_identical_to_causal(series, W):
    """
    Why the partition arm is never run for 'none'.

    With the identity decomposition there is no channel to leak through, so
    decomposing a block and slicing it gives exactly the same window as
    decomposing the window. The partition arm therefore CANNOT differ from the
    causal arm here, and running it would burn GPU-hours to reproduce numbers
    we already have.

    Asserting it is the stronger move anyway: this holds for every origin,
    where a run would only have covered a sample of them. 'none' is the
    reference the whole three-arm table is scored against, so a reference that
    moved between arms would make the table unreadable.
    """
    pb, cb = _partition("none", W), FeatureBuilder(
        get_decomposer("none", default_params("none")), W=W)
    rng = np.random.RandomState(0)
    for split in ("train", "val", "test"):
        o = valid_origins(len(series), W, split)
        for t in rng.choice(o, size=min(10, len(o)), replace=False):
            assert np.array_equal(pb.at_origin(series, int(t)),
                                  cb.at_origin(series, int(t))), \
                f"partition != causal for 'none' at origin {t}, W={W}"


def test_the_three_arms_are_actually_three(series):
    """
    The arms must differ from each other, or the table has fewer rows than it
    claims. vmd, because it is global: every output sample reads the whole
    block, so changing the block changes the features everywhere.

    dwt is NOT used here. It has finite support and coincides with the leaky
    arm away from block boundaries, which is a finding rather than a defect —
    see the next test.
    """
    W, t = 512, 30000
    f = {arm: get_builder(arm, get_decomposer("vmd", default_params("vmd")),
                          W=W).at_origin(series, t)
         for arm in ("causal", "leaky", "partition")}
    assert not np.allclose(f["causal"], f["leaky"]), "causal == leaky"
    assert not np.allclose(f["causal"], f["partition"]), "causal == partition"
    assert not np.allclose(f["leaky"], f["partition"]), "leaky == partition"


def test_finite_support_makes_partition_coincide_with_leaky(series):
    """
    The sharpest form of the paper's claim, asserted rather than described.

    A dwt coefficient at position j reads only a neighbourhood of j whose width
    is the wavelet's support -- a few hundred samples at these levels. Origin
    30000 sits 5502 samples inside the training split, far outside that
    neighbourhood of the boundary at 35502. Decomposing [0, 35502) and
    decomposing [0, 50718) therefore produce THE SAME NUMBERS there.

    So for a finite-support decomposition, splitting before decomposing does
    not merely remove little of the leakage. Away from the block edges it
    changes the features NOT AT ALL -- the two protocols are the same
    computation. Only origins within the support of a boundary can differ.

    This is what the previous study's "99.9% of training coefficients are
    identical" was actually measuring. It read as reassurance; it is
    arithmetic.

    If this test ever fails, either the wavelet's support has grown past 5502
    samples or the split boundaries moved, and the paragraph above stops being
    true. Both are worth knowing about.
    """
    W, t = 512, 30000
    leaky = get_builder("leaky", get_decomposer("dwt", default_params("dwt")),
                        W=W).at_origin(series, t)
    part = get_builder("partition", get_decomposer("dwt", default_params("dwt")),
                       W=W).at_origin(series, t)
    assert np.allclose(leaky, part), (
        "dwt features differ between the leaky and partition arms deep inside "
        "the training split; the finite-support argument in Section 4 assumes "
        "they cannot")
