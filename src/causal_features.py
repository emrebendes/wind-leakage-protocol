# -*- coding: utf-8 -*-
"""
Causal feature construction.
============================

THE ONE RULE OF THIS FILE
-------------------------
The feature vector for forecast origin t may depend on x[0..t] and on nothing
else. Every function here is written so that violating that rule is difficult,
and `assert_causal` makes a violation loud.

At origin t we take the trailing window x[t-W+1 .. t], decompose THAT WINDOW
ALONE, and read the last `look_back` samples of each resulting component:

        window            = x[t-W+1 .. t]              (W samples, all <= t)
        components        = decompose(window)          -> (K, W)
        features          = components[:, -look_back:] -> (K, look_back)

Contrast with the pipeline this project is auditing, which decomposed the whole
series once and then split. There the component at time t is a function of
x[t+1], x[t+2], ... — information no operator has at time t.

The cache stores features at LOOK_BACK_MAX depth; a smaller look_back is
obtained by slicing the tail, which is exact and needs no recomputation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time

import numpy as np

from config import LOOK_BACK_MAX, TRAIN_RATIO, VAL_RATIO, MAX_HORIZON


# =============================================================================
# WINDOW / ORIGIN BOOKKEEPING
# =============================================================================
def split_bounds(n: int):
    """Chronological split points for a series of length n."""
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return train_end, val_end


def valid_origins(n: int, W: int, split: str, max_horizon: int = MAX_HORIZON):
    """
    Forecast origins that have W samples of history behind them and
    max_horizon samples of future ahead of them, restricted to one split.

    Every condition (with and without decomposition) is evaluated on the SAME
    origin set so that comparisons are not confounded by differing coverage.
    """
    train_end, val_end = split_bounds(n)
    lo = W - 1
    if split == "train":
        return np.arange(max(lo, 0), train_end - max_horizon)
    if split == "val":
        return np.arange(max(lo, train_end), val_end - max_horizon)
    if split == "test":
        return np.arange(max(lo, val_end), n - max_horizon)
    raise ValueError(f"unknown split: {split}")


def cache_signature(method: str, params: dict, W: int) -> str:
    """
    Stable identifier for a cache file. Any change to the decomposition
    parameters produces a different signature, so a stale cache can never be
    silently reused.
    """
    payload = json.dumps({"method": method, "params": params, "W": W,
                          "depth": LOOK_BACK_MAX}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# =============================================================================
# FEATURE BUILDERS
# =============================================================================
def features_at_origin(x: np.ndarray, t: int, decompose, W: int,
                       depth: int = LOOK_BACK_MAX) -> np.ndarray:
    """
    Feature block for a single origin.

    Returns (K, depth). Only x[t-W+1 .. t] is read — enforced by construction:
    the slice below is the only place `x` is touched.
    """
    if t - W + 1 < 0:
        raise ValueError(f"origin {t} has less than W={W} samples of history")
    window = x[t - W + 1: t + 1]                 # <-- the ONLY read of x
    comps = decompose(window)                    # (K, W)
    comps = np.asarray(comps, dtype=np.float32)
    if comps.ndim == 1:
        comps = comps[None, :]
    if comps.shape[1] < depth:
        raise ValueError(f"W={W} shorter than depth={depth}")
    return comps[:, -depth:]


def features_for_origins(x: np.ndarray, origins: np.ndarray, decompose, W: int,
                         depth: int = LOOK_BACK_MAX,
                         progress_every: int = 0) -> np.ndarray:
    """Feature tensor (n_origins, K, depth) for a list of origins."""
    out = None
    for i, t in enumerate(origins):
        f = features_at_origin(x, int(t), decompose, W, depth)
        if out is None:
            out = np.empty((len(origins), f.shape[0], depth), dtype=np.float32)
        elif f.shape[0] != out.shape[1]:
            # EMD-family methods can return a different number of IMFs per
            # window. Harmonise by folding the surplus into the residual.
            f = _harmonise(f, out.shape[1])
        out[i] = f
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    {i + 1}/{len(origins)}", flush=True)
    return out


def _harmonise(f: np.ndarray, k_target: int) -> np.ndarray:
    """
    Force a (K, depth) block to k_target components while preserving the sum,
    so that the components remain exactly additive.
    """
    k = f.shape[0]
    if k == k_target:
        return f
    if k > k_target:
        head = f[: k_target - 1]
        residual = f[k_target - 1:].sum(axis=0, keepdims=True)
        return np.concatenate([head, residual], axis=0)
    pad = np.zeros((k_target - k, f.shape[1]), dtype=f.dtype)
    return np.concatenate([f, pad], axis=0)


def probe_component_count(x: np.ndarray, origins: np.ndarray, decompose,
                          W: int, n_probe: int = 64,
                          quantile: float = 0.10) -> int:
    """
    Decide the channel count K before building the cache.

    This used to return the raw MINIMUM over the probes, on the reasoning that
    harmonisation should only ever fold components together and never invent
    them. The reasoning is right; the estimator was not. These records contain
    stretches where the sensor is stuck — station 3 reads a constant for 1687
    consecutive hours, station 1 for 820 — and a window that lands inside one
    of them is flat, so EMD returns a single component. One such probe out of
    thirty-two dragged K to 1 for the whole station, which would have made the
    EMD family numerically identical to no decomposition at all. Seven
    stations answered 5; the eighth answered 1 because of a broken sensor.

    Two changes make the estimate robust without abandoning the intent:
    constant windows are dropped, because a flat signal has no decomposition
    to speak of and says nothing about how many components the method yields;
    and a low quantile replaces the minimum, so a single near-degenerate
    window cannot decide the answer for everything else. Windows that do come
    out short are padded with zeros by `_harmonise`, which is the honest
    reading — that window has no energy at that scale.
    """
    rng = np.random.RandomState(0)
    sample = rng.choice(origins, size=min(n_probe, len(origins)), replace=False)
    counts, skipped = [], 0
    for t in sample:
        window = x[int(t) - W + 1: int(t) + 1]
        if not np.any(np.diff(window)):
            skipped += 1                  # stuck sensor: nothing to decompose
            continue
        comps = np.asarray(decompose(window))
        counts.append(1 if comps.ndim == 1 else comps.shape[0])
    if not counts:
        return 1                          # every probe was flat
    k = int(np.floor(np.quantile(counts, quantile)))
    return max(1, k)


# =============================================================================
# THE GUARD
# =============================================================================
def assert_causal(x: np.ndarray, decompose, W: int, t: int,
                  depth: int = LOOK_BACK_MAX, n_trials: int = 3,
                  atol: float = 1e-5, seed: int = 0) -> None:
    """
    Verify that the features at origin t do not depend on x[t+1:].

    Method: compute the features, then destroy the future by permuting
    x[t+1:], and compute them again. Anything other than an identical result
    means future information is reaching the model.

    Stochastic methods (EEMD, CEEMDAN) inject random noise, so their output is
    not bit-reproducible; the caller is expected to seed them. Where that is
    impossible the comparison uses `atol`.
    """
    rng = np.random.RandomState(seed)
    base = features_at_origin(x, t, decompose, W, depth)

    for trial in range(n_trials):
        x2 = np.array(x, dtype=float, copy=True)
        tail = x2[t + 1:]
        if len(tail) > 1:
            x2[t + 1:] = rng.permutation(tail)
        again = features_at_origin(x2, t, decompose, W, depth)

        if again.shape != base.shape:
            raise AssertionError(
                f"CAUSALITY VIOLATED: component count changed when the future "
                f"was permuted (origin {t}, W={W}, trial {trial})"
            )
        if not np.allclose(base, again, atol=atol, rtol=0.0):
            worst = float(np.max(np.abs(base - again)))
            raise AssertionError(
                f"CAUSALITY VIOLATED at origin {t} (W={W}, trial {trial}): "
                f"features changed by up to {worst:.3e} when only x[t+1:] was "
                f"permuted. The decomposition is reading the future."
            )


class WindowFeatureBuilder:
    """
    A decomposer bound to a window size and a depth.

    Without this, every caller has to carry the (decomposer, W, depth) triple
    around and pass it through each function — and a caller that passes the
    wrong W silently produces a cache that disagrees with its metadata. Binding
    them together makes that mistake impossible and gives the causality guard
    a natural home.

    Two subclasses differ in ONE respect — when the decomposition is computed —
    and in nothing else. Same origins, same shapes, same channel count, same
    downstream code. That is what makes the two arms of the study comparable:
    any difference in the results is attributable to the timing alone.
    """

    arm: str = "base"
    is_causal: bool = True

    def __init__(self, decomposer, W: int, depth: int = LOOK_BACK_MAX):
        need = getattr(decomposer, "min_window", lambda: 1)()
        if W < need:
            raise ValueError(
                f"{getattr(decomposer, 'name', 'decomposer')}: W={W} is below "
                f"the {need}-sample minimum for params "
                f"{getattr(decomposer, 'params', {})}"
            )
        if depth > W:
            raise ValueError(f"depth={depth} exceeds W={W}")
        self.decomposer = decomposer
        self.W = int(W)
        self.depth = int(depth)

    # ------------------------------------------------------------- contract
    def at_origin(self, x: np.ndarray, t: int) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------- features
    def for_origins(self, x: np.ndarray, origins, k_target: int | None = None,
                    progress_every: int = 0) -> np.ndarray:
        out = None
        for i, t in enumerate(origins):
            f = self.at_origin(x, int(t))
            if out is None:
                k = k_target or f.shape[0]
                out = np.empty((len(origins), k, self.depth), dtype=np.float32)
            if f.shape[0] != out.shape[1]:
                f = _harmonise(f, out.shape[1])
            out[i] = f
            if progress_every and (i + 1) % progress_every == 0:
                print(f"    {i + 1}/{len(origins)}", flush=True)
        return out

    def component_count(self, x: np.ndarray, origins, n_probe: int = 32) -> int:
        return probe_component_count(x, origins, self.decomposer,
                                     self.W, n_probe)

    # ---------------------------------------------------------------- guard
    def verify(self, x: np.ndarray, t: int, n_trials: int = 2) -> None:
        """
        Assert that this builder behaves as its class claims.

        A causal builder must ignore x[t+1:]. A leaky builder must NOT — if it
        somehow passed the causality test it would not be reproducing the
        pipeline this study is auditing, and the comparison would be empty.
        """
        probe = lambda series, origin: self.at_origin(series, origin)  # noqa: E731
        if self.is_causal:
            assert_pipeline_causal(probe, x, t, n_trials=n_trials)
            return True

        if self.arm == "partition":
            # A partition builder passes the plain causality test whenever t
            # sits in the last split, and fails it whenever t sits in an
            # earlier one — so that test says nothing useful here. The regime
            # is defined by two conditions instead, and both are asserted:
            #
            #   (a) insensitive to samples OUTSIDE its own block
            #       -> train->test contamination really is gone
            #   (b) sensitive to samples AFTER t INSIDE its own block
            #       -> the look-ahead the remedy does not remove
            #
            # This pair is the executable definition of the regime; Section
            # 4.1 of the paper quotes it.
            if not getattr(self.decomposer, "can_leak", True):
                assert_pipeline_causal(probe, x, t, n_trials=n_trials)
                return True
            lo, hi = self.block_of(len(x), t)
            assert_block_isolated(probe, x, t, lo, hi, n_trials=n_trials)
            # Scale the "is there enough future to test?" threshold to the
            # block. A fixed 1024 is right for the real 7775-sample test block
            # but silently skips the whole check on the short series the
            # precompute guard uses.
            #
            # The return value matters and is passed up: an origin near the end
            # of its block legitimately has nothing to look ahead at, so the
            # skip is correct there — but if EVERY sampled origin skips, the
            # leak half was never checked and verify_sample must say so rather
            # than report success.
            return assert_leaks_within(
                probe, x, t, hi, n_trials=n_trials,
                min_future=min(1024, max(32, (hi - lo) // 8)))

        if not getattr(self.decomposer, "can_leak", True):
            # 'none' has no channel to leak through: decomposing the whole
            # series with the identity and slicing it gives exactly the window.
            # The two arms therefore coincide here, and demanding leakage would
            # be demanding the impossible.
            #
            # So the check inverts. This builder must be CAUSAL, and the fact
            # is load-bearing: 'none' is the reference every method is scored
            # against, and a reference that moved between arms would make the
            # whole comparison unreadable.
            assert_pipeline_causal(probe, x, t, n_trials=n_trials)
            return True

        try:
            assert_pipeline_causal(probe, x, t, n_trials=n_trials)
        except AssertionError:
            return True                             # expected: it leaks
        raise AssertionError(
            f"{type(self).__name__} passed the causality test. It is supposed "
            f"to leak — it reproduces the paradigm under audit. Check that it "
            f"really decomposes the full series."
        )

    def verify_sample(self, x: np.ndarray, origins, n: int = 3,
                      seed: int = 0, max_len: int | None = None) -> None:
        """
        Check the contract on `n` origins.

        max_len TRUNCATES THE SERIES FIRST, AND THAT IS THE POINT
            The guard works by permuting parts of the series and rebuilding the
            features, so every trial is a fresh decomposition of a block that
            has never been seen and can never be cached. On the real record a
            partition block is 35502 samples; a ceemdan pass over it takes
            about 1266 seconds, the guard runs eight of them per group, and
            precompute has 72 groups. That is 2.8 hours per group of pure
            verification — measured, on job 6288323, which sat on
            ceemdan|W1024|st3 for ten minutes without writing a chunk and would
            have sat there for hours.

            The contract being checked is structural: does this builder read
            outside its block, and does it read ahead inside it. Neither answer
            depends on the block being 35502 samples long. `block_of` and
            `valid_origins` both derive their boundaries from len(x), so a
            truncated series produces proportionally smaller blocks and
            exercises exactly the same code path in seconds.

            It is still real data, and still the real decomposer with the real
            tuned parameters. Only the length changes.
        """
        if max_len is not None and len(x) > max_len:
            x = np.asarray(x[:max_len])
            origins = valid_origins(len(x), self.W, "train")
        rng = np.random.RandomState(seed)
        origins = np.asarray(origins)
        if len(origins) == 0:
            raise ValueError(
                f"no origins to verify on: max_len={max_len} is too short for "
                f"W={self.W}")
        exercised = 0
        for t in rng.choice(origins, size=min(n, len(origins)), replace=False):
            exercised += bool(self.verify(x, int(t)))
        if not exercised:
            raise AssertionError(
                f"{type(self).__name__}: every sampled origin skipped the "
                f"look-ahead check (W={self.W}, series {len(x)}). The guard "
                f"reported success without testing the half that matters. "
                f"Sample more origins, or lengthen the series.")

    def describe(self) -> dict:
        d = {"arm": self.arm, "is_causal": self.is_causal,
             "W": self.W, "depth": self.depth}
        if hasattr(self.decomposer, "describe"):
            d.update(self.decomposer.describe())
        return d

    def signature(self) -> str:
        return cache_signature(
            f"{self.arm}:{getattr(self.decomposer, 'name', 'unknown')}",
            getattr(self.decomposer, "params", {}), self.W)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.decomposer!r}, W={self.W}, "
                f"depth={self.depth})")


# =============================================================================
# SHARED BLOCK CACHE  (leaky and partition arms)
# =============================================================================
# WHY THIS EXISTS -- measured, not anticipated
#     The leaky and partition builders decompose one large block and then slice
#     every origin out of it. Each builder memoises that block in memory, which
#     is correct for one process and useless across a node: `ledger.claim()` has
#     no group affinity, so N workers are handed consecutive units of the SAME
#     group and each decomposes the SAME block independently.
#
#     On hamsi66 with 56 workers and ceemdan over a 35502-sample partition
#     block, that produced 34 finished chunks in 7.4 hours out of 1872 -- an
#     extrapolated 400 hours for a job with a 72-hour limit. Three workers were
#     seen holding chunks 1, 2 and 3 of the identical group for 7 hours each.
#     The same shape of failure OOM-killed 20 chunks of `partition|vmd|W1024`
#     on barbun8: 40 workers x ~2.8 GB of vmdpy state at once.
#
#     There is a third copy of the same waste: the block depends on the split
#     and the depth, NOT on W, but the feature cache is keyed by W -- so every
#     block was decomposed once more for each of the three window sizes.
#
# WHAT THIS DOES
#     Memoise the decomposition of a block on disk, keyed by a hash of the
#     block's contents and the decomposition's identity. A worker that finds
#     the file loads it. A worker that does not takes an exclusive lock, so the
#     other 55 wait for the result instead of recomputing it -- which caps peak
#     memory at one decomposition per node rather than N.
#
#     The key is a content hash, so it is correct across arms, stations and W
#     without any of them appearing in it: identical input, identical
#     parameters, identical output. Nothing about the numbers changes; only how
#     many times they are computed.
BLOCK_CACHE_DIRNAME = "_blocks"
_BLOCK_LOCK_POLL = 5.0            # seconds between checks while another worker works

# A lock is only meaningful while its holder is alive. SIGKILL — an OOM kill,
# `scancel`, a wall-clock stop — leaves the file behind with no result beside
# it, and the first version then waited SIX HOURS on a process that no longer
# existed. That is exactly what happened: job 6287133 was cancelled mid-block
# and the next job sat at 78/1872 for half an hour.
#
# Two independent escapes, because either alone can fail. The age check covers
# a holder on another node, whose pid means nothing here. The pid check
# catches a same-node death long before the age threshold.
_BLOCK_LOCK_STALE = 45 * 60.0     # older than the slowest block (~13 min) x3
_BLOCK_LOCK_TIMEOUT = 90 * 60.0   # absolute cap on waiting for a live holder


def _lock_is_dead(lock: str) -> bool:
    """True if the lock's holder is provably gone, or it is simply too old."""
    try:
        age = time.time() - os.path.getmtime(lock)
    except OSError:
        return False                       # vanished: the holder finished
    if age > _BLOCK_LOCK_STALE:
        return True
    try:
        with open(lock, encoding="utf-8") as f:
            pid, _, host = (f.read().split() + ["", ""])[:3]
        if host and host != platform.node():
            return False                   # another node; only age can judge
        os.kill(int(pid), 0)               # signal 0: existence check only
        return False
    except (ProcessLookupError, ValueError, OSError):
        return True
    except PermissionError:
        return False                       # alive, just not ours to signal


def _steal_lock(lock: str) -> bool:
    """Remove an abandoned lock. False if someone beat us to it."""
    try:
        os.remove(lock)
        return True
    except OSError:
        return False


def _block_key(decomposer, block: np.ndarray) -> str:
    """Content hash of (decomposition identity, block values)."""
    h = hashlib.sha1()
    h.update(getattr(decomposer, "name", type(decomposer).__name__).encode())
    h.update(json.dumps(getattr(decomposer, "params", {}), sort_keys=True,
                        default=str).encode())
    h.update(np.ascontiguousarray(block, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _block_cache_dir() -> str:
    from config import CACHE_DIR
    return os.path.join(CACHE_DIR, BLOCK_CACHE_DIRNAME)


def decompose_block_cached(decomposer, block: np.ndarray) -> np.ndarray:
    """
    `decomposer(block)`, computed at most once per node instead of once per
    worker. Falls back to a plain call if anything about the cache misbehaves —
    a broken cache must slow the run down, never change its results.
    """
    try:
        root = _block_cache_dir()
        os.makedirs(root, exist_ok=True)
        key = _block_key(decomposer, block)
    except Exception:                                          # noqa: BLE001
        return np.asarray(decomposer(block), dtype=np.float32)

    path = os.path.join(root, f"{key}.npy")
    lock = os.path.join(root, f"{key}.lock")

    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:                                      # noqa: BLE001
            pass          # truncated or half-written by an older run; redo it

    # O_EXCL is the whole mechanism: exactly one worker creates the lock and
    # computes; everyone else falls into the wait loop below.
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.time()} {platform.node()}\n"
                 .encode())
        os.close(fd)
        mine = True
    except FileExistsError:
        # Before queueing behind it, check the holder is actually alive.
        if _lock_is_dead(lock) and _steal_lock(lock):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time()} "
                         f"{platform.node()}\n".encode())
                os.close(fd)
                mine = True
            except FileExistsError:
                mine = False               # someone else claimed it first
        else:
            mine = False
    except Exception:                                          # noqa: BLE001
        return np.asarray(decomposer(block), dtype=np.float32)

    if not mine:
        waited = 0.0
        while waited < _BLOCK_LOCK_TIMEOUT:
            if os.path.exists(path):
                try:
                    return np.load(path)
                except Exception:                              # noqa: BLE001
                    break
            if not os.path.exists(lock):
                break                     # holder finished; recheck the file
            if _lock_is_dead(lock):
                _steal_lock(lock)
                break                     # abandoned; compute it ourselves
            time.sleep(_BLOCK_LOCK_POLL)
            waited += _BLOCK_LOCK_POLL
        return np.asarray(decomposer(block), dtype=np.float32)

    try:
        comps = np.asarray(decomposer(block), dtype=np.float32)
        from atomicio import write_npy
        write_npy(path, comps)
        return comps
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


class FeatureBuilder(WindowFeatureBuilder):
    """
    Causal arm: at origin t, decompose only x[t-W+1 .. t].

        builder = FeatureBuilder(get_decomposer("dwt", params), W=512)
        builder.verify(x, t=30000)             # raises if the future leaks
        block = builder.at_origin(x, 30000)    # (K, depth)
    """

    arm = "causal"
    is_causal = True

    def at_origin(self, x: np.ndarray, t: int) -> np.ndarray:
        return features_at_origin(x, t, self.decomposer, self.W, self.depth)


class LeakyFeatureBuilder(WindowFeatureBuilder):
    """
    Leaky arm: decompose the WHOLE series once, then slice the window out of
    the result — the pipeline this study audits, and the one used by most of
    the decomposition-forecasting literature.

    This is deliberately incorrect. It exists so that the two arms differ in
    exactly one respect, which is the only way to attribute the difference in
    results to leakage rather than to some incidental change in architecture,
    tuning or data handling.

    The full-series decomposition is computed once and cached in memory,
    keyed by the identity of the array, because it is the same for every
    origin.
    """

    arm = "leaky"
    is_causal = False

    def __init__(self, decomposer, W: int, depth: int = LOOK_BACK_MAX):
        super().__init__(decomposer, W, depth)
        self._cache_key = None
        self._components = None

    def _full(self, x: np.ndarray) -> np.ndarray:
        key = (id(x), len(x), float(x[0]), float(x[-1]))
        if self._cache_key != key:
            # Shared across workers: see decompose_block_cached. Without it,
            # every worker on the node repeats the whole-series decomposition.
            self._components = decompose_block_cached(self.decomposer, x)
            self._cache_key = key
        return self._components

    def at_origin(self, x: np.ndarray, t: int) -> np.ndarray:
        comps = self._full(x)                    # <-- sees the entire series
        lo = t - self.depth + 1
        if lo < 0:
            raise ValueError(f"origin {t} has less than depth={self.depth}")
        return comps[:, lo: t + 1]


class PartitionFeatureBuilder(WindowFeatureBuilder):
    """
    Partition arm: decompose each chronological split ONCE, in isolation, then
    slice the window out of that split's result.

    This is the remedy the literature adopts against leakage — "split first,
    then decompose". It is included here not because it is correct but because
    the paper's central claim is about how much of the leakage it removes, and
    that number has to be measured rather than assumed.

    What it removes and what it leaves:

        removed : train -> test contamination. The coefficients a test origin
                  reads were computed without a single training sample.
        LEFT    : look-ahead inside the origin's own split. The coefficient at
                  time t is still a function of x[t+1], x[t+2], ... as long as
                  those samples fall in the same split — and for a one-step
                  forecast that is exactly the information that matters.

    AT TEST ORIGINS IT REMOVES NO LOOK-AHEAD AT ALL, AND THIS IS DEDUCTIVE
        The splits are chronological and test is the last one, so both the
        full series and the test block end at n. For a test origin t the
        future visible to the decomposition is [t+1, n) in BOTH regimes — the
        same set of samples, not merely a similar amount. What per-partition
        changes at test time is how much PAST the decomposition conditions on,
        and past is not leakage.

        So the middle row of the paper's table is not an empirical surprise;
        it is forced by the design, and the experiment confirms a proof rather
        than discovering a fact. The pilot's -0.5% for DWT is what that looks
        like when measured.

        The arms do differ in the train and validation blocks, where the
        remedy does remove contamination from later splits. That changes what
        the model LEARNS and which configuration is SELECTED. It does not
        change the reported test score, which is where the inflation lives.

    The whole training block is decomposed at once, so a training origin's
    coefficients are a function of samples after it too. That is not a defect
    in this class — it is the definition of the protocol under audit. This
    reproduces it faithfully; it does not repair it.

    So this builder is causal with respect to the split boundary and leaky with
    respect to time. `verify()` asserts both halves; see the class docstring of
    WindowFeatureBuilder and the `partition` branch there.

    Like the leaky builder, the decomposition is computed once per split and
    memoised, so this arm costs three decompositions per station rather than
    one per origin.

    THE WARM-UP PREFIX — read this before changing anything here
        The first origin of the validation split is the split's own first
        sample, and a model needs `look_back` samples of history to form an
        input. A block containing only the split therefore cannot serve its own
        first `depth - 1` origins. Every real implementation of split-then-
        decompose faces this and resolves it the same way: the block starts
        `depth - 1` samples before the split boundary.

        We do the same, explicitly, and report it. The cost is that at most
        `depth - 1` = 167 samples of the preceding split enter the block —
        about 2% of a validation or test block. The alternative, stitching each
        early window together from two separate decompositions, would put a
        discontinuity in the middle of the model's input, which is worse.

        The warm-up is used ONLY to make the window exist. Origins are
        unchanged, so all three arms are still evaluated on exactly the same
        forecast origins, which is what makes them comparable.
    """

    arm = "partition"
    is_causal = False

    def __init__(self, decomposer, W: int, depth: int = LOOK_BACK_MAX):
        super().__init__(decomposer, W, depth)
        self._cache_key = None
        self._components = None

    # ------------------------------------------------------------- internals
    @staticmethod
    def split_of(n: int, t: int):
        """
        (lo, hi) of the split containing origin t, as a half-open range.

        The boundaries come from split_bounds, the same function every other
        stage uses, so this arm cannot drift away from the splits the models
        are actually trained and evaluated on.
        """
        train_end, val_end = split_bounds(n)
        if t < train_end:
            return 0, train_end
        if t < val_end:
            return train_end, val_end
        return val_end, n

    def block_of(self, n: int, t: int):
        """(block_lo, hi) — the split containing t, plus the warm-up prefix."""
        lo, hi = self.split_of(n, t)
        return max(0, lo - (self.depth - 1)), hi

    def _components_for(self, x: np.ndarray, lo: int, hi: int) -> np.ndarray:
        key = (id(x), len(x), lo, hi, float(x[lo]), float(x[hi - 1]))
        if self._cache_key != key:
            block = np.asarray(x[lo:hi], dtype=float)
            # Shared across workers AND across W: the block is fixed by the
            # split and the depth, so the three window sizes ask for exactly
            # the same decomposition. See decompose_block_cached.
            self._components = decompose_block_cached(self.decomposer, block)
            self._cache_key = key
        return self._components

    # ------------------------------------------------------------- contract
    def at_origin(self, x: np.ndarray, t: int) -> np.ndarray:
        lo, hi = self.block_of(len(x), int(t))
        comps = self._components_for(x, lo, hi)  # <-- sees this block only
        j = int(t) - lo                          # index of t inside the block
        start = j - self.depth + 1
        if start < 0:
            raise ValueError(
                f"origin {t} sits {j} samples into its block, less than "
                f"depth={self.depth}. The warm-up prefix should have made "
                f"this impossible; the split boundaries and the depth have "
                f"gone out of step."
            )
        return comps[:, start: j + 1]


BUILDERS = {"causal": FeatureBuilder,
            "leaky": LeakyFeatureBuilder,
            "partition": PartitionFeatureBuilder}


def get_builder(arm: str, decomposer, W: int,
                depth: int = LOOK_BACK_MAX) -> WindowFeatureBuilder:
    if arm not in BUILDERS:
        raise ValueError(f"unknown arm: {arm}. Known: {sorted(BUILDERS)}")
    return BUILDERS[arm](decomposer, W, depth)


def assert_pipeline_causal(build_fn, x: np.ndarray, t: int,
                           n_trials: int = 3, atol: float = 1e-5,
                           seed: int = 0) -> None:
    """
    End-to-end guard.

    `assert_causal` above only sees information that flows through the
    `window` argument, so it cannot catch a pipeline that bypasses
    `features_at_origin` — for example one that decomposes the full series and
    then indexes into it. This function tests the whole path instead.

        build_fn(series, origin) -> feature array

    Any code path that turns a series and an origin into features must pass
    this test. The precompute stage runs it before writing a cache.
    """
    rng = np.random.RandomState(seed)
    base = np.asarray(build_fn(x, t))

    for trial in range(n_trials):
        x2 = np.array(x, dtype=float, copy=True)
        tail = x2[t + 1:]
        if len(tail) > 1:
            x2[t + 1:] = rng.permutation(tail)
        again = np.asarray(build_fn(x2, t))

        if again.shape != base.shape:
            raise AssertionError(
                f"CAUSALITY VIOLATED: feature shape changed when the future "
                f"was permuted (origin {t}, trial {trial})"
            )
        if not np.allclose(base, again, atol=atol, rtol=0.0):
            worst = float(np.max(np.abs(base - again)))
            raise AssertionError(
                f"CAUSALITY VIOLATED in pipeline at origin {t} "
                f"(trial {trial}): features changed by up to {worst:.3e} when "
                f"only x[t+1:] was permuted."
            )


def assert_block_isolated(build_fn, x: np.ndarray, t: int, lo: int, hi: int,
                          n_trials: int = 3, atol: float = 1e-5,
                          seed: int = 0) -> None:
    """
    Assert that features at origin t ignore every sample outside [lo, hi).

    This is half of the partition regime's definition: whatever else the
    builder does, it must not read another split. Permuting everything outside
    the block must leave the features bit-for-bit alone.

    Note this is a strictly weaker claim than causality — it says nothing about
    x[t+1:hi], which the partition arm is allowed (and expected) to use.
    """
    rng = np.random.RandomState(seed)
    base = np.asarray(build_fn(x, t))

    for trial in range(n_trials):
        x2 = np.array(x, dtype=float, copy=True)
        outside = np.concatenate([np.arange(0, lo), np.arange(hi, len(x))])
        if len(outside) > 1:
            x2[outside] = rng.permutation(x2[outside])
        again = np.asarray(build_fn(x2, t))

        if again.shape != base.shape:
            raise AssertionError(
                f"BLOCK ISOLATION VIOLATED: feature shape changed when data "
                f"outside [{lo}, {hi}) was permuted (origin {t}, trial "
                f"{trial})"
            )
        if not np.allclose(base, again, atol=atol, rtol=0.0):
            worst = float(np.max(np.abs(base - again)))
            raise AssertionError(
                f"BLOCK ISOLATION VIOLATED at origin {t} (trial {trial}): "
                f"features changed by up to {worst:.3e} when only samples "
                f"outside the block [{lo}, {hi}) were permuted. This builder "
                f"is reading another split."
            )


def assert_leaks_within(build_fn, x: np.ndarray, t: int, hi: int,
                        n_trials: int = 3, atol: float = 1e-5,
                        seed: int = 0, min_future: int = 1024) -> bool:
    """
    Assert that features at origin t DO depend on x[t+1 : hi].

    The other half of the partition regime's definition, and the reason the
    arm exists: splitting before decomposing does not stop the coefficient at
    time t from being computed out of samples that come after t. If this
    assertion ever passed silently, the arm would have quietly become causal
    and the middle row of the paper's main table would be measuring nothing.

    Deliberately a POSITIVE claim about a defect, in the same spirit as
    LeakyFeatureBuilder.verify().

    Returns True if the claim was tested, False if the origin was skipped.

    WHY THE SKIP EXISTS
        The last origins of a split have almost no future left inside it: the
        final train origin sits max_horizon = 24 samples from the boundary.
        There the partition arm genuinely has nearly nothing to look ahead at,
        and it converges towards the causal arm — which is a property of the
        regime, not a bug in this code, and asserting leakage there would be
        asserting something false.

        min_future is set above the largest W so that any origin actually
        tested has more future inside its block than the causal arm has past.
    """
    if hi - (t + 1) < max(2, min_future):
        return False              # too close to the block's end to be tested

    rng = np.random.RandomState(seed)
    base = np.asarray(build_fn(x, t))

    for trial in range(n_trials):
        x2 = np.array(x, dtype=float, copy=True)
        x2[t + 1: hi] = rng.permutation(x2[t + 1: hi])
        again = np.asarray(build_fn(x2, t))
        if again.shape != base.shape:
            return True                             # expected: it responded
        if not np.allclose(base, again, atol=atol, rtol=0.0):
            return True                             # expected: it leaks

    raise AssertionError(
        f"partition builder at origin {t} ignored x[{t + 1}:{hi}] — it did "
        f"not leak within its own block. Either the decomposition has finite "
        f"support that does not reach t (check the window), or this builder "
        f"is not decomposing the block as a whole. Without within-block "
        f"look-ahead this arm is indistinguishable from the causal arm."
    )
