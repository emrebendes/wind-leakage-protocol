# -*- coding: utf-8 -*-
"""
Configuration — leak-free wind speed forecasting benchmark.

Everything that defines the experimental protocol lives here so that a single
file documents what was run. Changes to this file invalidate caches; the cache
key records the relevant values (see causal_features.cache_signature).
"""

import os

# =============================================================================
# PATHS
# =============================================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data_files")
DATA_FILE = os.path.join(DATA_DIR, "all_station_data.npy")

# The record ends with 426 hours that are constant in every station, each at
# its own value, all starting at index 50718. That is gap filling, not
# measurement: a stretch of fabricated data sitting inside the test split,
# where a persistence baseline scores perfectly and every model's error is
# flattered. The series is therefore cut before it.
#
# Set to None to use the full record.
SERIES_END = 50_718


def load_series():
    """
    The measurement record, as a (station, hour) float array.

    Every stage loads the data through here so that the truncation above
    cannot be applied in one place and forgotten in another — a split
    boundary that differs between precompute and training would silently
    invalidate the cache.
    """
    import numpy as np
    raw = np.load(DATA_FILE, allow_pickle=True)
    x = np.asarray(raw, dtype=np.float64)
    return x[:, :SERIES_END] if SERIES_END else x

RESULTS_DIR = os.path.join(BASE_DIR, "results_causal")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models_causal")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
STATE_DIR = os.path.join(BASE_DIR, "state")          # job ledgers + optuna
CACHE_DIR = os.path.join(BASE_DIR, "causal_cache")   # per-origin features
PARAMS_DIR = os.path.join(BASE_DIR, "decomp_params") # tuned decomposition params

for _d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR,
           STATE_DIR, CACHE_DIR, PARAMS_DIR):
    os.makedirs(_d, exist_ok=True)

LEDGER_DB = os.path.join(STATE_DIR, "jobs.db")

# =============================================================================
# DEVICE
# =============================================================================
# torch is imported lazily so that `run.py status` and `run.py doctor` work in
# an environment where it is missing — diagnosing a missing dependency is
# exactly what doctor is for, and it cannot do that if importing config fails.
def _device(prefer_cuda: bool = True):
    import torch
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# A proxy object is not good enough here. `model.to(device)` and
# `tensor.to(device)` are parsed in C and accept only a real torch.device, so a
# wrapper that merely forwards attributes is rejected with an unhelpful
# TypeError — which is what happened: every Optuna trial died at `model.to()`.
#
# PEP 562 module-level __getattr__ gives the same laziness without a proxy.
# `from config import DEVICE` resolves through it and yields a genuine
# torch.device, while `run.py doctor` — which touches DATA_FILE and the
# directories but never DEVICE — still imports config without torch present.
_DEVICE_CACHE = {}


def __getattr__(name):
    if name in ("DEVICE", "DEVICE_OPTUNA"):
        if name not in _DEVICE_CACHE:
            _DEVICE_CACHE[name] = _device(prefer_cuda=(name == "DEVICE"))
        return _DEVICE_CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# =============================================================================
# DATA / SPLIT
# =============================================================================
NUM_STATIONS = 8
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15          # test = remaining 0.15
SEED = 42

# =============================================================================
# FORECAST TASK
# =============================================================================
FORECAST_HORIZONS = [1, 8, 16, 24]
MAX_HORIZON = max(FORECAST_HORIZONS)

# The model reads LOOK_BACK steps of each component. Tuned by Optuna.
LOOK_BACK_RANGE = (24, 168)   # widened: old runs clustered in the upper half
LOOK_BACK_MAX = LOOK_BACK_RANGE[1]   # cache is stored at this depth and sliced

# =============================================================================
# CAUSAL DECOMPOSITION WINDOW
# =============================================================================
# W is the trailing history handed to the decomposition at each forecast
# origin. It is a hyperparameter selected on validation, never on test.
# Lower bound is principled: a level-L DWT needs >= ~2^L * filter_support
# samples for the coarsest scale to be meaningful.
W_CHOICES = [256, 512, 1024]
W_DEFAULT = 512

# =============================================================================
# DECOMPOSITION METHODS
# =============================================================================
DECOMPOSITION_METHODS = ["none", "emd", "eemd", "ceemdan", "vmd", "dwt"]

# The two arms of the study. They share every line of code except the moment
# the decomposition is computed, which is the whole point: any difference in
# the results is then attributable to that timing alone.
#   causal : decompose x[t-W+1..t] at each forecast origin
#   leaky  : decompose the whole series once, then slice   (the audited paradigm)
ARMS = ["causal", "leaky"]

# The third regime: decompose each chronological split on its own, then slice.
# This is the remedy the literature adopts against leakage, and the paper's
# claim is about how much of the leakage it actually removes.
#
# It is deliberately NOT in ARMS. ARMS is the default for `precompute`, and a
# parameterless `run.py precompute` must keep producing exactly the two arms
# the existing results were built from. The third arm is always asked for by
# name:
#
#     python run.py precompute --arms partition --methods dwt vmd
#
# ALL_ARMS is what the argument parsers accept, so `--arm partition` is legal
# everywhere without changing any default.
PARTITION_ARM = "partition"
ALL_ARMS = ARMS + [PARTITION_ARM]

# Decomposition parameter defaults and search spaces are NOT listed here.
# They belong to the method classes in decompositions.py, next to the code that
# uses them, so the two cannot drift apart:
#
#     from decompositions import default_params, param_space, min_window_for
#
# Each class also declares min_window(), which is enforced on every call — a
# level-6 DWT on a 128-sample window now raises instead of silently returning a
# degenerate coarsest scale.

# Number of training windows sampled when scoring decomposition parameters.
DECOMP_TUNE_WINDOWS = 200

# Composite decomposition quality score weights
# (envelope entropy, orthogonality index, energy conservation ratio)
DECOMP_SCORE_WEIGHTS = (1.0, 0.5, 0.3)

# =============================================================================
# MODELS
# =============================================================================
MODELS = ["lstm", "bilstm", "gru", "tcn", "tcan", "cnn_lstm", "transformer"]

# Shared search space
COMMON_SEARCH = {
    "lr": (1e-4, 5e-3),
    "batch_size": [32, 64, 128],
    "dropout": (0.0, 0.5),
    "look_back": LOOK_BACK_RANGE,
    "W": W_CHOICES,
}

# Architecture-specific search space.
# num_layers upper bound raised: in the previous study 5 of 36 best trials sat
# at the old ceiling of 6, all of them TCN/TCAN, whose receptive field grows
# with depth.
MODEL_SEARCH = {
    "lstm":        {"hidden_size": (24, 256), "num_layers": (1, 6)},
    "bilstm":      {"hidden_size": (24, 256), "num_layers": (1, 6)},
    "gru":         {"hidden_size": (24, 256), "num_layers": (1, 6)},
    "tcn":         {"num_channels": (16, 128), "num_layers": (2, 10),
                    "kernel_size": [2, 3, 5]},
    "tcan":        {"num_channels": (16, 128), "num_layers": (2, 10),
                    "kernel_size": [2, 3, 5], "att_dim": [16, 32, 64]},
    "cnn_lstm":    {"cnn_filters": (16, 128), "cnn_kernel": [2, 3, 5],
                    "lstm_hidden": (24, 256), "lstm_layers": (1, 3)},
    "transformer": {"d_model": [32, 64, 128, 256], "nhead": [2, 4, 8],
                    "num_layers": (1, 6), "dim_feedforward": (64, 512)},
}

# =============================================================================
# OPTIMISATION / TRAINING
# =============================================================================
# ---------------------------------------------------------------- search cost
# Trials run on a subsample of the training and validation origins. The full
# grid on all origins was measured at 12.6 hours per trial on one core, i.e.
# 45 days on a whole 52-core hamsi node — against a three-day wall clock.
#
# The previous study did the same thing (`OPT_DATA_LIMIT = 8760`, one year per
# station) and its final models were trained on everything, on a GPU. Only the
# search is cheap; nothing that is reported comes from a subsample.
#
# These are per station. Eight stations give ~49k training and ~10k validation
# origins per trial, which measured at 2.4 hours per trial.
#
# One difference from the old study: it took the first 8760 hours as a
# contiguous block. This draws the same number of origins spread over the whole
# training region, so the search sees every season instead of one year's.
OPTUNA_TRAIN_ORIGINS = 6_100      # per station, ~1 year after the 70% split
OPTUNA_VAL_ORIGINS = 1_300        # per station

# The search compares the minimum of a moving-averaged validation curve rather
# than the raw per-epoch minimum. The raw minimum over N noisy estimates is
# optimistically biased and its variance alone can decide a comparison: in the
# pilot, DWT's validation advantage over the raw signal (0.00058) was smaller
# than the spread among its own three best epochs (0.00081).
#
# Cawley & Talbot (JMLR 2010) is the general argument — in model selection, low
# variance of the criterion matters as much as unbiasedness. Set to 1 to
# recover the raw minimum.
#
# Early stopping and the saved checkpoint are unaffected; they still use the
# true per-epoch minimum, so the evaluated model is the best one that existed.
OPTUNA_SELECTION_WINDOW = 3

# How often a trial reports to the Optuna storage and asks the pruner.
#
# Reporting every epoch is the natural choice and works fine for a handful of
# workers. It does not survive several hundred: `trial.should_prune()` makes
# MedianPruner read every intermediate value of every trial in the study, so
# the read grows with the study and happens once per epoch per worker. With
# 576 workers on one SQLite file over a shared filesystem the result was 4000
# "database is locked" errors per job and 0.4% CPU efficiency — the workers
# spent their time queueing for the database, not training.
#
# Reporting every 4th epoch cuts that traffic fourfold. With OPTUNA_EPOCHS=20
# the pruner still sees five checkpoints per trial, which is enough for a
# median rule; the cost is that a doomed trial runs up to three extra epochs
# before it is cut.
OPTUNA_REPORT_EVERY = 4

OPTUNA_TRIALS = 50
OPTUNA_EPOCHS = 20            # surrogate; rank correlation vs full reported
OPTUNA_PATIENCE = 5
OPTUNA_SUBSET_YEARS = 1       # taken from the TRAINING region only

FINAL_EPOCHS = 100
FINAL_PATIENCE = 15
FINAL_SEEDS = [0, 1, 2]       # replicates for significance testing

# Number of forecast origins used. `None` means every valid origin.
# Sub-sampling is only applied to the EMD family, whose per-origin cost is
# prohibitive; the choice is reported in the paper.
MAX_TRAIN_ORIGINS = {
    "none": None, "dwt": None, "vmd": None,
    "emd": None, "eemd": 60000, "ceemdan": 40000,
}

# =============================================================================
# EVALUATION
# =============================================================================
METRICS = ["RMSE", "MAE", "MAPE", "R2", "DA", "nRMSE"]
USE_PERSISTENCE_BASELINE = True
BOOTSTRAP_SAMPLES = 1000
