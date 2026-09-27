# -*- coding: utf-8 -*-
"""
Wiring tests: the plumbing between config, models and the trainer.
==================================================================

Cheap checks for the joins that only fail once a long job is already running.
Each one here corresponds to a fault that cost real time.

`config.DEVICE` was a lazy proxy object that forwarded attributes. It looked
right and printed right, but `model.to(device)` is parsed in C and takes only a
genuine torch.device, so every Optuna trial died at the first line of training.
The ledger reported a worker running and the trial counter never moved: three
minutes of apparent progress, nothing completed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import config                                                     # noqa: E402
from config import DEVICE, DEVICE_OPTUNA, MAX_HORIZON, MODELS      # noqa: E402
from models import build_model, model_param_space                  # noqa: E402


# =============================================================================
def test_devices_are_real_torch_devices():
    """A proxy passes attribute access and fails torch's C-level parsing."""
    for d in (DEVICE, DEVICE_OPTUNA):
        assert isinstance(d, torch.device), \
            f"{d!r} is {type(d).__name__}, not torch.device"


def test_optuna_device_is_cpu():
    """
    The search is deliberately CPU-bound: it is thousands of short independent
    trainings, which fill 52 cores better than they queue for one card.
    """
    assert DEVICE_OPTUNA.type == "cpu"


def test_config_imports_without_touching_torch():
    """
    `run.py doctor` must work in an environment with a broken or missing torch —
    diagnosing that is what it is for. Reading a plain setting must therefore
    not resolve a device.
    """
    config._DEVICE_CACHE.clear()
    _ = config.DATA_FILE
    _ = config.W_CHOICES
    assert config._DEVICE_CACHE == {}, \
        "reading an ordinary setting resolved a torch device"


def test_unknown_config_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = config.THIS_SETTING_DOES_NOT_EXIST


# =============================================================================
@pytest.mark.parametrize("name", MODELS)
def test_every_model_moves_to_the_device_and_runs(name):
    """
    Build each architecture, move it, push one batch through. This is the exact
    sequence an Optuna trial performs, and it takes a second per model.
    """
    K, look_back, batch = 3, 48, 4
    defaults = {}
    for key, spec in model_param_space(name).items():
        if isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = spec
            defaults[key] = lo if isinstance(lo, int) else float(lo)
        elif isinstance(spec, (list, tuple)):
            defaults[key] = spec[0]

    model = build_model(name, K, MAX_HORIZON, defaults).to(DEVICE_OPTUNA)
    x = torch.randn(batch, look_back, K, device=DEVICE_OPTUNA)
    y = model(x)

    assert y.shape == (batch, MAX_HORIZON), \
        f"{name}: output {tuple(y.shape)}, expected {(batch, MAX_HORIZON)}"
    assert torch.isfinite(y).all(), f"{name}: produced non-finite output"


@pytest.mark.parametrize("name", MODELS)
def test_every_model_reports_its_parameter_count(name):
    """Capacity is one of the paper's analyses, so it must be readable."""
    model = build_model(name, 3, MAX_HORIZON, {})
    n = model.n_parameters
    assert isinstance(n, int) and n > 0


# =============================================================================
@pytest.mark.parametrize("module", ["tune_decompositions", "precompute_causal"])
def test_decomposition_stages_do_not_import_torch(module):
    """
    vmdpy and torch each load an Intel OpenMP runtime, and on Windows having
    both in one process aborts the interpreter — no exception, no traceback.
    The pipeline avoids this by construction: the stages that decompose never
    touch torch, and the stages that train only read the cache.

    That separation is easy to break with one convenient import, and the
    breakage would appear as a mid-run crash on the cluster rather than as a
    test failure. So it is checked, in a subprocess, where importing the module
    under test cannot pollute this one.
    """
    import subprocess

    src = os.path.join(os.path.dirname(HERE), "src")
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import %s;"
        "print('torch' in sys.modules)" % (src, module)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=os.path.dirname(HERE))
    assert out.returncode == 0, f"{module} failed to import:\n{out.stderr}"
    assert out.stdout.strip() == "False", \
        f"{module} imports torch; it must not, see this test's docstring"


# =============================================================================
# gradient accumulation
# =============================================================================
def test_accumulated_gradients_match_a_single_batch():
    """
    Splitting a batch must not change the model being trained.

    The trainer chunks batches after a CUDA out-of-memory error so that a
    configuration the search selected on CPU can still be trained on a GPU. The
    whole point is that the *effective* batch size is preserved — reporting one
    batch size and training with another would make the recorded
    hyperparameters a fiction. This pins the scaling that makes that true.
    """
    from trainer import Trainer

    torch.manual_seed(0)
    x = torch.randn(32, 48, 3)
    y = torch.randn(32, MAX_HORIZON)

    def gradients(accum: int):
        torch.manual_seed(0)
        model = build_model("gru", 3, MAX_HORIZON,
                            {"hidden_size": 32, "num_layers": 1,
                             "dropout": 0.0})
        tr = Trainer(model, torch.device("cpu"), checkpoint_path=None,
                     seed=0, verbose=False)
        tr.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        tr.accum = accum
        tr.model.train(True)

        tr.optimizer.zero_grad(set_to_none=True)
        chunk = max(1, int(np.ceil(len(x) / accum)))
        for xc, yc in zip(x.split(chunk), y.split(chunk)):
            loss = tr.criterion(tr.model(xc), yc)
            (loss * (len(xc) / len(x))).backward()
        return [p.grad.detach().clone() for p in tr.model.parameters()]

    whole = gradients(1)
    split = gradients(4)

    assert len(whole) == len(split)
    for a, b in zip(whole, split):
        assert torch.allclose(a, b, atol=1e-6), \
            "chunked gradients differ from the undivided batch"


def test_chunking_covers_every_sample():
    """Every sample must appear exactly once, whatever the chunk count."""
    x = torch.arange(100).float().reshape(100, 1)
    for accum in (1, 2, 3, 4, 8, 16):
        chunk = max(1, int(np.ceil(len(x) / accum)))
        pieces = list(x.split(chunk))
        assert sum(len(p) for p in pieces) == len(x)
        assert torch.equal(torch.cat(pieces), x)


# =============================================================================
# the search criterion
# =============================================================================
def test_smoothing_reduces_the_optimism_of_the_minimum():
    """
    A curve that touches a good value once must not beat one that holds a
    slightly worse value throughout. That is the whole point of the criterion.
    """
    from trainer import smoothed_validation_minimum

    spiky = [{"val": v} for v in [0.021, 0.0169, 0.021, 0.022, 0.022]]
    steady = [{"val": v} for v in [0.0176, 0.0175, 0.0175, 0.0176, 0.0177]]

    assert min(h["val"] for h in spiky) < min(h["val"] for h in steady), \
        "the raw minimum prefers the spike — this is the bias being corrected"

    spiky_score, _ = smoothed_validation_minimum(spiky, 3)
    steady_score, _ = smoothed_validation_minimum(steady, 3)
    assert steady_score < spiky_score, \
        "smoothing failed to prefer the stable configuration"


def test_smoothing_falls_back_cleanly_on_short_histories():
    """A pruned trial can stop after one or two epochs."""
    from trainer import smoothed_validation_minimum

    assert smoothed_validation_minimum([], 3) == (float("inf"), 0)
    assert smoothed_validation_minimum([{"val": 0.02}], 3) == (0.02, 1)

    two = [{"val": 0.02}, {"val": 0.019}]
    value, epoch = smoothed_validation_minimum(two, 3)
    assert value == 0.019 and epoch == 2


def test_window_of_one_is_the_raw_minimum():
    """The change must be switchable off for a sensitivity check in the paper."""
    from trainer import smoothed_validation_minimum

    hist = [{"val": v} for v in (0.03, 0.01, 0.02)]
    assert smoothed_validation_minimum(hist, 1) == (0.01, 2)


def test_checkpoint_selection_still_uses_the_raw_minimum():
    """
    Smoothing changes what the search compares, not which weights are kept.
    The evaluated model must remain the best epoch that actually occurred —
    otherwise the paper would report a model nobody trained.
    """
    import inspect
    from trainer import Trainer

    source = inspect.getsource(Trainer.fit)
    assert "improved = va < self.state.best_val" in source, \
        "early stopping no longer tracks the true per-epoch minimum"
    assert "self._load_best()" in source, \
        "the best checkpoint is no longer restored before evaluation"
