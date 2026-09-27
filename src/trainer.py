# -*- coding: utf-8 -*-
"""
Training loop with epoch-level checkpointing.
=============================================

The ledger in `pipeline.Stage` resumes at unit granularity — a killed job
restarts the unit it was working on. For final training a unit is one model
trained for up to 100 epochs, which can take hours, so unit-level resume alone
would throw away a lot of work.

`Trainer` therefore checkpoints after every epoch: model weights, optimiser
state, epoch counter, best score and early-stopping patience. Restarting picks
up at the next epoch. Checkpoints are written to a temporary file and renamed,
so a crash during the write cannot corrupt the previous one.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomicio import write_with


@dataclass
class TrainState:
    """Everything needed to resume, kept in one place so nothing is forgotten."""

    epoch: int = 0
    best_val: float = float("inf")
    best_epoch: int = -1
    patience_left: int = 0
    history: list = field(default_factory=list)
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainState":
        s = cls()
        s.__dict__.update(d)
        return s


def smoothed_validation_minimum(history, window: int = 3):
    """
    Minimum of the validation curve after a moving average. Returns (value, epoch).

    Why not the raw minimum
    -----------------------
    The raw minimum over N noisy epoch estimates is optimistically biased: the
    more epochs you look at, the lower it goes, whether or not the model is any
    better. Cawley & Talbot (JMLR 2010) make the general case — in model
    selection, low variance in the selection criterion matters as much as
    unbiasedness, because variance alone is enough to over-fit the selection.

    This project saw it directly. On one station, DWT reached a lower validation
    minimum than the raw signal (0.01689 vs 0.01747) and then lost on the test
    set (RMSE 0.5167 vs 0.5068). Its advantage, 0.00058, was smaller than the
    spread among its own three best epochs, 0.00081 — it had won on noise.

    Smoothing before taking the minimum is established practice for exactly this
    reason: a configuration must hold its performance for a few epochs rather
    than touch a good value once. The window is the only parameter, and it is
    reported.

    Note what this does NOT change: early stopping and the saved checkpoint
    still use the true per-epoch minimum, so the model that gets evaluated is
    the best one that existed. Only the number the *search* compares is
    smoothed.
    """
    vals = [float(h["val"]) for h in history]
    if not vals:
        return float("inf"), 0
    if len(vals) < window or window <= 1:
        i = int(np.argmin(vals))
        return vals[i], i + 1

    kernel = np.ones(window) / window
    smooth = np.convolve(np.asarray(vals), kernel, mode="valid")
    i = int(np.argmin(smooth))
    # 'valid' drops the edges; the centre of window i is epoch i + window//2 + 1
    return float(smooth[i]), i + window // 2 + 1


class Trainer:
    """
    Trains one model. Resumable, deterministic given a seed.

    Usage
    -----
        trainer = Trainer(model, device=DEVICE, checkpoint_path=ckpt)
        result = trainer.fit(train_loader, val_loader,
                             epochs=100, patience=15, lr=1e-3)
        preds = trainer.predict(test_loader)
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 checkpoint_path: Optional[str] = None,
                 seed: int = 0, verbose: bool = True,
                 selection_window: int = 1):
        self.model = model.to(device)
        self.accum = 1          # batch chunks; raised only after a CUDA OOM
        self.nonfinite_batches = 0   # chunks skipped for a NaN/inf loss
        self.selection_window = selection_window
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.seed = int(seed)
        self.verbose = verbose
        self.state = TrainState()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion = nn.MSELoss()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    # ---------------------------------------------------------- checkpoints
    def save_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "state": self.state.to_dict(),
            "seed": self.seed,
        }
        # A private temp name matters here even with one writer per unit:
        # a resubmitted array task can overlap with the job it is replacing.
        write_with(self.checkpoint_path, lambda p: torch.save(payload, p))

    def load_checkpoint(self) -> bool:
        """Restore a previous run. Returns True if anything was restored."""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return False
        payload = torch.load(self.checkpoint_path, map_location=self.device,
                             weights_only=False)
        self.model.load_state_dict(payload["model"])
        if self.optimizer is not None and payload.get("optimizer"):
            self.optimizer.load_state_dict(payload["optimizer"])
        self.state = TrainState.from_dict(payload["state"])
        if self.verbose:
            print(f"  resumed from epoch {self.state.epoch} "
                  f"(best val {self.state.best_val:.5f})", flush=True)
        return True

    def _best_path(self) -> Optional[str]:
        return (self.checkpoint_path.replace(".ckpt", ".best.ckpt")
                if self.checkpoint_path else None)

    def _save_best(self) -> None:
        p = self._best_path()
        if not p:
            return
        payload = {"model": self.model.state_dict(),
                   "epoch": self.state.epoch,
                   "val": self.state.best_val}
        write_with(p, lambda q: torch.save(payload, q))

    def _load_best(self) -> None:
        p = self._best_path()
        if p and os.path.exists(p):
            payload = torch.load(p, map_location=self.device, weights_only=False)
            self.model.load_state_dict(payload["model"])

    # ----------------------------------------------------------------- loops
    def _run_epoch(self, loader, train: bool) -> float:
        """
        One pass over a loader, surviving GPU memory exhaustion.

        The hazard is structural, not incidental: the search runs on CPU
        (DEVICE_OPTUNA), where there is no VRAM ceiling, and final training runs
        on a GPU. So Optuna can hand back a configuration — a wide transformer
        at look_back=168 and batch 128 — that simply does not fit on the card,
        and the failure surfaces hours later in a different stage.

        Rather than fail, the batch is split and gradients accumulated. The
        *effective* batch size is unchanged, so the model being trained is still
        the one whose hyperparameters were selected and recorded; only the
        arithmetic is chunked. That distinction matters: silently halving the
        batch would mean reporting one configuration and training another.

        Recovery also empties the allocator's cache. Without that, the memory
        left fragmented by the failure makes the next attempt — and every later
        unit claimed by this same worker — more likely to fail too.
        """
        while True:
            try:
                return self._epoch_pass(loader, train)
            except torch.cuda.OutOfMemoryError:
                if not train or self.accum >= self.MAX_ACCUM:
                    self._free_gpu()
                    raise
                self.accum *= 2
                self._free_gpu()
                print(f"  CUDA out of memory: splitting each batch into "
                      f"{self.accum} chunks (effective batch size unchanged) "
                      f"and retrying the epoch", flush=True)

    MAX_ACCUM = 16          # beyond this the configuration is simply too large

    def _free_gpu(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _epoch_pass(self, loader, train: bool) -> float:
        self.model.train(train)
        total, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                seen = 0            # finite samples contributing to this step
                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                # accum == 1 is the ordinary path: one chunk, one step.
                chunks = max(1, int(np.ceil(len(xb) / self.accum)))
                batch_loss = 0.0
                for xc, yc in zip(xb.split(chunks), yb.split(chunks)):
                    xc = xc.to(self.device, non_blocking=True)
                    yc = yc.to(self.device, non_blocking=True)
                    out = self.model(xc)
                    loss = self.criterion(out, yc)

                    # A non-finite loss must never reach the optimiser.
                    # clip_grad_norm_ computes a NaN norm from NaN gradients
                    # and scales by it, so the step writes NaN into every
                    # weight and the model stays poisoned for the rest of the
                    # run — which is what "train nan val nan" from some epoch
                    # onwards means. Optuna then rejects the trial outright
                    # ("The value nan is not acceptable") and the whole
                    # configuration is lost rather than scored. Skipping the
                    # offending chunk keeps the model alive; a configuration
                    # that produces nothing but NaN is caught below.
                    if not torch.isfinite(loss):
                        self.nonfinite_batches += 1
                        continue

                    if train:
                        # Scale so the accumulated gradient equals the one a
                        # single undivided batch would have produced.
                        (loss * (len(xc) / len(xb))).backward()
                    batch_loss += float(loss.item()) * len(xc)
                    seen += len(xc)

                if train and seen:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()

                total += batch_loss
                n += seen
        if n == 0:
            # Every chunk was non-finite: the configuration diverged.
            return float("nan")
        return total / n

    def fit(self, train_loader, val_loader, epochs: int, patience: int,
            lr: float = 1e-3, weight_decay: float = 0.0,
            max_seconds: Optional[float] = None,
            on_epoch: Optional[Callable[[int, float, float], bool]] = None) -> dict:
        """
        Train with early stopping.

        `max_seconds` stops cleanly with the checkpoint intact so the SLURM
        wall clock never kills the process mid-write; the next run resumes.

        `on_epoch(epoch, train_loss, val_loss) -> bool` lets a caller (Optuna)
        request a stop, which is how pruning is wired in.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.load_checkpoint()
        if self.state.patience_left == 0 and self.state.epoch == 0:
            self.state.patience_left = patience

        t0 = time.time()
        stopped = "completed"

        while self.state.epoch < epochs:
            if max_seconds is not None and time.time() - t0 > max_seconds:
                stopped = "time_budget"
                break

            t_epoch = time.time()
            tr = self._run_epoch(train_loader, train=True)
            va = self._run_epoch(val_loader, train=False)
            self.state.epoch += 1

            # This epoch's time, not the time since fit() started. Adding the
            # latter every epoch accumulates a triangular sum: an 18-epoch run
            # at 47 s reported 8071 seconds instead of 855, nine times the
            # truth. Training times are reported in the paper and used to size
            # the cluster budget, so the number has to mean what it says.
            self.state.elapsed += time.time() - t_epoch
            self.state.history.append({"epoch": self.state.epoch,
                                       "train": tr, "val": va})

            improved = va < self.state.best_val - 1e-6
            if improved:
                self.state.best_val = va
                self.state.best_epoch = self.state.epoch
                self.state.patience_left = patience
                self._save_best()
            else:
                self.state.patience_left -= 1

            self.save_checkpoint()

            if self.verbose and (self.state.epoch % 5 == 0 or improved):
                print(f"    epoch {self.state.epoch:3d}  train {tr:.5f}  "
                      f"val {va:.5f}{'  *' if improved else ''}", flush=True)

            if on_epoch is not None and on_epoch(self.state.epoch, tr, va):
                stopped = "pruned"
                break
            if self.state.patience_left <= 0:
                stopped = "early_stop"
                break

        self._load_best()
        sel_val, sel_epoch = smoothed_validation_minimum(
            self.state.history, self.selection_window)
        return {"best_val": self.state.best_val,
                "best_epoch": self.state.best_epoch,
                # What the hyperparameter search compares. Separate from
                # best_val, which still decides the checkpoint.
                "selection_val": sel_val,
                "selection_epoch": sel_epoch,
                "selection_window": self.selection_window,
                "epochs_run": self.state.epoch,
                "stopped": stopped,
                "seconds": self.state.elapsed,
                "history": self.state.history}

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, loader) -> np.ndarray:
        self.model.eval()
        out = []
        for xb, _ in loader:
            out.append(self.model(xb.to(self.device)).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0,))

    def is_finished(self, epochs: int) -> bool:
        """True when a stored checkpoint already reached the target."""
        return (self.state.epoch >= epochs
                or self.state.patience_left <= 0) and self.state.epoch > 0
