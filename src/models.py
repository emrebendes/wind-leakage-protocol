# -*- coding: utf-8 -*-
"""
Forecasting architectures.
==========================

Every model is a `BaseForecaster` subclass with the same contract:

        input   (batch, look_back, K)     K decomposition channels
        output  (batch, horizon)          direct multi-step forecast

The only thing that changed relative to the previous study is `input_size`:
it was 1 (one model per component, predictions summed) and is now K (one model,
components as channels). The architectures themselves are unchanged, so the
comparison with the earlier results stays interpretable.

Each class declares its own `param_space()`, next to the code that consumes it,
for the same reason the decomposition classes do: a search space kept in a
separate file drifts away from the model it describes.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# =============================================================================
class BaseForecaster(nn.Module, ABC):
    """Contract shared by every architecture in the benchmark."""

    name: str = "base"

    def __init__(self, input_size: int, horizon: int, params: Dict[str, Any]):
        super().__init__()
        self.input_size = int(input_size)
        self.horizon = int(horizon)
        self.params = dict(params)
        self.build()

    # ------------------------------------------------------------- contract
    @abstractmethod
    def build(self) -> None:
        """Create the layers. Called by __init__ after params are stored."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, look_back, K) -> (batch, horizon)"""

    @staticmethod
    @abstractmethod
    def param_space() -> Dict[str, Any]:
        """Search space, using the same conventions as the decompositions."""

    # ---------------------------------------------------------------- shared
    @property
    def n_parameters(self) -> int:
        """Used by the capacity analysis: does decomposition help small models more?"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "input_size": self.input_size,
                "horizon": self.horizon, "params": self.params,
                "n_parameters": self.n_parameters}

    def _check(self, x: torch.Tensor) -> None:
        if x.dim() != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"{self.name}: expected (batch, look_back, {self.input_size}), "
                f"got {tuple(x.shape)}")


# =============================================================================
# recurrent
# =============================================================================
class _RecurrentBase(BaseForecaster):
    rnn_cls = nn.LSTM
    bidirectional = False

    def build(self):
        h = int(self.params.get("hidden_size", 64))
        layers = int(self.params.get("num_layers", 2))
        drop = float(self.params.get("dropout", 0.1))
        self.rnn = self.rnn_cls(
            input_size=self.input_size, hidden_size=h, num_layers=layers,
            batch_first=True, bidirectional=self.bidirectional,
            dropout=drop if layers > 1 else 0.0,
        )
        self.head = nn.Linear(h * (2 if self.bidirectional else 1), self.horizon)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        self._check(x)
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1]))

    @staticmethod
    def param_space():
        # num_layers reaches 6, not 4. The five-trial pilot already selected 4
        # — the old ceiling — and the boundary check fired. A bound that binds
        # is not a choice the search made, it is a limit it ran into, and the
        # previous study was criticised for reporting exactly that (VMD's K
        # came back at the top of its grid).
        #
        # The evidence here is thin: one station, five trials. If the full run
        # never selects more than four layers, the wider range costs nothing
        # and the paper can say the bound was tested rather than assumed.
        return {"hidden_size": (24, 256), "num_layers": (1, 6),
                "dropout": (0.0, 0.5)}


class LSTMForecaster(_RecurrentBase):
    name = "lstm"
    rnn_cls = nn.LSTM


class BiLSTMForecaster(_RecurrentBase):
    name = "bilstm"
    rnn_cls = nn.LSTM
    bidirectional = True


class GRUForecaster(_RecurrentBase):
    name = "gru"
    rnn_cls = nn.GRU


# =============================================================================
# temporal convolution
# =============================================================================
class _Chomp(nn.Module):
    """Trim the right padding so the convolution stays causal."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x[:, :, :-self.size].contiguous() if self.size > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)),
            _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)),
            _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCNForecaster(BaseForecaster):
    """
    Dilated causal convolutions.

    Worth noting for the capacity analysis: a dilated causal TCN is itself a
    learned multiscale filter bank, so it already performs something close to
    what the hand-crafted decomposition provides. The hypothesis that
    decomposition helps low-capacity models more than high-capacity ones is
    tested directly on this grid.
    """

    name = "tcn"

    def build(self):
        ch = int(self.params.get("num_channels", 64))
        layers = int(self.params.get("num_layers", 4))
        kernel = int(self.params.get("kernel_size", 3))
        drop = float(self.params.get("dropout", 0.1))
        blocks, c_in = [], self.input_size
        for i in range(layers):
            blocks.append(_TemporalBlock(c_in, ch, kernel, 2 ** i, drop))
            c_in = ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(ch, self.horizon)

    def forward(self, x):
        self._check(x)
        h = self.tcn(x.transpose(1, 2))
        return self.head(h[:, :, -1])

    @staticmethod
    def param_space():
        return {"num_channels": (16, 128), "num_layers": (2, 10),
                "kernel_size": [2, 3, 5], "dropout": (0.0, 0.5)}


class TCANForecaster(TCNForecaster):
    """TCN with a sparse attention layer over time steps."""

    name = "tcan"

    def build(self):
        super().build()
        ch = int(self.params.get("num_channels", 64))
        att = int(self.params.get("att_dim", 32))
        self.att = nn.Sequential(nn.Linear(ch, att), nn.Tanh(), nn.Linear(att, 1))

    def forward(self, x):
        self._check(x)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)     # (B, T, C)
        w = torch.softmax(self.att(h), dim=1)               # (B, T, 1)
        return self.head((h * w).sum(dim=1))

    @staticmethod
    def param_space():
        space = TCNForecaster.param_space()
        space["att_dim"] = [16, 32, 64]
        return space


# =============================================================================
# hybrid
# =============================================================================
class CNNLSTMForecaster(BaseForecaster):
    """Local feature extraction followed by sequential modelling."""

    name = "cnn_lstm"

    def build(self):
        filters = int(self.params.get("cnn_filters", 64))
        kernel = int(self.params.get("cnn_kernel", 3))
        hidden = int(self.params.get("lstm_hidden", 64))
        layers = int(self.params.get("lstm_layers", 1))
        drop = float(self.params.get("dropout", 0.1))
        pad = (kernel - 1)                       # causal padding
        self.conv = nn.Conv1d(self.input_size, filters, kernel, padding=pad)
        self.chomp = _Chomp(pad)
        self.act = nn.ReLU()
        self.lstm = nn.LSTM(filters, hidden, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, self.horizon)

    def forward(self, x):
        self._check(x)
        h = self.act(self.chomp(self.conv(x.transpose(1, 2)))).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(out[:, -1])

    @staticmethod
    def param_space():
        return {"cnn_filters": (16, 128), "cnn_kernel": [2, 3, 5],
                "lstm_hidden": (24, 256), "lstm_layers": (1, 3),
                "dropout": (0.0, 0.5)}


# =============================================================================
# attention
# =============================================================================
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerForecaster(BaseForecaster):
    """Encoder-only transformer with a causal mask."""

    name = "transformer"

    def build(self):
        d_model = int(self.params.get("d_model", 64))
        nhead = int(self.params.get("nhead", 4))
        if d_model % nhead:                       # keep the search space valid
            d_model = nhead * max(1, d_model // nhead)
        layers = int(self.params.get("num_layers", 2))
        ff = int(self.params.get("dim_feedforward", 128))
        drop = float(self.params.get("dropout", 0.1))

        self.proj = nn.Linear(self.input_size, d_model)
        self.pos = _PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, drop,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, self.horizon)
        self.d_model = d_model

    def forward(self, x):
        self._check(x)
        h = self.pos(self.proj(x))
        t = h.size(1)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool),
                          diagonal=1)
        return self.head(self.encoder(h, mask=mask)[:, -1])

    @staticmethod
    def param_space():
        return {"d_model": [32, 64, 128, 256], "nhead": [2, 4, 8],
                "num_layers": (1, 6), "dim_feedforward": (64, 512),
                "dropout": (0.0, 0.5)}


# =============================================================================
REGISTRY: Dict[str, type] = {
    cls.name: cls for cls in (
        LSTMForecaster, BiLSTMForecaster, GRUForecaster,
        TCNForecaster, TCANForecaster, CNNLSTMForecaster,
        TransformerForecaster,
    )
}


def build_model(name: str, input_size: int, horizon: int,
                params: Optional[dict] = None) -> BaseForecaster:
    if name not in REGISTRY:
        raise ValueError(f"unknown model: {name}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](input_size, horizon, params or {})


def model_param_space(name: str) -> dict:
    return dict(REGISTRY[name].param_space())
