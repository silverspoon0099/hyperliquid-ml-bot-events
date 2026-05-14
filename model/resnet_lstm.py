"""ResNet-LSTM L1 model (DR v3.0.17 / spec §5.2).

PyTorch CPU implementation of a 1D-ResNet + LSTM hybrid for
sequence-based bar-direction classification. Input: 96-bar sequences
of 33 features. Output: 3-class softmax (LONG / SHORT / NEUTRAL).

Architecture:
    (batch, 96, 33) — input
    → transpose to (batch, 33, 96) for Conv1d
    → Conv1d(33 → conv_channels, kernel=conv_kernel) → BN → ReLU
    → ResBlock1D (Conv → BN → ReLU → Dropout → Conv → BN → skip → ReLU)
    → transpose to (batch, 96, conv_channels) for LSTM
    → LSTM(conv_channels → lstm_hidden, layers=lstm_layers)
    → take last-layer last-step hidden state
    → Dropout → Linear(lstm_hidden → 3)
    → logits (cross-entropy applies internal softmax)

Training:
    - Optimizer: Adam (default β1=0.9, β2=0.999)
    - Loss: cross-entropy (3-class)
    - Max epochs: 30 with early stopping on val_logloss (patience 5)
    - Standardize features on train fold; apply to val/oot
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

LOG = logging.getLogger("model.resnet_lstm")


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        return F.relu(x)


class ResNetLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        conv_channels: int = 64,
        conv_kernel: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        n_classes: int = 3,
    ):
        super().__init__()
        padding = conv_kernel // 2
        self.input_proj = nn.Conv1d(n_features, conv_channels, conv_kernel,
                                     padding=padding, bias=False)
        self.input_bn = nn.BatchNorm1d(conv_channels)
        self.resblock = ResBlock1D(conv_channels, conv_kernel, dropout)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → transpose for Conv1d
        x = x.transpose(1, 2)  # (batch, n_features, seq_len)
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.resblock(x)
        x = x.transpose(1, 2)  # (batch, seq_len, conv_channels)
        out, (h_n, _) = self.lstm(x)
        last = h_n[-1]  # (batch, lstm_hidden)
        last = self.dropout(last)
        return self.fc(last)


# ─────────────────────────────────────────────────────────────────────
# Config & training
# ─────────────────────────────────────────────────────────────────────
@dataclass
class L1Config:
    name: str
    conv_kernel: int
    conv_channels: int
    lstm_hidden: int
    lstm_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    max_epochs: int = 30
    patience: int = 5


# DR v3.0.17 hand-picked configs (5 variants)
L1_CONFIGS = [
    L1Config(name="A_small",       conv_kernel=3, conv_channels=32,  lstm_hidden=64,  lstm_layers=1, dropout=0.1, learning_rate=1e-4, batch_size=64),
    L1Config(name="B_medium",      conv_kernel=5, conv_channels=64,  lstm_hidden=128, lstm_layers=1, dropout=0.2, learning_rate=1e-4, batch_size=64),
    L1Config(name="C_large",       conv_kernel=5, conv_channels=128, lstm_hidden=256, lstm_layers=2, dropout=0.3, learning_rate=5e-5, batch_size=64),
    L1Config(name="D_deep",        conv_kernel=7, conv_channels=64,  lstm_hidden=128, lstm_layers=2, dropout=0.2, learning_rate=1e-4, batch_size=64),
    L1Config(name="E_wide_batch",  conv_kernel=3, conv_channels=128, lstm_hidden=128, lstm_layers=1, dropout=0.2, learning_rate=1e-4, batch_size=128),
]


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    seq_len: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (n_valid, seq_len, n_features) and (n_valid,) labels.

    Args:
        features: (N, n_features) standardized features for all bars
        labels: (N,) class labels (0/1/2) or -1 for skip
        indices: array of bar positions to attempt — must be sorted ascending
        seq_len: window length (default 96)

    Returns:
        (X, y, kept_indices) where:
            X: (n_valid, seq_len, n_features)
            y: (n_valid,) labels
            kept_indices: subset of `indices` actually included (those with
                          enough history AND label != -1 AND no NaN in sequence)
    """
    keep = []
    for i in indices:
        if i < seq_len - 1:
            continue
        if labels[i] == -1:
            continue
        seq = features[i - seq_len + 1: i + 1]
        if np.isnan(seq).any():
            continue
        keep.append(i)
    if not keep:
        return (np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64))
    keep = np.asarray(keep, dtype=np.int64)
    X = np.stack([features[i - seq_len + 1: i + 1] for i in keep]).astype(np.float32)
    y = labels[keep].astype(np.int64)
    return X, y, keep


def train_resnet_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    config: L1Config,
    n_features: int,
    n_classes: int = 3,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[ResNetLSTM, dict]:
    """Train a single ResNet-LSTM with early stopping on val_logloss.

    Returns (best_model, history) where history has per-epoch train/val loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ResNetLSTM(
        n_features=n_features,
        conv_channels=config.conv_channels,
        conv_kernel=config.conv_kernel,
        lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
        n_classes=n_classes,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    best_val_loss = float("inf")
    best_state = None
    patience_left = config.patience
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.max_epochs):
        # Train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_losses.append(loss.item())

        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_mean)
        history["val_loss"].append(val_mean)

        if verbose:
            LOG.info("  epoch %3d  train_loss=%.4f  val_loss=%.4f", epoch + 1, train_mean, val_mean)

        if val_mean < best_val_loss - 1e-5:
            best_val_loss = val_mean
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    LOG.info("  early stop at epoch %d (best val_loss=%.4f)", epoch + 1, best_val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val_loss
    history["epochs_run"] = len(history["train_loss"])
    return model, history


def predict_proba(model: ResNetLSTM, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Return (n, 3) class probabilities via softmax."""
    model.eval()
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb)
            p = F.softmax(logits, dim=1).numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0) if probs else np.empty((0, 3), dtype=np.float32)
