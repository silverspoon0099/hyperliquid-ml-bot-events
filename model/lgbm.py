"""LightGBM L0 baseline + Platt calibration (spec §9.2 + DR v3.0.9 §6, §11).

Per spec §9.2: gradient-boosted multiclass with hyperparams pinned in
config.yaml:90-104. Calibration: per-class one-vs-rest sigmoid (Platt)
fit on the val fold's raw probabilities.
"""
from __future__ import annotations

import logging
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

LOG = logging.getLogger("model.lgbm")


def train_lgbm(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    params: dict,
) -> lgb.Booster:
    """Train LightGBM with early stopping on the val set.

    `params` should be the config.yaml `model.L0_lightgbm` dict; extracts
    `num_boost_round` and `early_stopping_rounds` for control flow.
    """
    p = dict(params)  # copy; consume some keys
    num_boost_round = p.pop("num_boost_round", 1000)
    early_stopping_rounds = p.pop("early_stopping_rounds", 50)
    p.setdefault("verbose", -1)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    booster = lgb.train(
        p,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return booster


def fit_platt(
    val_raw_probs: np.ndarray, y_val: np.ndarray, n_classes: int = 3,
) -> list[Optional[LogisticRegression]]:
    """Per-class one-vs-rest sigmoid (Platt). Returns list of LR (or None
    for classes absent from val — caller falls back to raw probs)."""
    out: list[Optional[LogisticRegression]] = []
    for k in range(n_classes):
        y_bin = (y_val == k).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            out.append(None)  # Skip Platt for absent class
            continue
        prob_k = val_raw_probs[:, k].reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(prob_k, y_bin)
        out.append(lr)
    return out


def apply_platt(
    raw_probs: np.ndarray, calibrators: list[Optional[LogisticRegression]],
) -> np.ndarray:
    """Apply per-class Platt calibrators; renormalize rows to sum to 1.0."""
    n_classes = raw_probs.shape[1]
    out = np.zeros_like(raw_probs)
    for k in range(n_classes):
        prob_k = raw_probs[:, k].reshape(-1, 1)
        if calibrators[k] is None:
            out[:, k] = raw_probs[:, k]  # fallback: raw
        else:
            out[:, k] = calibrators[k].predict_proba(prob_k)[:, 1]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid /0 if all calibrators returned 0
    return out / row_sums


def feature_importance_top_k(
    booster: lgb.Booster, feature_names: list[str], k: int = 10,
) -> list[dict]:
    """Top-k features by gain (loss-improvement contribution)."""
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    rows = sorted(
        [
            {"feature": fn, "gain": float(g), "split": int(s)}
            for fn, g, s in zip(feature_names, gain, split)
        ],
        key=lambda r: -r["gain"],
    )
    return rows[:k]
