"""Pre-gate metric per spec §10.3 + DR v3.0.9 §7.

ratio = val_logloss / H(p_train); pass if ratio < threshold (default 0.99).
Aggregate pass: ≥ required_pass of first 6 folds (default 4 of 6).
"""
from __future__ import annotations

import numpy as np


def class_prior_entropy(labels: np.ndarray, n_classes: int = 3) -> float:
    """H(p) = -Σ p_i · ln(p_i) over class proportions."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes)
    p = counts / counts.sum()
    p = p[p > 0]
    return -float(np.sum(p * np.log(p)))


def pre_gate_ratio(val_logloss: float, train_labels: np.ndarray,
                   n_classes: int = 3) -> float:
    """val_logloss divided by H(p_train)."""
    H = class_prior_entropy(train_labels, n_classes=n_classes)
    return float(val_logloss / H)


def aggregate_pre_gate(
    fold_ratios: list[float],
    threshold: float = 0.99,
    required_pass: int = 4,
    n_first: int = 6,
) -> dict:
    """Pass if at least `required_pass` of the first `n_first` folds
    have ratio < threshold."""
    first = fold_ratios[:n_first]
    n_passed = sum(1 for r in first if r < threshold)
    return {
        "n_evaluated": len(first),
        "n_passed": n_passed,
        "required_pass": required_pass,
        "passed": n_passed >= required_pass,
        "threshold": threshold,
    }
