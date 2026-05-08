"""Tests for model.lgbm (spec §9.2 + DR v3.0.9 §6, §11, §16).

Includes the leakage-detection test (DR §16 fold (a)): inject a
future-information feature into a synthetic random-walk dataset and
verify val_logloss crashes; without the leak, val_logloss / H(p) > 0.7.
"""
import math

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from cv.pre_gate import class_prior_entropy
from model.lgbm import apply_platt, fit_platt, train_lgbm


L0_PARAMS = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "boosting_type": "gbdt", "num_leaves": 31, "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "min_child_samples": 20, "lambda_l1": 0.1, "lambda_l2": 0.1,
    "num_boost_round": 200, "early_stopping_rounds": 20,
    "seed": 42, "verbose": -1, "deterministic": True,
}


def _synthetic_random_walk(seed: int = 42, n: int = 1500):
    """Return (close_series, future_ret_24, labels_array).

    Synthetic 3-class problem for the leakage test ONLY (this is a
    pipeline-integrity fixture, not a model-quality fixture):

      label = LONG    if future_ret_24 > +0.05
            = SHORT   if future_ret_24 < -0.05
            = NEUTRAL otherwise
            = -1      if t + 24 >= n  (UNLABELABLE)

    With this construction `future_ret_24 = log(close[t+24]/close[t])`
    is a deterministic predictor of the label. The leakage test
    asserts the pipeline lets that determinism translate to a logloss
    crash — a property orthogonal to the production triple-barrier
    semantics (which are intra-window-path-dependent and would NOT be
    fully resolved by future_ret_24 alone).

    The DR's literal "future_ret_5" suggestion was a 5-bar leak; we
    use 24 to match the label horizon for a clean determinism check.
    """
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.015, n)  # σ=0.015 → 24-bar σ≈0.073, balanced classes
    close = 100.0 * np.exp(np.cumsum(log_rets))

    labels = np.full(n, -1, dtype=np.int64)
    future_ret_24 = np.full(n, np.nan)
    for t in range(n - 24):
        fr_24 = math.log(close[t + 24] / close[t])
        future_ret_24[t] = fr_24
        if fr_24 > 0.05:
            labels[t] = 0   # LONG
        elif fr_24 < -0.05:
            labels[t] = 1   # SHORT
        else:
            labels[t] = 2   # NEUTRAL

    return close, future_ret_24, labels


def _build_features(close: np.ndarray) -> pd.DataFrame:
    s = pd.Series(close)
    return pd.DataFrame({
        "ret_1": np.log(s / s.shift(1)),
        "ret_5": np.log(s / s.shift(5)),
        "vol_10": np.log(s / s.shift(1)).rolling(10).std(),
        "ema_5": s.ewm(alpha=2 / 6, adjust=False).mean(),
        "ema_20": s.ewm(alpha=2 / 21, adjust=False).mean(),
    })


# ─────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────
def test_lgbm_determinism_seed_42():
    """Two runs with seed=42 produce identical predictions."""
    close, _f24, labels = _synthetic_random_walk(seed=42, n=600)
    feats = _build_features(close)
    feats["label"] = labels
    feats = feats.dropna()
    feats = feats[feats["label"] != -1].copy()

    split = int(0.7 * len(feats))
    train, val = feats.iloc[:split], feats.iloc[split:]
    fcols = ["ret_1", "ret_5", "vol_10", "ema_5", "ema_20"]

    b1 = train_lgbm(train[fcols], train["label"].values,
                    val[fcols], val["label"].values, dict(L0_PARAMS))
    b2 = train_lgbm(train[fcols], train["label"].values,
                    val[fcols], val["label"].values, dict(L0_PARAMS))
    p1 = b1.predict(val[fcols])
    p2 = b2.predict(val[fcols])
    np.testing.assert_array_almost_equal(p1, p2, decimal=12)


def test_platt_calibrated_probs_sum_to_one():
    """Calibrated probabilities row-sum to 1.0 within float epsilon."""
    close, _f24, labels = _synthetic_random_walk(seed=42, n=600)
    feats = _build_features(close)
    feats["label"] = labels
    feats = feats.dropna()
    feats = feats[feats["label"] != -1].copy()

    split = int(0.7 * len(feats))
    train, val = feats.iloc[:split], feats.iloc[split:]
    fcols = ["ret_1", "ret_5", "vol_10", "ema_5", "ema_20"]

    booster = train_lgbm(train[fcols], train["label"].values,
                         val[fcols], val["label"].values, dict(L0_PARAMS))
    val_raw = booster.predict(val[fcols])
    cal = fit_platt(val_raw, val["label"].values, n_classes=3)
    val_cal = apply_platt(val_raw, cal)
    row_sums = val_cal.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────
# Leakage-detection (DR v3.0.9 §16 fold (a))
# ─────────────────────────────────────────────────────────────────────
def test_no_future_leakage():
    """Validates two pipeline-integrity properties (DR v3.0.9 §16 fold (a)):

      (1) WITHOUT leak: random-walk features → val_logloss / H(p) > 0.7
          (no implausibly low logloss from accidental signal — would
          indicate look-ahead in feature computation or full-dataset fit)
      (2) WITH future_ret_24 leak: val_logloss < 0.5 × H(p_train)
          (pipeline lets a deliberate strong leak land; otherwise the
          model is somehow being prevented from using it — a different
          class of bug).

    The leak feature uses the LABEL horizon (24 bars) rather than the
    DR's literal '5' suggestion: a 5-bar forward return carries only
    5/24 of the info needed to resolve 24-bar triple-barrier labels,
    making the strict < 0.5 × H(p) bound infeasible by construction.
    `future_ret_24` directly mirrors the label window and is the
    canonical strong-leak feature for this property test.
    """
    close, future_ret_24, labels = _synthetic_random_walk(seed=42, n=1500)

    feats = _build_features(close)
    feats["label"] = labels
    feats["future_ret_24"] = future_ret_24
    feats = feats.dropna()
    feats = feats[feats["label"] != -1].copy()

    split = int(0.7 * len(feats))
    train = feats.iloc[:split]
    val = feats.iloc[split:]

    # Class prior entropy (anchor)
    H_p = class_prior_entropy(train["label"].values, n_classes=3)

    # ───── No-leak baseline ─────
    no_leak_cols = ["ret_1", "ret_5", "vol_10", "ema_5", "ema_20"]
    booster_no_leak = train_lgbm(
        train[no_leak_cols], train["label"].values,
        val[no_leak_cols], val["label"].values,
        dict(L0_PARAMS),
    )
    val_p_no_leak = booster_no_leak.predict(val[no_leak_cols])
    val_ll_no_leak = log_loss(val["label"].values, val_p_no_leak,
                              labels=[0, 1, 2])
    ratio_no_leak = val_ll_no_leak / H_p

    assert ratio_no_leak > 0.7, (
        f"Pipeline produced implausibly strong signal from random-walk "
        f"features alone: val_logloss/H(p) = {ratio_no_leak:.4f} "
        f"(expected > 0.7). Possible leak in feature pipeline."
    )

    # ───── With future_ret_24 leak ─────
    leak_cols = no_leak_cols + ["future_ret_24"]
    booster_leak = train_lgbm(
        train[leak_cols], train["label"].values,
        val[leak_cols], val["label"].values,
        dict(L0_PARAMS),
    )
    val_p_leak = booster_leak.predict(val[leak_cols])
    val_ll_leak = log_loss(val["label"].values, val_p_leak, labels=[0, 1, 2])
    ratio_leak = val_ll_leak / H_p

    assert ratio_leak < 0.5, (
        f"Leak feature failed to crush logloss: val_logloss/H(p) = "
        f"{ratio_leak:.4f} (expected < 0.5). Either pipeline somehow "
        f"prevents the model using the leak (unexpected) or a different bug."
    )

    # Sanity: leak should be MUCH better than no-leak
    assert ratio_leak < ratio_no_leak * 0.7, (
        f"Leak provides insufficient improvement: no_leak={ratio_no_leak:.4f} "
        f"leak={ratio_leak:.4f}. Expected leak / no_leak < 0.7."
    )
