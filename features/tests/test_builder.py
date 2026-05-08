"""Unit tests for features.builder (spec §7.1 + DR v3.0.7).

Synthetic-OHLCV fixture set: shape, NaN warmup, indicator ranges,
determinism. Plus four golden-value / consistency tests (DR §13):
RSI(14) Wilder, EMA(20) α=2/(N+1), MACD hist consistency, BB symmetry.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from features.builder import (
    ALL_COLS, FEATURE_COLS,
    build_features,
    bollinger, cmf, ema, log_return_1, macd, mfi,
    rsi, stddev, stochastic, williams_r,
)

T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _make_bars(close_prices, qty=1.0, ts_step_min=30):
    """Build a fake bars DataFrame from a price list."""
    n = len(close_prices)
    return pd.DataFrame({
        "bar_id": list(range(1, n + 1)),
        "bar_open_ts": [T0 + timedelta(minutes=i * ts_step_min) for i in range(n)],
        "bar_close_ts": [T0 + timedelta(minutes=(i + 1) * ts_step_min) for i in range(n)],
        "open":  [float(p) for p in close_prices],
        "high":  [float(p) * 1.001 for p in close_prices],
        "low":   [float(p) * 0.999 for p in close_prices],
        "close": [float(p) for p in close_prices],
        "volume":   [float(qty)] * n,
        "n_trades": [1] * n,
        "cusum_pos": [0.02] * n,
        "cusum_neg": [0.0] * n,
    })


# ─────────────────────────────────────────────────────────────────────
# Shape / dtype / column order
# ─────────────────────────────────────────────────────────────────────
def test_output_has_35_columns_in_order():
    bars = _make_bars([100.0 + i * 0.5 for i in range(60)])
    feats = build_features(bars)
    assert list(feats.columns) == ALL_COLS
    assert len(feats.columns) == 35
    assert len(feats) == len(bars)


def test_dtypes_canonical():
    bars = _make_bars([100.0 + i for i in range(60)])
    feats = build_features(bars)
    assert feats["bar_id"].dtype == np.int64
    assert pd.api.types.is_datetime64_any_dtype(feats["bar_close_ts"])
    for col in FEATURE_COLS:
        assert feats[col].dtype == np.float64, f"{col} dtype = {feats[col].dtype}"


# ─────────────────────────────────────────────────────────────────────
# Warmup / NaN behavior — DR v3.0.7 §12
# ─────────────────────────────────────────────────────────────────────
def test_warmup_nan_then_clean_after_50_rows():
    """NaN density drops to 0 at row 50 (longest ramp = std_50)."""
    bars = _make_bars([100.0 + 0.5 * i + 0.1 * (i % 3) for i in range(60)])
    feats = build_features(bars)
    fc = feats[FEATURE_COLS]
    assert fc.iloc[:5].isna().sum().sum() > 0
    assert fc.iloc[50:].isna().sum().sum() == 0


# ─────────────────────────────────────────────────────────────────────
# Range plausibility
# ─────────────────────────────────────────────────────────────────────
def test_rsi_in_zero_to_hundred():
    bars = _make_bars([100.0 + math.sin(i * 0.1) * 5 for i in range(80)])
    feats = build_features(bars)
    for col in ("rsi_6", "rsi_10", "rsi_14"):
        v = feats[col].dropna()
        assert (v >= 0).all() and (v <= 100).all()


def test_williams_r_in_minus_hundred_to_zero():
    bars = _make_bars([100.0 + math.sin(i * 0.2) * 3 for i in range(80)])
    feats = build_features(bars)
    v = feats["williams_r"].dropna()
    assert (v >= -100).all() and (v <= 0).all()


def test_seasonality_sin_cos_in_minus_one_to_one():
    bars = _make_bars([100.0] * 60)
    feats = build_features(bars)
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        v = feats[col]
        assert (v >= -1.0 - 1e-12).all() and (v <= 1.0 + 1e-12).all()


# ─────────────────────────────────────────────────────────────────────
# Determinism — same input → identical output
# ─────────────────────────────────────────────────────────────────────
def test_determinism_same_input_same_output():
    prices = [100.0 + 0.5 * i + 0.1 * (i % 5) for i in range(100)]
    feats1 = build_features(_make_bars(prices))
    feats2 = build_features(_make_bars(prices))
    pd.testing.assert_frame_equal(feats1, feats2)


# ─────────────────────────────────────────────────────────────────────
# Golden-value / consistency tests (DR v3.0.7 §13)
# ─────────────────────────────────────────────────────────────────────
def test_ema_20_golden_value():
    """EMA(20) hand-computed: y[0]=x[0], y[i]=α·x[i]+(1−α)·y[i−1], α=2/21."""
    n = 50
    prices = [100.0 + i for i in range(n)]
    s = pd.Series(prices, dtype=float)

    alpha = 2.0 / (20 + 1)
    expected = [prices[0]]
    for i in range(1, n):
        expected.append(alpha * prices[i] + (1 - alpha) * expected[-1])

    actual = ema(s, 20)
    for i in (20, 30, 40):
        assert abs(actual.iloc[i] - expected[i]) < 1e-9, \
            f"EMA(20)[{i}]: actual={actual.iloc[i]}, expected={expected[i]}"


def test_rsi_14_golden_value():
    """Wilder's RSI(14) hand-computed: SMA-init at index 14, then RMA."""
    prices = [
        100.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0,
        104.5, 106.0, 105.5, 107.0, 106.5, 108.0, 107.5, 109.0,
        108.5, 110.0, 109.5, 111.0, 110.5, 112.0, 111.5, 113.0,
        112.5, 111.0, 110.0, 109.0, 108.0, 107.0,
    ]
    n = len(prices)
    s = pd.Series(prices, dtype=float)
    length = 14

    deltas = [prices[i] - prices[i - 1] for i in range(1, n)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [-min(d, 0.0) for d in deltas]

    avg_gains = [None] * (n - 1)
    avg_losses = [None] * (n - 1)
    avg_gains[length - 1] = sum(gains[:length]) / length
    avg_losses[length - 1] = sum(losses[:length]) / length
    for i in range(length, n - 1):
        avg_gains[i] = (avg_gains[i - 1] * (length - 1) + gains[i]) / length
        avg_losses[i] = (avg_losses[i - 1] * (length - 1) + losses[i]) / length

    expected = [None] * n
    for i in range(length, n):
        ag = avg_gains[i - 1]
        al = avg_losses[i - 1]
        expected[i] = 100.0 - (100.0 / (1.0 + ag / al))

    actual = rsi(s, length)
    for i in (14, 20, 25):
        assert abs(actual.iloc[i] - expected[i]) < 1e-9, \
            f"RSI(14)[{i}]: actual={actual.iloc[i]}, expected={expected[i]}"


def test_macd_hist_equals_line_minus_signal():
    """macd_hist == macd_line − macd_signal exactly (derived; no float epsilon)."""
    bars = _make_bars([100.0 + 0.3 * i + 0.05 * (i % 7) for i in range(80)])
    feats = build_features(bars)
    diff = feats["macd_line"] - feats["macd_signal"]
    pd.testing.assert_series_equal(
        feats["macd_hist"].dropna(),
        diff.dropna(),
        check_names=False,
    )


def test_bollinger_band_symmetry():
    """(bb_upper − sma_5) ≈ (sma_5 − bb_lower) within float epsilon."""
    bars = _make_bars([100.0 + math.sin(i * 0.3) * 2 for i in range(60)])
    feats = build_features(bars)
    sma_5 = bars["close"].rolling(5).mean().values
    upper_dev = feats["bb_upper"].values - sma_5
    lower_dev = sma_5 - feats["bb_lower"].values
    diff = np.abs(upper_dev - lower_dev)
    diff_clean = diff[~np.isnan(diff)]
    assert (diff_clean < 1e-9).all()
