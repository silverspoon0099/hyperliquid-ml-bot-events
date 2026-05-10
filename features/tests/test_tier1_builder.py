"""Tests for features.tier1_builder (spec §7.2 + DR v3.0.13)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from features.tier1_builder import (
    TIER1_ALL, TIER1_ATR, TIER1_BARS_SINCE, TIER1_HTF, TIER1_PIVOT,
    _bars_since_event, compute_atr, compute_htf, compute_pivot,
)


T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────
# bars_since_event helper
# ─────────────────────────────────────────────────────────────────────
def test_bars_since_event_simple():
    """Events at indices 2, 5; bars_since at index 6 should be 1."""
    events = pd.Series([False, False, True, False, False, True, False, False])
    out = _bars_since_event(events)
    # Before any event: NaN
    assert pd.isna(out.iloc[0]) and pd.isna(out.iloc[1])
    # At event index 2: 0 (current = last_event)
    assert out.iloc[2] == 0
    # 1 bar later
    assert out.iloc[3] == 1
    # At event index 5: 0
    assert out.iloc[5] == 0
    # 2 bars after the most recent event (index 5)
    assert out.iloc[7] == 2


def test_bars_since_event_no_events():
    """All False → NaN throughout."""
    events = pd.Series([False] * 10)
    out = _bars_since_event(events)
    assert out.isna().all()


# ─────────────────────────────────────────────────────────────────────
# HTF — log-return at exact lookup point
# ─────────────────────────────────────────────────────────────────────
def test_compute_htf_4h_exact_lookup():
    """Bars at hour 0, 4, 8, ...; htf_ret_4h at bar k = log(c[k]/c[k-1])."""
    n = 12
    closes = [100.0 * math.exp(0.01 * i) for i in range(n)]  # +1% per 4h step
    bars = pd.DataFrame({
        "bar_id": range(1, n + 1),
        "bar_close_ts": [T0 + timedelta(hours=4 * i) for i in range(n)],
        "close": closes,
        "high": closes,
        "low": closes,
    })
    out = compute_htf(bars)
    # First bar can't look back 4h → NaN
    assert pd.isna(out["htf_ret_4h"].iloc[0])
    # Subsequent bars: htf_ret_4h ≈ +0.01 (one 4h step back)
    for i in range(1, n):
        assert abs(out["htf_ret_4h"].iloc[i] - 0.01) < 1e-9, f"bar {i}"


def test_compute_htf_columns_and_dtype():
    n = 50
    closes = [100.0 + i for i in range(n)]
    bars = pd.DataFrame({
        "bar_id": range(1, n + 1),
        "bar_close_ts": [T0 + timedelta(hours=i) for i in range(n)],
        "close": closes, "high": closes, "low": closes,
    })
    out = compute_htf(bars)
    assert set(TIER1_HTF).issubset(out.columns)
    for c in TIER1_HTF:
        assert out[c].dtype == np.float64


# ─────────────────────────────────────────────────────────────────────
# ATR — rolling mean of (high - low) + percentile rank
# ─────────────────────────────────────────────────────────────────────
def test_compute_atr_constant_range():
    """If high-low is constant=2.0, atr_14 = 2.0 once warm."""
    n = 50
    bars = pd.DataFrame({
        "bar_id": range(1, n + 1),
        "high": [102.0] * n,
        "low":  [100.0] * n,
    })
    out = compute_atr(bars)
    # First 13 NaN (warmup), then 2.0
    assert out["atr_14"].iloc[:13].isna().all()
    assert all(abs(v - 2.0) < 1e-9 for v in out["atr_14"].iloc[13:])


def test_compute_atr_pct_rank_in_zero_to_one():
    """Percentile ranks must be in [0, 1] when not NaN."""
    n = 200
    rng = np.random.default_rng(seed=42)
    high = 100 + rng.uniform(0, 5, n)
    low = high - rng.uniform(0.1, 3, n)
    bars = pd.DataFrame({
        "bar_id": range(1, n + 1),
        "high": high, "low": low,
    })
    out = compute_atr(bars)
    pct = out["atr_pct_rank_100"].dropna()
    assert (pct >= 0).all() and (pct <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────
# Pivot proximity — uses PRIOR day's H/L/C
# ─────────────────────────────────────────────────────────────────────
def test_compute_pivot_uses_prior_day():
    """Build 2-day fixture; verify day-2 bars look up day-1's pivot."""
    # Day 1: 10 hourly bars at constant prices around 100
    # Day 2: 10 hourly bars at 110
    rows = []
    for h in range(24):
        rows.append({
            "bar_id": h + 1,
            "bar_close_ts": T0 + timedelta(hours=h),
            "close": 100.0, "high": 101.0, "low": 99.0,
        })
    for h in range(24):
        rows.append({
            "bar_id": 25 + h,
            "bar_close_ts": T0 + timedelta(days=1, hours=h),
            "close": 110.0, "high": 111.0, "low": 109.0,
        })
    bars = pd.DataFrame(rows)
    out = compute_pivot(bars)

    # Day 1 bars: prior day doesn't exist → NaN
    assert out["pivot_distance"].iloc[:24].isna().all()

    # Day 2 bars: prior day H=101, L=99, C=100, P=(101+99+100)/3=100
    # Each day-2 bar has close=110; pivot_distance = log(110/100) ≈ 0.0953
    expected_log = math.log(110.0 / 100.0)
    for v in out["pivot_distance"].iloc[24:]:
        assert abs(v - expected_log) < 1e-9


def test_compute_pivot_columns():
    n = 100
    rows = [{
        "bar_id": i + 1,
        "bar_close_ts": T0 + timedelta(hours=i),
        "close": 100.0 + i * 0.1,
        "high": 100.5 + i * 0.1,
        "low": 99.5 + i * 0.1,
    } for i in range(n)]
    bars = pd.DataFrame(rows)
    out = compute_pivot(bars)
    assert set(TIER1_PIVOT).issubset(out.columns)


# ─────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────
def test_htf_determinism():
    n = 50
    closes = [100.0 + math.sin(i * 0.1) * 5 for i in range(n)]
    bars = pd.DataFrame({
        "bar_id": range(1, n + 1),
        "bar_close_ts": [T0 + timedelta(hours=i) for i in range(n)],
        "close": closes, "high": closes, "low": closes,
    })
    out1 = compute_htf(bars)
    out2 = compute_htf(bars)
    pd.testing.assert_frame_equal(out1, out2)
