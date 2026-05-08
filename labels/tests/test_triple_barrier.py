"""Unit tests for labels.triple_barrier (spec §8.1 + DR v3.0.8).

Synthetic-bars set: TP-then-SL → LONG, SL-then-TP → SHORT, neither →
NEUTRAL/timeout, whipsaw → NEUTRAL/ambiguous, last-24 → UNLABELABLE,
determinism. Plus 4 boundary tests + 30-bar golden.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from labels.triple_barrier import (
    LABEL_LONG, LABEL_SHORT, LABEL_NEUTRAL, LABEL_UNLABELABLE,
    apply_triple_barrier,
)

T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _make_bars(
    prices,
    highs: Optional[list] = None,
    lows: Optional[list] = None,
    qty: float = 1.0,
):
    """Build a fake bars DataFrame. high/low default to close * (1 ± 0.001)."""
    n = len(prices)
    if highs is None:
        highs = [p * 1.001 for p in prices]
    if lows is None:
        lows = [p * 0.999 for p in prices]
    return pd.DataFrame({
        "bar_id": list(range(1, n + 1)),
        "bar_open_ts": [T0 + timedelta(minutes=i * 30) for i in range(n)],
        "bar_close_ts": [T0 + timedelta(minutes=(i + 1) * 30) for i in range(n)],
        "open":  [float(p) for p in prices],
        "high":  [float(h) for h in highs],
        "low":   [float(l) for l in lows],
        "close": [float(p) for p in prices],
        "volume":   [float(qty)] * n,
        "n_trades": [1] * n,
        "cusum_pos": [0.02] * n,
        "cusum_neg": [0.0] * n,
    })


# ─────────────────────────────────────────────────────────────────────
# Original synthetic-bars set
# ─────────────────────────────────────────────────────────────────────
def test_tp_then_sl_first_touch_wins_long():
    """Bar 5 hits TP, bar 10 hits SL → LONG (first touch)."""
    prices = [100.0] * 5 + [110.0] + [100.0] * 4 + [85.0] + [100.0] * 19
    bars = _make_bars(prices)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)
    row = out.iloc[0]
    assert row["label"] == LABEL_LONG
    assert row["exit_reason"] == "tp"
    assert row["holding_bars"] == 5
    assert row["exit_bar_id"] == 6  # bar_id 1-indexed; index 5 → bar_id 6


def test_sl_then_tp_first_touch_wins_short():
    """Bar 5 hits SL, bar 10 hits TP → SHORT."""
    prices = [100.0] * 5 + [85.0] + [100.0] * 4 + [110.0] + [100.0] * 19
    bars = _make_bars(prices)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)
    row = out.iloc[0]
    assert row["label"] == LABEL_SHORT
    assert row["exit_reason"] == "sl"
    assert row["holding_bars"] == 5


def test_neither_hit_yields_neutral_timeout():
    """No barrier touched in 24 bars → NEUTRAL/timeout."""
    prices = [100.0] * 30
    bars = _make_bars(prices)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)
    row = out.iloc[0]
    assert row["label"] == LABEL_NEUTRAL
    assert row["exit_reason"] == "timeout"
    assert row["holding_bars"] == 24


def test_whipsaw_both_barriers_in_one_bar_yields_ambiguous():
    """Bar 1 high≥TP AND low≤SL → NEUTRAL with exit_reason='ambiguous'."""
    prices = [100.0] * 30
    highs = [100.001] * 30
    lows = [99.999] * 30
    highs[1] = 106.0   # ≥ 100*1.05
    lows[1] = 94.0     # ≤ 100*0.95
    bars = _make_bars(prices, highs=highs, lows=lows)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)
    row = out.iloc[0]
    assert row["label"] == LABEL_NEUTRAL
    assert row["exit_reason"] == "ambiguous"
    assert row["holding_bars"] == 1


def test_last_24_bars_are_unlabelable():
    """Bars at indices N-24..N-1 → label=-1, all diagnostic null."""
    prices = [100.0] * 30
    bars = _make_bars(prices)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)
    # Bars 0..5 are labelable; bars 6..29 are UNLABELABLE
    for i in range(6, 30):
        row = out.iloc[i]
        assert row["label"] == LABEL_UNLABELABLE, f"bar {i} should be UNLABELABLE"
        assert pd.isna(row["exit_bar_id"])
        assert pd.isna(row["exit_reason"])
        assert pd.isna(row["holding_bars"])
        assert np.isnan(row["exit_price"])


def test_determinism_same_input_same_output():
    """Two runs on the same bars → identical output."""
    prices = [100.0 + 0.5 * (i % 7) - 0.3 * (i % 11) for i in range(60)]
    bars1 = _make_bars(prices)
    bars2 = _make_bars(prices)
    out1 = apply_triple_barrier(bars1, 0.05, 0.05, 24)
    out2 = apply_triple_barrier(bars2, 0.05, 0.05, 24)
    pd.testing.assert_frame_equal(out1, out2)


# ─────────────────────────────────────────────────────────────────────
# Boundary tests (DR v3.0.8 §13 fold)
# ─────────────────────────────────────────────────────────────────────
def test_tp_at_exactly_t_plus_1():
    """TP fires on the very first forward bar → holding_bars=1."""
    prices = [100.0, 110.0] + [100.0] * 28
    bars = _make_bars(prices)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)
    row = out.iloc[0]
    assert row["label"] == LABEL_LONG
    assert row["exit_reason"] == "tp"
    assert row["holding_bars"] == 1


def test_tp_at_exactly_t_plus_24():
    """TP fires on the vertical bar — exit_reason must be 'tp', NOT 'timeout'."""
    prices = [100.0] * 24 + [110.0] + [100.0] * 5
    bars = _make_bars(prices)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)
    row = out.iloc[0]
    assert row["label"] == LABEL_LONG
    assert row["exit_reason"] == "tp"
    assert row["holding_bars"] == 24


def test_no_tp_no_sl_through_t_plus_24_yields_timeout():
    """Corollary of t+24 boundary: when nothing fires by t+24 → 'timeout'."""
    prices = [100.0] * 30
    bars = _make_bars(prices)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)
    row = out.iloc[0]
    assert row["label"] == LABEL_NEUTRAL
    assert row["exit_reason"] == "timeout"
    assert row["holding_bars"] == 24


def test_sl_at_exactly_t_plus_24_yields_short_sl():
    """SL fires on the vertical bar — exit_reason='sl' (mirror of t+24 TP)."""
    prices = [100.0] * 24 + [85.0] + [100.0] * 5
    bars = _make_bars(prices)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)
    row = out.iloc[0]
    assert row["label"] == LABEL_SHORT
    assert row["exit_reason"] == "sl"
    assert row["holding_bars"] == 24


# ─────────────────────────────────────────────────────────────────────
# 30-bar golden fixture (DR v3.0.8 §13)
# ─────────────────────────────────────────────────────────────────────
def test_30_bar_golden_fixture():
    """Hand-computed labels for a 30-bar engineered price path.

    With vertical=24, only bars 0..5 are labelable (bars 6..29 lack 24
    forward bars). Price path:
        bars 0..4: $100 (flat)
        bar 5:    $110 (TP for bars 0..4 fires here; this bar is the source for bar 5's own forward window)
        bar 6:    $85  (SL for bar 5 fires here; SL_5 = $104.5; low_6 = $84.915)
        bars 7..29: $100

    Expected:
        bar 0: LONG,  holding=5  (TP_0 = $105; high_5 = $110.11 ≥ $105)
        bar 1: LONG,  holding=4
        bar 2: LONG,  holding=3
        bar 3: LONG,  holding=2
        bar 4: LONG,  holding=1
        bar 5: SHORT, holding=1  (SL_5 = $104.5; low_6 = $84.915 ≤ $104.5)
        bars 6..29: UNLABELABLE
    """
    prices = [100.0] * 5 + [110.0, 85.0] + [100.0] * 23
    bars = _make_bars(prices)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)

    expected = [
        (LABEL_LONG,  "tp", 5),
        (LABEL_LONG,  "tp", 4),
        (LABEL_LONG,  "tp", 3),
        (LABEL_LONG,  "tp", 2),
        (LABEL_LONG,  "tp", 1),
        (LABEL_SHORT, "sl", 1),
    ]
    for i, (lab, reason, hold) in enumerate(expected):
        row = out.iloc[i]
        assert row["label"] == lab, f"bar {i}: label {row['label']} != {lab}"
        assert row["exit_reason"] == reason, (
            f"bar {i}: reason {row['exit_reason']} != {reason}"
        )
        assert row["holding_bars"] == hold, (
            f"bar {i}: holding {row['holding_bars']} != {hold}"
        )

    # Bars 6..29: UNLABELABLE
    for i in range(6, 30):
        assert out.iloc[i]["label"] == LABEL_UNLABELABLE


# ─────────────────────────────────────────────────────────────────────
# Schema / dtype
# ─────────────────────────────────────────────────────────────────────
def test_output_schema_six_cols_in_order():
    """DR v3.0.8 §1: exact 6 columns in specified order with correct dtypes."""
    bars = _make_bars([100.0] * 30)
    out = apply_triple_barrier(bars, 0.05, 0.05, 24)
    assert list(out.columns) == [
        "bar_id", "label", "exit_bar_id", "exit_reason",
        "holding_bars", "exit_price",
    ]
    assert out["bar_id"].dtype == np.int64
    assert out["label"].dtype == np.int8
    assert str(out["exit_bar_id"].dtype) == "Int64"
    assert str(out["exit_reason"].dtype) == "string"
    assert str(out["holding_bars"].dtype) == "Int8"
    assert out["exit_price"].dtype == np.float64
