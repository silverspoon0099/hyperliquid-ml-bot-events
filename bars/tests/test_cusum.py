"""Unit tests for bars.cusum (spec §6.4 + DR v3.0.5).

Fixture set per DR v3.0.5: empty input, single tick, threshold-edge ≥
and < ε, monotonic up/down, sideways force-close, mixed walk invariants,
reset after CUSUM-triggered close, explicit OHLC/volume/n_trades,
cusum_pos/neg at close == trigger state, determinism.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from bars.cusum import CusumBuilder, cusum_bars

T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
THRESHOLD = 0.02


def _ticks_at(prices, qty=1.0, ts_step_s=1):
    """Build (ts, price, qty) tuples from a price list, evenly spaced."""
    return [
        (T0 + timedelta(seconds=i * ts_step_s), float(p), float(qty))
        for i, p in enumerate(prices)
    ]


# ─────────────────────────────────────────────────────────────────────
# Boundary cases
# ─────────────────────────────────────────────────────────────────────
def test_empty_input_yields_no_bars():
    assert list(cusum_bars(iter([]), THRESHOLD)) == []


def test_single_tick_yields_no_bars():
    """CUSUM requires >=2 ticks to compute the first return."""
    assert list(cusum_bars(iter(_ticks_at([100.0])), THRESHOLD)) == []


# ─────────────────────────────────────────────────────────────────────
# Threshold edge cases
# ─────────────────────────────────────────────────────────────────────
def test_return_at_threshold_emits_bar():
    """r ≈ exactly threshold → trigger fires (>= comparison, not >).

    A tiny positive epsilon on `exp(threshold)` sidesteps the
    log/exp round-trip ambiguity and guarantees r ≥ threshold.
    """
    p1 = 100.0 * math.exp(THRESHOLD + 1e-12)
    bars = list(cusum_bars(iter(_ticks_at([100.0, p1])), THRESHOLD))
    assert len(bars) == 1
    assert bars[0].cusum_pos >= THRESHOLD


def test_return_just_below_threshold_emits_no_bar():
    p1 = 100.0 * math.exp(THRESHOLD - 1e-6)
    bars = list(cusum_bars(iter(_ticks_at([100.0, p1])), THRESHOLD))
    assert bars == []


# ─────────────────────────────────────────────────────────────────────
# Monotonic price walks
# ─────────────────────────────────────────────────────────────────────
def test_monotonic_up_one_bar_per_step():
    """Each tick contributes +(threshold * 1.01) → exactly 1 bar per step."""
    step = THRESHOLD * 1.01
    prices = [100.0 * math.exp(step * i) for i in range(11)]
    bars = list(cusum_bars(iter(_ticks_at(prices)), THRESHOLD))
    assert len(bars) == 10
    for b in bars:
        assert b.cusum_pos >= THRESHOLD
        assert -b.cusum_neg < THRESHOLD


def test_monotonic_down_one_bar_per_step():
    step = THRESHOLD * 1.01
    prices = [100.0 * math.exp(-step * i) for i in range(11)]
    bars = list(cusum_bars(iter(_ticks_at(prices)), THRESHOLD))
    assert len(bars) == 10
    for b in bars:
        assert -b.cusum_neg >= THRESHOLD
        assert b.cusum_pos < THRESHOLD


# ─────────────────────────────────────────────────────────────────────
# Sideways → 168-h force-close (spec §6.5 fail-safe)
# ─────────────────────────────────────────────────────────────────────
def test_sideways_below_threshold_force_closes_at_max_duration():
    """±0.5% oscillation never triggers CUSUM; bar force-closes at 168h."""
    n = 170  # ticks at 1h spacing → covers 169 hours
    prices = [100.0 * (1.005 if i % 2 == 0 else 0.995) for i in range(n)]
    ticks = [(T0 + timedelta(hours=i), p, 1.0) for i, p in enumerate(prices)]
    bars = list(cusum_bars(iter(ticks), THRESHOLD, max_duration_h=168))
    assert len(bars) >= 1
    b0 = bars[0]
    duration_h = (b0.bar_close_ts - b0.bar_open_ts).total_seconds() / 3600
    assert duration_h >= 168.0
    assert duration_h < 169.0
    assert max(b0.cusum_pos, -b0.cusum_neg) < THRESHOLD


# ─────────────────────────────────────────────────────────────────────
# Reset after CUSUM-triggered close — INTERNAL state assertion
# ─────────────────────────────────────────────────────────────────────
def test_reset_after_cusum_triggered_close():
    """S+ / S- / bar_open_ts must be zeroed after a triggered bar."""
    builder = CusumBuilder(THRESHOLD)
    builder.step(T0, 100.0, 1.0)
    p1 = 100.0 * math.exp(THRESHOLD + 1e-12)
    bar = builder.step(T0 + timedelta(seconds=1), p1, 1.0)
    assert bar is not None
    assert builder.s_pos == 0.0
    assert builder.s_neg == 0.0
    assert builder.bar_open_ts is None

    # Next tick starts a fresh bar; small step → no immediate trigger.
    p2 = p1 * math.exp(THRESHOLD * 0.001)
    bar2 = builder.step(T0 + timedelta(seconds=2), p2, 1.0)
    assert bar2 is None
    assert builder.s_pos > 0.0
    assert builder.s_pos < THRESHOLD
    assert builder.bar_open_ts == T0 + timedelta(seconds=2)


# ─────────────────────────────────────────────────────────────────────
# Explicit OHLC / volume / n_trades
# ─────────────────────────────────────────────────────────────────────
def test_ohlc_volume_n_trades_match_constructed_bar():
    p3 = 100.0 * math.exp(0.025)  # > threshold so this tick triggers
    prices = [100.0, 100.5, 99.5, p3]
    qtys = [1.0, 0.5, 2.0, 0.3]
    ticks = [(T0 + timedelta(seconds=i), p, q)
             for i, (p, q) in enumerate(zip(prices, qtys))]
    bars = list(cusum_bars(iter(ticks), THRESHOLD))
    assert len(bars) == 1
    b = bars[0]
    assert b.bar_open_ts == ticks[0][0]
    assert b.bar_close_ts == ticks[-1][0]
    assert b.open == prices[0]
    assert b.close == prices[-1]
    assert b.high == max(prices)
    assert b.low == min(prices)
    assert b.volume == pytest.approx(sum(qtys))
    assert b.n_trades == len(ticks)


# ─────────────────────────────────────────────────────────────────────
# cusum_pos / cusum_neg at close = algorithm S+/S- at trigger moment
# ─────────────────────────────────────────────────────────────────────
def test_cusum_pos_neg_at_close_record_trigger_state():
    """For a CUSUM-triggered bar, max(cusum_pos, -cusum_neg) >= threshold."""
    p1 = 100.0 * math.exp(THRESHOLD * 1.5)
    bars = list(cusum_bars(iter(_ticks_at([100.0, p1])), THRESHOLD))
    assert len(bars) == 1
    b = bars[0]
    assert max(b.cusum_pos, -b.cusum_neg) >= THRESHOLD
    assert b.cusum_pos == pytest.approx(THRESHOLD * 1.5, rel=1e-6)
    # Recorded state is non-zero (the WHY of the close)
    assert b.cusum_pos != 0.0


# ─────────────────────────────────────────────────────────────────────
# Mixed walk — bar invariants
# ─────────────────────────────────────────────────────────────────────
def test_mixed_walk_bar_invariants_hold():
    random.seed(20260507)
    n = 2000
    p = 100.0
    prices = []
    for _ in range(n):
        p *= math.exp(random.gauss(0, 0.003))
        prices.append(p)
    bars = list(cusum_bars(iter(_ticks_at(prices, ts_step_s=10)), THRESHOLD))
    assert len(bars) > 0
    for b in bars:
        assert b.high >= max(b.open, b.close)
        assert b.low <= min(b.open, b.close)
        assert b.high >= b.low
        assert b.volume > 0
        assert b.n_trades >= 1
        assert b.bar_close_ts >= b.bar_open_ts


# ─────────────────────────────────────────────────────────────────────
# Determinism — same input twice → identical bars
# ─────────────────────────────────────────────────────────────────────
def _make_random_walk(n=500, seed=20260507):
    """Generate a reproducible random-walk tick stream."""
    random.seed(seed)
    p = 100.0
    out = []
    for i in range(n):
        p *= math.exp(random.gauss(0, 0.005))
        q = random.uniform(0.001, 5.0)
        out.append((T0 + timedelta(milliseconds=i * 100), p, q))
    return out


def test_determinism_same_input_same_output():
    ticks_a = _make_random_walk()
    ticks_b = _make_random_walk()
    bars_a = list(cusum_bars(iter(ticks_a), THRESHOLD))
    bars_b = list(cusum_bars(iter(ticks_b), THRESHOLD))
    assert bars_a == bars_b
    assert len(bars_a) > 0  # sanity: the random walk does produce some bars
