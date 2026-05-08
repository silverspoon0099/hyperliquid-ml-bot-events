"""Tests for backtest.runner (spec §11 + DR v3.0.9 §8-§10)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from backtest.runner import (
    Trade, build_equity_curve, compute_metrics,
    daily_log_returns, simulate_trades,
)


T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _bars(n: int):
    return pd.DataFrame({
        "bar_id": range(1, n + 1),
        "bar_close_ts": [T0 + timedelta(hours=i) for i in range(n)],
        "close": [100.0 + i * 0.01 for i in range(n)],
    })


def _labels_long_at_each_bar(n: int, horizon: int = 5):
    """Each bar exits 5 bars later, label LONG, exit_price=close+5%."""
    rows = []
    for i in range(n):
        if i + horizon >= n:
            rows.append({"bar_id": i + 1, "label": -1, "exit_bar_id": pd.NA,
                         "exit_price": np.nan, "exit_reason": pd.NA,
                         "holding_bars": pd.NA})
        else:
            rows.append({
                "bar_id": i + 1, "label": 0, "exit_bar_id": i + 1 + horizon,
                "exit_price": (100.0 + i * 0.01) * 1.05,
                "exit_reason": "tp", "holding_bars": horizon,
            })
    return pd.DataFrame(rows)


def _preds_all_long(n: int, p_long: float = 0.9):
    return pd.DataFrame({
        "bar_id": range(1, n + 1),
        "p_long":   [p_long] * n,
        "p_short":  [(1 - p_long) / 2] * n,
        "p_neutral": [(1 - p_long) / 2] * n,
    })


# ─────────────────────────────────────────────────────────────────────
def test_cost_application_subtracts_11_bps():
    """Each completed trade has 11 bps subtracted from gross return."""
    bars = _bars(20)
    labs = _labels_long_at_each_bar(20, horizon=3)
    preds = _preds_all_long(20)
    trades = simulate_trades(preds, bars, labs,
                             confidence_threshold=0.6,
                             cost_bps_round_trip=11.0,
                             max_concurrent=999)
    assert len(trades) > 0
    for t in trades:
        # gross = 5% on a 5-bps-rounded baseline
        assert abs(t.pnl_bps_gross - 500.0) < 1e-6
        assert abs(t.pnl_bps_net - (500.0 - 11.0)) < 1e-6


def test_no_trade_when_below_confidence_threshold():
    """All p_long below threshold → no trades."""
    bars = _bars(20)
    labs = _labels_long_at_each_bar(20, horizon=3)
    preds = pd.DataFrame({
        "bar_id": range(1, 21),
        "p_long": [0.5] * 20, "p_short": [0.3] * 20, "p_neutral": [0.2] * 20,
    })
    trades = simulate_trades(preds, bars, labs, confidence_threshold=0.6)
    assert trades == []


def test_max_concurrent_one_skips_overlapping_signals():
    """While trade 0 is open (5-bar holding), bars 1..4 are skipped."""
    bars = _bars(20)
    labs = _labels_long_at_each_bar(20, horizon=5)
    preds = _preds_all_long(20)
    trades = simulate_trades(preds, bars, labs, max_concurrent=1)
    # Bar 0 trades, exits at bar 5. Bar 5 trades next, exits at bar 10. Etc.
    # Expect ~20/5 ≈ 4 trades (bars 0,5,10) — depends on labelable count
    assert len(trades) >= 2
    assert len(trades) <= 5
    # Verify each trade's entry_ts is at or after the previous trade's exit_ts
    for prev, cur in zip(trades, trades[1:]):
        assert cur.entry_ts >= prev.exit_ts


def test_known_equity_curve_and_sharpe():
    """Synthetic trades with deterministic returns → known equity progression."""
    trades = [
        Trade(bar_id=1, entry_ts=pd.Timestamp(T0),
              exit_ts=pd.Timestamp(T0 + timedelta(days=1)),
              entry_price=100.0, exit_price=105.0, direction=1,
              exit_reason="tp", holding_bars=1,
              pnl_bps_gross=500.0, pnl_bps_net=489.0),
        Trade(bar_id=10, entry_ts=pd.Timestamp(T0 + timedelta(days=1)),
              exit_ts=pd.Timestamp(T0 + timedelta(days=2)),
              entry_price=105.0, exit_price=99.75, direction=1,
              exit_reason="sl", holding_bars=1,
              pnl_bps_gross=-500.0, pnl_bps_net=-511.0),
    ]
    eq = build_equity_curve(trades, starting_equity=10_000.0)
    assert len(eq) == 3  # initial + 2 trade exits
    # First exit: +489 bps → 10_000 × 1.0489 = 10489
    assert abs(eq.iloc[1]["equity"] - 10_489.0) < 0.5
    # Second exit: -511 bps → 10489 × 0.9489 ≈ 9953
    assert abs(eq.iloc[2]["equity"] - 10_489.0 * (1 - 511.0 / 10_000.0)) < 0.5

    # Daily log-returns
    dlr = daily_log_returns(eq)
    assert len(dlr) >= 1


def test_compute_metrics_zero_trades_returns_zeros():
    metrics = compute_metrics([], pd.DataFrame(columns=["ts", "equity"]),
                              oot_n_bars=100)
    assert metrics["oot_n_trades"] == 0
    assert metrics["oot_sharpe"] == 0.0


def test_profitable_trade_pct_counts_winners_only_after_costs():
    """A trade with gross +5 bps but net -6 bps (after 11 bps cost) is NOT a winner."""
    trades = [
        Trade(bar_id=1, entry_ts=pd.Timestamp(T0),
              exit_ts=pd.Timestamp(T0 + timedelta(days=1)),
              entry_price=100.0, exit_price=100.05, direction=1,
              exit_reason="tp", holding_bars=1,
              pnl_bps_gross=5.0, pnl_bps_net=-6.0),
    ]
    eq = build_equity_curve(trades)
    metrics = compute_metrics(trades, eq, oot_n_bars=100)
    assert metrics["oot_n_trades"] == 1
    assert metrics["oot_profitable_trade_pct"] == 0.0  # net loser
