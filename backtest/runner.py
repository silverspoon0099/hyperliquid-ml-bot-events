"""PnL / equity curve / Sharpe (spec §11 + DR v3.0.9 §8-§10).

Trade signal rule per §8.4 + max 1 concurrent (§13).
PnL via label exit_price (the actual triple-barrier outcome).
Cost: 11 bps round-trip per trade.
Sharpe: daily-resample equity × √252.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


LABEL_LONG = 0
LABEL_SHORT = 1
LABEL_NEUTRAL = 2


@dataclass
class Trade:
    bar_id: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int      # +1 LONG, -1 SHORT
    exit_reason: str    # tp / sl / timeout / ambiguous
    holding_bars: int
    pnl_bps_gross: float
    pnl_bps_net: float


def simulate_trades(
    predictions: pd.DataFrame,        # bar_id, p_long, p_short, p_neutral
    bars_df: pd.DataFrame,            # bar_id, bar_close_ts, close
    labels_df: pd.DataFrame,          # bar_id, exit_bar_id, exit_price,
                                       #         exit_reason, holding_bars, label
    confidence_threshold: float = 0.60,
    cost_bps_round_trip: float = 11.0,
    max_concurrent: int = 1,
) -> list[Trade]:
    """Walk OOT bars in chronological order; take trades per DR §8."""
    df = predictions.merge(
        bars_df[["bar_id", "bar_close_ts", "close"]], on="bar_id", how="inner"
    ).merge(
        labels_df[["bar_id", "exit_bar_id", "exit_price", "exit_reason",
                   "holding_bars", "label"]],
        on="bar_id", how="inner",
    ).sort_values("bar_close_ts").reset_index(drop=True)

    bar_id_to_ts = bars_df.set_index("bar_id")["bar_close_ts"]

    trades: list[Trade] = []
    open_exit_times: list[pd.Timestamp] = []

    for _, row in df.iterrows():
        entry_ts = pd.Timestamp(row["bar_close_ts"])
        # Close any positions whose exit_ts has passed
        open_exit_times = [t for t in open_exit_times if t > entry_ts]

        if len(open_exit_times) >= max_concurrent:
            continue  # in position; skip new signal

        p_long = float(row["p_long"])
        p_short = float(row["p_short"])
        if p_long > confidence_threshold:
            direction = 1
        elif p_short > confidence_threshold:
            direction = -1
        else:
            continue  # no trade

        if pd.isna(row["exit_bar_id"]) or pd.isna(row["exit_price"]):
            continue  # UNLABELABLE — should already be filtered, defensive

        exit_bar_id = int(row["exit_bar_id"])
        exit_price = float(row["exit_price"])
        exit_ts = pd.Timestamp(bar_id_to_ts.loc[exit_bar_id])
        entry_price = float(row["close"])

        gross_bps = direction * ((exit_price / entry_price) - 1.0) * 10000.0
        net_bps = gross_bps - cost_bps_round_trip

        trades.append(Trade(
            bar_id=int(row["bar_id"]),
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            exit_reason=str(row["exit_reason"]),
            holding_bars=int(row["holding_bars"]),
            pnl_bps_gross=float(gross_bps),
            pnl_bps_net=float(net_bps),
        ))
        open_exit_times.append(exit_ts)

    return trades


def build_equity_curve(
    trades: list[Trade], starting_equity: float = 10_000.0,
    oot_start_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Equity curve indexed by trade exit time."""
    if not trades:
        if oot_start_ts is not None:
            return pd.DataFrame({"ts": [oot_start_ts], "equity": [starting_equity]})
        return pd.DataFrame(columns=["ts", "equity"])

    sorted_trades = sorted(trades, key=lambda t: t.exit_ts)
    rows = [{"ts": oot_start_ts or sorted_trades[0].entry_ts,
             "equity": starting_equity}]
    cur = starting_equity
    for t in sorted_trades:
        cur *= 1.0 + t.pnl_bps_net / 10000.0
        rows.append({"ts": t.exit_ts, "equity": cur})
    return pd.DataFrame(rows)


def daily_log_returns(equity_curve: pd.DataFrame) -> pd.Series:
    """Resample equity curve to daily, then take log-returns."""
    if equity_curve.empty:
        return pd.Series([], dtype=float)
    eq = equity_curve.set_index("ts").sort_index()["equity"]
    eq = eq[~eq.index.duplicated(keep="last")]
    daily = eq.resample("1D").last().ffill()
    return np.log(daily / daily.shift(1)).dropna()


def compute_metrics(
    trades: list[Trade],
    equity_curve: pd.DataFrame,
    oot_n_bars: int,
) -> dict:
    """Per spec §10.4 metrics."""
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "oot_sharpe": 0.0, "oot_sortino": 0.0, "oot_max_dd": 0.0,
            "oot_pct_time_in_market": 0.0, "oot_n_trades": 0,
            "oot_profitable_trade_pct": 0.0, "oot_annual_return": 0.0,
        }

    daily_ret = daily_log_returns(equity_curve)

    sharpe = 0.0
    sortino = 0.0
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
    neg_ret = daily_ret[daily_ret < 0]
    if len(neg_ret) > 1 and neg_ret.std() > 0:
        sortino = float(daily_ret.mean() / neg_ret.std() * np.sqrt(252))

    eq = equity_curve["equity"].astype(float)
    if len(eq) > 1:
        running_max = eq.cummax()
        drawdown = (eq - running_max) / running_max
        max_dd = float(drawdown.min())
    else:
        max_dd = 0.0

    total_holding = sum(t.holding_bars for t in trades)
    pct_time = (total_holding / max(oot_n_bars, 1)) * 100.0
    n_winners = sum(1 for t in trades if t.pnl_bps_net > 0)
    profitable_pct = n_winners / n_trades * 100.0

    if len(daily_ret) > 0:
        cum_ret = float((1.0 + daily_ret).prod() - 1.0) if len(daily_ret) > 0 else 0.0
        n_days = max(len(daily_ret), 1)
        annual_return = (1.0 + cum_ret) ** (252.0 / n_days) - 1.0
    else:
        annual_return = 0.0

    return {
        "oot_sharpe": sharpe,
        "oot_sortino": sortino,
        "oot_max_dd": max_dd,
        "oot_pct_time_in_market": pct_time,
        "oot_n_trades": n_trades,
        "oot_profitable_trade_pct": profitable_pct,
        "oot_annual_return": float(annual_return),
    }


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades])
