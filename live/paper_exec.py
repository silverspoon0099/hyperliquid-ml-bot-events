"""Paper trade execution state machine (DR v3.0.23).

Manages paper positions in events.paper_trades. Two main operations:

  1. open_trade(): record a new entry (direction, prices, probas, reason)
  2. check_open_trades(): for each open trade, walk forward through bars
     since entry and apply triple-barrier exit logic. Close any trade
     whose TB outcome is now determined.

Uses the same triple-barrier semantics as labels/triple_barrier.py:
  - LONG TP at entry × (1 + tp_pct); SL at entry × (1 - sl_pct)
  - SHORT TP at entry × (1 - tp_pct); SL at entry × (1 + sl_pct)
  - On both hit in same bar: ambiguous (treat as loss = negative PnL on the
    side hit second is undefined; we mark as 'ambiguous' and exit at close)
  - Timeout after vertical_bars: exit at that bar's close

PnL convention matches backtest.runner.Trade:
  pnl_bps_gross = direction × ((exit_price / entry_price) - 1) × 10000
  pnl_bps_net   = pnl_bps_gross - cost_bps_round_trip

Concurrency:
  max_concurrent: int (default 1) — refuses new entries while N open

All state is in Postgres (events.paper_trades). Restart-safe.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from data.db import get_connection, bars_table

LOG = logging.getLogger("live.paper_exec")


class PaperTradeManager:
    def __init__(
        self,
        session_id: str,
        asset: str,
        bar_threshold: float,
        tp_pct: float,
        sl_pct: float,
        vertical_bars: int,
        cost_bps_round_trip: float,
        position_size_usd: float,
        max_concurrent: int = 1,
    ):
        self.session_id = session_id
        self.asset = asset
        self.bar_threshold = bar_threshold
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.vertical_bars = vertical_bars
        self.cost_bps_round_trip = cost_bps_round_trip
        self.position_size_usd = position_size_usd
        self.max_concurrent = max_concurrent

    # ─────────────────────────────────────────────────────────────────
    # Open-trade operations
    # ─────────────────────────────────────────────────────────────────

    def n_open(self) -> int:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS n FROM events.paper_trades "
                    "WHERE session_id = %s AND status = 'open'",
                    (self.session_id,),
                )
                row = cur.fetchone()
                return int(row["n"])

    def open_trade(
        self,
        bar_id: int,
        entry_ts: datetime,
        entry_price: float,
        direction: int,
        p_long: float, p_short: float, p_neutral: float,
        entry_reason: str,
    ) -> Optional[int]:
        """Insert a new paper trade. Returns trade_id, or None if blocked
        by max_concurrent."""
        if self.n_open() >= self.max_concurrent:
            return None
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events.paper_trades
                      (session_id, bar_id_entry, entry_ts, entry_price, direction,
                       p_long, p_short, p_neutral, entry_reason, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'open')
                    RETURNING trade_id
                    """,
                    (self.session_id, int(bar_id), entry_ts, float(entry_price),
                     int(direction), float(p_long), float(p_short), float(p_neutral),
                     entry_reason),
                )
                trade_id = int(cur.fetchone()["trade_id"])
                conn.commit()
        return trade_id

    # ─────────────────────────────────────────────────────────────────
    # Exit-check operations (triple-barrier evaluation)
    # ─────────────────────────────────────────────────────────────────

    def _fetch_bars_since(self, bar_id_entry: int, limit_bars: int) -> pd.DataFrame:
        """Return up to `limit_bars` bars AFTER bar_id_entry (excluding entry),
        ordered by bar_close_ts ascending. Filtered to our bar_threshold."""
        tbl = bars_table(self.asset)
        sql = (
            f"SELECT bar_id, bar_close_ts, high, low, close "
            f"FROM {tbl} "
            f"WHERE threshold_pct = {self.bar_threshold} AND bar_id > %s "
            f"ORDER BY bar_close_ts ASC, bar_id ASC LIMIT %s"
        )
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (int(bar_id_entry), int(limit_bars)))
                rows = cur.fetchall()
        return pd.DataFrame(rows)

    def _evaluate_exit(
        self, direction: int, entry_price: float, bars_since: pd.DataFrame,
    ) -> Optional[dict]:
        """Apply triple-barrier logic to `bars_since`. Returns exit dict or
        None if still open (insufficient forward bars and no early hit).

        exit dict: {bar_id, exit_ts, exit_price, exit_reason, holding_bars}
        """
        if len(bars_since) == 0:
            return None
        # TP/SL prices conditional on direction
        if direction == 1:  # LONG
            tp_price = entry_price * (1.0 + self.tp_pct)
            sl_price = entry_price * (1.0 - self.sl_pct)
        else:  # SHORT (-1)
            tp_price = entry_price * (1.0 - self.tp_pct)
            sl_price = entry_price * (1.0 + self.sl_pct)

        for i, row in enumerate(bars_since.itertuples(index=False), start=1):
            high = float(row.high)
            low = float(row.low)
            if direction == 1:
                tp_hit = high >= tp_price
                sl_hit = low <= sl_price
            else:
                tp_hit = low <= tp_price
                sl_hit = high >= sl_price

            if tp_hit and sl_hit:
                # Whipsaw — both hit in same bar. Pessimistic: exit at close.
                return {
                    "bar_id": int(row.bar_id),
                    "exit_ts": row.bar_close_ts,
                    "exit_price": float(row.close),
                    "exit_reason": "ambiguous",
                    "holding_bars": i,
                }
            if tp_hit:
                return {
                    "bar_id": int(row.bar_id),
                    "exit_ts": row.bar_close_ts,
                    "exit_price": float(tp_price),
                    "exit_reason": "tp",
                    "holding_bars": i,
                }
            if sl_hit:
                return {
                    "bar_id": int(row.bar_id),
                    "exit_ts": row.bar_close_ts,
                    "exit_price": float(sl_price),
                    "exit_reason": "sl",
                    "holding_bars": i,
                }

        # No early exit. If we have enough forward bars → timeout at vertical_bars.
        if len(bars_since) >= self.vertical_bars:
            last = bars_since.iloc[self.vertical_bars - 1]
            return {
                "bar_id": int(last["bar_id"]),
                "exit_ts": last["bar_close_ts"],
                "exit_price": float(last["close"]),
                "exit_reason": "timeout",
                "holding_bars": int(self.vertical_bars),
            }
        # Still open
        return None

    def check_open_trades(self) -> list[dict]:
        """For each open trade, walk forward and apply TB. Close any whose
        outcome is now determined. Returns list of closed-trade dicts."""
        closed = []
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT trade_id, bar_id_entry, entry_price, direction "
                    "FROM events.paper_trades "
                    "WHERE session_id = %s AND status = 'open' "
                    "ORDER BY entry_ts",
                    (self.session_id,),
                )
                open_trades = cur.fetchall()

        for t in open_trades:
            bars_since = self._fetch_bars_since(
                t["bar_id_entry"], limit_bars=self.vertical_bars,
            )
            exit_info = self._evaluate_exit(
                direction=int(t["direction"]),
                entry_price=float(t["entry_price"]),
                bars_since=bars_since,
            )
            if exit_info is None:
                continue  # still open
            # Compute PnL
            entry_p = float(t["entry_price"])
            exit_p = exit_info["exit_price"]
            direction = int(t["direction"])
            pnl_bps_gross = direction * ((exit_p / entry_p) - 1.0) * 10_000.0
            pnl_bps_net = pnl_bps_gross - self.cost_bps_round_trip

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE events.paper_trades SET
                          status = 'exited',
                          exit_bar_id = %s,
                          exit_ts = %s,
                          exit_price = %s,
                          exit_reason = %s,
                          holding_bars = %s,
                          pnl_bps_gross = %s,
                          pnl_bps_net = %s,
                          updated_at = now()
                        WHERE trade_id = %s
                        """,
                        (
                            exit_info["bar_id"], exit_info["exit_ts"], exit_info["exit_price"],
                            exit_info["exit_reason"], exit_info["holding_bars"],
                            float(pnl_bps_gross), float(pnl_bps_net),
                            int(t["trade_id"]),
                        ),
                    )
                    conn.commit()

            closed_record = {
                "trade_id": int(t["trade_id"]),
                "bar_id_entry": int(t["bar_id_entry"]),
                "direction": direction,
                "entry_price": entry_p,
                **exit_info,
                "pnl_bps_gross": float(pnl_bps_gross),
                "pnl_bps_net": float(pnl_bps_net),
            }
            closed.append(closed_record)
            LOG.info("CLOSED trade %d: dir=%+d entry=%.2f exit=%.2f reason=%s "
                     "hold=%d pnl_net=%+.1f bps",
                     int(t["trade_id"]), direction, entry_p, exit_p,
                     exit_info["exit_reason"], exit_info["holding_bars"], pnl_bps_net)
        return closed

    # ─────────────────────────────────────────────────────────────────
    # Session-level queries
    # ─────────────────────────────────────────────────────────────────

    def session_summary(self) -> dict:
        """Aggregate stats for this session's trades."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      COUNT(*) AS n_total,
                      SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS n_open,
                      SUM(CASE WHEN status='exited' THEN 1 ELSE 0 END) AS n_exited,
                      SUM(CASE WHEN status='exited' AND pnl_bps_net > 0 THEN 1 ELSE 0 END) AS n_winners,
                      AVG(pnl_bps_net) FILTER (WHERE status='exited') AS mean_pnl_bps_net,
                      SUM(pnl_bps_net) FILTER (WHERE status='exited') AS total_pnl_bps,
                      MIN(entry_ts) AS first_entry_ts,
                      MAX(updated_at) AS last_updated
                    FROM events.paper_trades
                    WHERE session_id = %s
                    """,
                    (self.session_id,),
                )
                row = cur.fetchone()
        if row is None:
            return {}
        n_exited = int(row["n_exited"] or 0)
        n_winners = int(row["n_winners"] or 0)
        return {
            "n_total": int(row["n_total"] or 0),
            "n_open": int(row["n_open"] or 0),
            "n_exited": n_exited,
            "n_winners": n_winners,
            "win_pct": (100.0 * n_winners / n_exited) if n_exited > 0 else 0.0,
            "mean_pnl_bps_net": float(row["mean_pnl_bps_net"] or 0.0),
            "total_pnl_bps": float(row["total_pnl_bps"] or 0.0),
            "first_entry_ts": row["first_entry_ts"],
            "last_updated": row["last_updated"],
        }

    def todays_realized_loss_pct(self) -> float:
        """Today's realized PnL as fraction of position_size_usd, for DD-kill check.
        Positive value = positive PnL; negative = loss. Returns loss fraction
        (e.g., -0.07 = 7% loss day)."""
        today = datetime.now(timezone.utc).date()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(pnl_bps_net), 0) AS total_bps
                    FROM events.paper_trades
                    WHERE session_id = %s AND status = 'exited'
                      AND exit_ts >= %s::date AND exit_ts < (%s::date + INTERVAL '1 day')
                    """,
                    (self.session_id, today, today),
                )
                row = cur.fetchone()
        total_bps = float(row["total_bps"]) if row else 0.0
        # As percent of position_size (each trade is 100% of position so bps direct):
        return total_bps / 10_000.0  # bps → fraction


def start_session(
    session_id: str,
    asset: str,
    model_artifact_path: str,
    bar_threshold: float,
    tp_pct: float,
    sl_pct: float,
    vertical_bars: int,
    confidence_threshold: float,
    position_size_usd: float,
    cost_bps_round_trip: float,
    max_daily_loss_pct: float,
    notes: str = "",
) -> None:
    """INSERT a new row in events.paper_sessions. Idempotent: skips if exists."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO events.paper_sessions
                  (session_id, asset, model_artifact_path, bar_threshold,
                   tp_pct, sl_pct, vertical_bars, confidence_threshold,
                   position_size_usd, cost_bps_round_trip, max_daily_loss_pct, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id, asset, model_artifact_path, float(bar_threshold),
                 float(tp_pct), float(sl_pct), int(vertical_bars), float(confidence_threshold),
                 float(position_size_usd), float(cost_bps_round_trip),
                 float(max_daily_loss_pct), notes),
            )
            conn.commit()


def end_session(session_id: str, end_reason: str) -> None:
    """Mark a session as ended."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE events.paper_sessions
                SET ended_at = now(), end_reason = %s
                WHERE session_id = %s AND ended_at IS NULL
                """,
                (end_reason, session_id),
            )
            conn.commit()
