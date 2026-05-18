"""Postgres queries for the trading dashboard (DR v3.0.25).

All DB access for the Streamlit app lives here. Reuses
data.db.get_connection() for connection pooling.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from data.db import get_connection


# ─────────────────────────────────────────────────────────────────────
# Paper trading queries
# ─────────────────────────────────────────────────────────────────────

def list_paper_sessions() -> pd.DataFrame:
    """All paper sessions with summary stats joined in."""
    sql = """
    SELECT
        s.session_id, s.asset, s.bar_threshold, s.tp_pct, s.sl_pct,
        s.vertical_bars, s.confidence_threshold, s.position_size_usd,
        s.cost_bps_round_trip, s.started_at, s.ended_at, s.end_reason, s.notes,
        COALESCE(t.n_total, 0) AS n_total,
        COALESCE(t.n_open, 0)  AS n_open,
        COALESCE(t.n_exited, 0) AS n_exited,
        COALESCE(t.n_winners, 0) AS n_winners,
        COALESCE(t.mean_pnl_bps, 0) AS mean_pnl_bps,
        COALESCE(t.total_pnl_bps, 0) AS total_pnl_bps
    FROM events.paper_sessions s
    LEFT JOIN (
        SELECT
            session_id,
            COUNT(*) AS n_total,
            SUM((status='open')::int) AS n_open,
            SUM((status='exited')::int) AS n_exited,
            SUM((status='exited' AND pnl_bps_net > 0)::int) AS n_winners,
            AVG(pnl_bps_net) FILTER (WHERE status='exited') AS mean_pnl_bps,
            SUM(pnl_bps_net) FILTER (WHERE status='exited') AS total_pnl_bps
        FROM events.paper_trades
        GROUP BY session_id
    ) t USING (session_id)
    ORDER BY s.started_at DESC
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return pd.DataFrame(cur.fetchall())


def paper_trades(session_id: Optional[str] = None,
                  status_filter: Optional[str] = None,
                  limit: int = 500) -> pd.DataFrame:
    """Paper trades, optionally filtered."""
    conds, params = [], []
    if session_id:
        conds.append("session_id = %s")
        params.append(session_id)
    if status_filter and status_filter != "all":
        conds.append("status = %s")
        params.append(status_filter)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"""
        SELECT trade_id, session_id, bar_id_entry, entry_ts, entry_price,
               direction, p_long, p_short, p_neutral, status, exit_ts,
               exit_price, exit_reason, holding_bars, pnl_bps_gross, pnl_bps_net
        FROM events.paper_trades
        {where}
        ORDER BY entry_ts DESC
        LIMIT %s
    """
    params.append(limit)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return pd.DataFrame(cur.fetchall())


def paper_decisions(session_id: Optional[str] = None,
                     traded_only: bool = False,
                     limit: int = 500) -> pd.DataFrame:
    conds, params = [], []
    if session_id:
        conds.append("session_id = %s")
        params.append(session_id)
    if traded_only:
        conds.append("traded = true")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"""
        SELECT session_id, bar_id, decided_at, p_long, p_short, p_neutral,
               max_prob, argmax_class, traded, skip_reason, trade_id
        FROM events.paper_decisions
        {where}
        ORDER BY decided_at DESC
        LIMIT %s
    """
    params.append(limit)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return pd.DataFrame(cur.fetchall())


def paper_equity_curve(session_id: str) -> pd.DataFrame:
    """Cumulative PnL curve for one session.
    Returns (exit_ts, pnl_bps_net, cum_pnl_bps_net)."""
    sql = """
        SELECT exit_ts, pnl_bps_net,
               SUM(pnl_bps_net) OVER (ORDER BY exit_ts) AS cum_pnl_bps
        FROM events.paper_trades
        WHERE session_id = %s AND status = 'exited'
        ORDER BY exit_ts
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))
            return pd.DataFrame(cur.fetchall())


# ─────────────────────────────────────────────────────────────────────
# Backtest queries
# ─────────────────────────────────────────────────────────────────────

def list_backtest_runs() -> pd.DataFrame:
    """All persisted backtest runs."""
    sql = """
    SELECT
        r.run_id, r.asset, r.bar_threshold, r.tb_pct, r.confidence_threshold,
        r.sharpe_all_folds, r.sharpe_nonzero, r.n_trades_total, r.win_pct_mean,
        r.feature_set, r.notes, r.created_at,
        COALESCE(t.n_actual, 0) AS n_actual_trades
    FROM events.backtest_runs r
    LEFT JOIN (
        SELECT run_id, COUNT(*) AS n_actual
        FROM events.backtest_trades
        GROUP BY run_id
    ) t USING (run_id)
    ORDER BY r.created_at DESC
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return pd.DataFrame(cur.fetchall())


def backtest_trades(run_id: Optional[str] = None,
                     fold_id: Optional[int] = None,
                     limit: int = 1000) -> pd.DataFrame:
    conds, params = [], []
    if run_id:
        conds.append("run_id = %s")
        params.append(run_id)
    if fold_id is not None:
        conds.append("fold_id = %s")
        params.append(fold_id)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"""
        SELECT trade_id, run_id, fold_id, bar_id_entry, entry_ts, entry_price,
               direction, p_long, p_short, p_neutral, exit_ts, exit_price,
               exit_reason, holding_bars, pnl_bps_gross, pnl_bps_net, label
        FROM events.backtest_trades
        {where}
        ORDER BY entry_ts DESC
        LIMIT %s
    """
    params.append(limit)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return pd.DataFrame(cur.fetchall())


def backtest_equity_curve(run_id: str) -> pd.DataFrame:
    """Cumulative PnL across all folds, time-ordered."""
    sql = """
        SELECT exit_ts, fold_id, pnl_bps_net,
               SUM(pnl_bps_net) OVER (ORDER BY exit_ts) AS cum_pnl_bps
        FROM events.backtest_trades
        WHERE run_id = %s
        ORDER BY exit_ts
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id,))
            return pd.DataFrame(cur.fetchall())


def backtest_fold_summary(run_id: str) -> pd.DataFrame:
    """Per-fold summary: n_trades, win%, mPnL, fold Sharpe-equivalent."""
    sql = """
        SELECT fold_id,
               COUNT(*) AS n_trades,
               SUM((pnl_bps_net > 0)::int)::float / COUNT(*) * 100 AS win_pct,
               AVG(pnl_bps_net) AS mean_pnl_bps,
               SUM(pnl_bps_net) AS total_pnl_bps,
               MIN(entry_ts) AS first_entry,
               MAX(exit_ts) AS last_exit
        FROM events.backtest_trades
        WHERE run_id = %s
        GROUP BY fold_id
        ORDER BY fold_id
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id,))
            return pd.DataFrame(cur.fetchall())


# ─────────────────────────────────────────────────────────────────────
# Cross queries
# ─────────────────────────────────────────────────────────────────────

def kpi_snapshot() -> dict:
    """Top-line KPIs across all sessions/runs."""
    out = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Paper
            cur.execute("SELECT COUNT(*) AS n FROM events.paper_sessions")
            out["n_paper_sessions"] = int(cur.fetchone()["n"])
            cur.execute(
                "SELECT "
                "  COUNT(*) AS n_trades, "
                "  SUM((status='open')::int) AS n_open, "
                "  SUM((status='exited')::int) AS n_exited, "
                "  SUM((status='exited' AND pnl_bps_net > 0)::int) AS n_winners, "
                "  AVG(pnl_bps_net) FILTER (WHERE status='exited') AS mean_pnl_bps, "
                "  SUM(pnl_bps_net) FILTER (WHERE status='exited') AS total_pnl_bps "
                "FROM events.paper_trades"
            )
            r = cur.fetchone() or {}
            out["paper_n_trades"] = int(r.get("n_trades") or 0)
            out["paper_n_open"] = int(r.get("n_open") or 0)
            out["paper_n_exited"] = int(r.get("n_exited") or 0)
            out["paper_n_winners"] = int(r.get("n_winners") or 0)
            out["paper_mean_pnl_bps"] = float(r.get("mean_pnl_bps") or 0.0)
            out["paper_total_pnl_bps"] = float(r.get("total_pnl_bps") or 0.0)
            out["paper_win_pct"] = (
                100.0 * out["paper_n_winners"] / out["paper_n_exited"]
                if out["paper_n_exited"] > 0 else 0.0
            )
            # Backtest
            cur.execute("SELECT COUNT(*) AS n FROM events.backtest_runs")
            out["n_backtest_runs"] = int(cur.fetchone()["n"])
            cur.execute("SELECT COUNT(*) AS n FROM events.backtest_trades")
            out["backtest_n_trades"] = int(cur.fetchone()["n"])
    return out
