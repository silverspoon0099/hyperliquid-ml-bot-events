"""Trade tracking dashboard — Streamlit app (DR v3.0.25).

Pages:
  1. Overview          — top-line KPIs + recent activity
  2. Paper Trading     — sessions, trades, decisions, equity curve
  3. Backtest          — historical runs, per-fold breakdown, all trades
  4. Side-by-side      — paper vs backtest comparison

Run:
    .venv/bin/streamlit run dashboard/app.py --server.port 8501

Read-only — no DB writes from the dashboard.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root on sys.path so `data.db` etc. import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import queries as Q


# ─────────────────────────────────────────────────────────────────────
# Page config + shared helpers
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ml-bot-events dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _fmt_int(n) -> str:
    if n is None:
        return "—"
    return f"{int(n):,}"


def _fmt_bps(b) -> str:
    if b is None or pd.isna(b):
        return "—"
    return f"{float(b):+.1f}"


def _fmt_pct(p) -> str:
    if p is None or pd.isna(p):
        return "—"
    return f"{float(p):.1f}%"


def _direction_label(d) -> str:
    if d == 1:
        return "LONG"
    if d == -1:
        return "SHORT"
    return "—"


# ─────────────────────────────────────────────────────────────────────
# Sidebar — page picker + global refresh
# ─────────────────────────────────────────────────────────────────────
PAGES = ["Overview", "Paper Trading", "Backtest", "Side-by-side"]

with st.sidebar:
    st.title("📊 ml-bot-events")
    st.caption("Trade tracking dashboard")
    page = st.radio("Page", PAGES, index=0)
    st.markdown("---")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"UTC now: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")


# Cache wrappers — 30-second TTL so the dashboard sees fresh paper data
@st.cache_data(ttl=30)
def _kpis():
    return Q.kpi_snapshot()

@st.cache_data(ttl=30)
def _paper_sessions():
    return Q.list_paper_sessions()

@st.cache_data(ttl=30)
def _paper_trades(session_id, status):
    return Q.paper_trades(session_id=session_id, status_filter=status, limit=1000)

@st.cache_data(ttl=30)
def _paper_decisions(session_id, traded_only):
    return Q.paper_decisions(session_id=session_id, traded_only=traded_only, limit=1000)

@st.cache_data(ttl=30)
def _paper_equity(session_id):
    return Q.paper_equity_curve(session_id)

@st.cache_data(ttl=60)
def _backtest_runs():
    return Q.list_backtest_runs()

@st.cache_data(ttl=60)
def _backtest_trades(run_id, fold_id):
    return Q.backtest_trades(run_id=run_id, fold_id=fold_id, limit=10_000)

@st.cache_data(ttl=60)
def _backtest_equity(run_id):
    return Q.backtest_equity_curve(run_id)

@st.cache_data(ttl=60)
def _backtest_folds(run_id):
    return Q.backtest_fold_summary(run_id)


# ─────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────
def render_overview() -> None:
    st.title("📊 Overview")

    k = _kpis()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paper sessions", _fmt_int(k["n_paper_sessions"]))
    c2.metric("Paper trades total", _fmt_int(k["paper_n_trades"]),
              help=f"open={k['paper_n_open']}, exited={k['paper_n_exited']}")
    c3.metric("Paper win %", _fmt_pct(k["paper_win_pct"]),
              help=f"{k['paper_n_winners']} winners / {k['paper_n_exited']} exits")
    c4.metric("Paper total PnL", _fmt_bps(k["paper_total_pnl_bps"]) + " bps",
              help=f"mean={_fmt_bps(k['paper_mean_pnl_bps'])} bps/trade")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backtest runs", _fmt_int(k["n_backtest_runs"]))
    c2.metric("Backtest trades", _fmt_int(k["backtest_n_trades"]))
    c3.metric("Champion baseline", "+1.204 Sharpe",
              help="L0 + 1.5% bars + thr=0.58 + TB=0.03 (DR v3.0.20)")
    c4.metric("Live session", "btc_thr015_20260518",
              help="Currently running under pm2")

    st.markdown("---")

    # Recent paper trades preview
    st.subheader("Recent paper trades")
    sessions = _paper_sessions()
    if sessions.empty:
        st.info("No paper sessions yet.")
    else:
        recent = _paper_trades(session_id=None, status=None).head(20)
        if recent.empty:
            st.info("No paper trades yet. The daemon is collecting bars; trades will appear when L0 confidence exceeds 0.58.")
        else:
            st.dataframe(_pretty_trades_df(recent), width="stretch", height=420)


# ─────────────────────────────────────────────────────────────────────
# PAPER TRADING
# ─────────────────────────────────────────────────────────────────────
def render_paper() -> None:
    st.title("📝 Paper Trading")

    sessions = _paper_sessions()
    if sessions.empty:
        st.warning("No paper sessions in DB yet. Start one via:\n```bash\npm2 start ecosystem.config.js\n```")
        return

    # Session picker (default to most-recent)
    session_options = sessions["session_id"].tolist()
    session_id = st.selectbox(
        "Session", session_options, index=0,
        help="Most-recent at top. Each row in events.paper_sessions.",
    )

    # Session metadata
    s = sessions.loc[sessions["session_id"] == session_id].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades total", _fmt_int(s["n_total"]),
              help=f"open={int(s['n_open'])} exited={int(s['n_exited'])}")
    c2.metric("Win %",
              _fmt_pct(100.0 * s["n_winners"] / max(s["n_exited"], 1)),
              help=f"{int(s['n_winners'])} / {int(s['n_exited'])}")
    c3.metric("Mean PnL", _fmt_bps(s["mean_pnl_bps"]) + " bps")
    c4.metric("Total PnL", _fmt_bps(s["total_pnl_bps"]) + " bps")

    with st.expander("Session config"):
        st.json({
            "asset": s["asset"],
            "bar_threshold": float(s["bar_threshold"]),
            "tp_pct": float(s["tp_pct"]), "sl_pct": float(s["sl_pct"]),
            "vertical_bars": int(s["vertical_bars"]),
            "confidence_threshold": float(s["confidence_threshold"]),
            "position_size_usd": float(s["position_size_usd"]),
            "cost_bps_round_trip": float(s["cost_bps_round_trip"]),
            "started_at": str(s["started_at"]),
            "ended_at": str(s["ended_at"]) if s["ended_at"] is not None else "<running>",
            "end_reason": s["end_reason"],
            "notes": s["notes"],
        })

    st.markdown("---")

    # Equity curve
    st.subheader("Cumulative PnL (bps net)")
    eq = _paper_equity(session_id)
    if eq.empty:
        st.info("No exited trades yet.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq["exit_ts"], y=eq["cum_pnl_bps"], mode="lines+markers",
            name="cum PnL bps", line=dict(width=2),
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            xaxis_title="Exit time", yaxis_title="Cumulative bps net",
            height=320, margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    # Tabs: Trades vs Decisions
    tab_t, tab_d = st.tabs(["📋 Trades", "🔍 Decisions"])

    with tab_t:
        status = st.selectbox(
            "Status filter", ["all", "open", "exited"], index=0, key="paper_status",
        )
        df = _paper_trades(session_id=session_id, status=status)
        if df.empty:
            st.info("No trades match the filter.")
        else:
            st.dataframe(_pretty_trades_df(df), width="stretch", height=420)

    with tab_d:
        traded_only = st.checkbox("Show only traded signals", value=False, key="paper_traded_only")
        df = _paper_decisions(session_id=session_id, traded_only=traded_only)
        if df.empty:
            st.info("No decisions logged for this session yet.")
        else:
            df = df.copy()
            df["max_prob"] = df["max_prob"].apply(lambda v: f"{v:.3f}")
            df["p_long"]  = df["p_long"].apply(lambda v: f"{v:.3f}")
            df["p_short"] = df["p_short"].apply(lambda v: f"{v:.3f}")
            df["p_neutral"] = df["p_neutral"].apply(lambda v: f"{v:.3f}")
            st.dataframe(df, width="stretch", height=420)


# ─────────────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────────────
def render_backtest() -> None:
    st.title("📈 Backtest")

    runs = _backtest_runs()
    if runs.empty:
        st.warning("No backtest runs persisted yet. Run:\n```bash\npython -m scripts.backfill_backtest_trades\n```")
        return

    run_options = runs["run_id"].tolist()
    run_id = st.selectbox("Run", run_options, index=0,
                           help="Most-recent backfill at top.")
    r = runs.loc[runs["run_id"] == run_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe (all folds)", f"{r['sharpe_all_folds']:+.3f}"
              if r["sharpe_all_folds"] is not None else "—")
    c2.metric("Sharpe (nonzero)", f"{r['sharpe_nonzero']:+.3f}"
              if r["sharpe_nonzero"] is not None else "—")
    c3.metric("Trades total", _fmt_int(r["n_actual_trades"]))
    c4.metric("Win %", _fmt_pct(r["win_pct_mean"]))

    with st.expander("Run config"):
        st.json({k: (str(v) if not isinstance(v, (int, float)) else v)
                 for k, v in r.to_dict().items()})

    st.markdown("---")

    # Equity curve
    st.subheader("Cumulative PnL across all folds (bps net, time-ordered)")
    eq = _backtest_equity(run_id)
    if eq.empty:
        st.info("No trades for this run.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["exit_ts"], y=eq["cum_pnl_bps"], mode="lines",
        name="cum PnL bps", line=dict(width=2),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    # Mark fold boundaries
    for f_id, group in eq.groupby("fold_id"):
        fig.add_annotation(
            x=group["exit_ts"].iloc[0],
            y=group["cum_pnl_bps"].iloc[0],
            text=f"f{f_id}",
            showarrow=False, font=dict(size=10, color="gray"),
            yshift=15,
        )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_title="Exit time", yaxis_title="Cumulative bps net")
    st.plotly_chart(fig, width="stretch")

    # Per-fold summary
    st.subheader("Per-fold breakdown")
    folds = _backtest_folds(run_id)
    if folds.empty:
        st.info("No fold breakdown available.")
    else:
        folds_disp = folds.copy()
        folds_disp["win_pct"] = folds_disp["win_pct"].apply(lambda v: f"{v:.1f}%")
        folds_disp["mean_pnl_bps"] = folds_disp["mean_pnl_bps"].apply(lambda v: f"{v:+.1f}")
        folds_disp["total_pnl_bps"] = folds_disp["total_pnl_bps"].apply(lambda v: f"{v:+.0f}")
        st.dataframe(folds_disp, width="stretch", height=min(600, 35 * (len(folds_disp) + 1)))

    # All trades (with fold filter)
    st.subheader("All trades")
    fold_filter = st.selectbox(
        "Filter by fold", ["all"] + [int(f) for f in folds["fold_id"].tolist()] if not folds.empty else ["all"],
        index=0,
    )
    fold_id_param = None if fold_filter == "all" else int(fold_filter)
    df = _backtest_trades(run_id=run_id, fold_id=fold_id_param)
    if df.empty:
        st.info("No trades match.")
    else:
        st.dataframe(_pretty_trades_df(df), width="stretch", height=480)


# ─────────────────────────────────────────────────────────────────────
# SIDE-BY-SIDE
# ─────────────────────────────────────────────────────────────────────
def render_side_by_side() -> None:
    st.title("🆚 Side-by-side: Paper vs Backtest")

    sessions = _paper_sessions()
    runs = _backtest_runs()

    if sessions.empty and runs.empty:
        st.warning("No data yet. Start a paper session and/or backfill a backtest run.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Paper session")
        if sessions.empty:
            st.info("No paper sessions.")
            session_id = None
        else:
            session_id = st.selectbox(
                "Session", sessions["session_id"].tolist(), key="sxs_session",
            )
            s = sessions.loc[sessions["session_id"] == session_id].iloc[0]
            st.metric("Trades", _fmt_int(s["n_total"]))
            st.metric("Win %", _fmt_pct(100.0 * s["n_winners"] / max(s["n_exited"], 1)))
            st.metric("Mean PnL bps", _fmt_bps(s["mean_pnl_bps"]))
            st.metric("Total PnL bps", _fmt_bps(s["total_pnl_bps"]))
    with c2:
        st.subheader("Backtest run")
        if runs.empty:
            st.info("No backtest runs.")
            run_id = None
        else:
            run_id = st.selectbox(
                "Run", runs["run_id"].tolist(), key="sxs_run",
            )
            r = runs.loc[runs["run_id"] == run_id].iloc[0]
            st.metric("Sharpe (all folds)",
                      f"{r['sharpe_all_folds']:+.3f}" if r["sharpe_all_folds"] is not None else "—")
            st.metric("Trades", _fmt_int(r["n_actual_trades"]))
            st.metric("Win %", _fmt_pct(r["win_pct_mean"]))

    if session_id and run_id:
        st.markdown("---")
        st.subheader("Cumulative PnL — overlay")
        paper_eq = _paper_equity(session_id)
        bt_eq = _backtest_equity(run_id)
        fig = go.Figure()
        if not bt_eq.empty:
            fig.add_trace(go.Scatter(
                x=bt_eq["exit_ts"], y=bt_eq["cum_pnl_bps"], mode="lines",
                name="Backtest", line=dict(width=2, color="#1f77b4"),
            ))
        if not paper_eq.empty:
            fig.add_trace(go.Scatter(
                x=paper_eq["exit_ts"], y=paper_eq["cum_pnl_bps"],
                mode="lines+markers", name="Paper",
                line=dict(width=2, color="#ff7f0e"),
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10),
                          xaxis_title="Exit time", yaxis_title="Cum bps net")
        st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────
# Shared: trade table formatting
# ─────────────────────────────────────────────────────────────────────
def _pretty_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "direction" in out:
        out["direction"] = out["direction"].apply(_direction_label)
    for col in ("entry_price", "exit_price"):
        if col in out:
            out[col] = out[col].apply(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
    for col in ("pnl_bps_gross", "pnl_bps_net"):
        if col in out:
            out[col] = out[col].apply(lambda v: f"{v:+.1f}" if pd.notna(v) else "—")
    for col in ("p_long", "p_short", "p_neutral"):
        if col in out:
            out[col] = out[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    return out


# ─────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────
if page == "Overview":
    render_overview()
elif page == "Paper Trading":
    render_paper()
elif page == "Backtest":
    render_backtest()
elif page == "Side-by-side":
    render_side_by_side()
