"""Paper-trading daemon loop (DR v3.0.23).

Periodic orchestrator that:
  1. Polls Binance REST for new aggTrades, inserts into DB (~10 sec)
  2. Rebuilds CUSUM bars for the current month (~1-5 min)
  3. Computes features for any new bars since last decision
  4. Runs L0 inference + Platt calibration on each new bar
  5. Applies confidence threshold → trade decision → opens paper position
  6. Checks open paper positions → applies TB exit logic → closes
  7. Checks daily drawdown → auto-halt if loss exceeds threshold
  8. Checks HALT flag file → graceful shutdown
  9. Sleeps until next interval

Run as a foreground process or under systemd / tmux for unattended operation.

Configuration via CLI:
    --session-id     ID for this paper session (default: btc_thr015_v<N>_<YYYYMMDD>)
    --artifact       Path to L0 .pkl artifact (default: latest in models/)
    --poll-seconds   Poll cadence (default: 600 = 10 minutes)
    --halt-file      File path; touch it to gracefully stop the daemon
    --log-dir        JSONL audit log directory
    --notes          Free-text session notes

Stop conditions:
  - HALT flag file exists → graceful halt
  - Today's realized loss exceeds max_daily_loss_pct → auto-halt for 24h
  - SIGINT (Ctrl-C) → graceful halt
"""
from __future__ import annotations

import argparse
import logging
import pickle
import signal
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.db import init_paper_schema, close_pool, bars_table, ticks_table, get_connection
from model.lgbm import apply_platt
from live import binance_rest
from live.paper_exec import PaperTradeManager, start_session, end_session
from live.audit_log import AuditLog

LOG = logging.getLogger("scripts.run_paper_trading_loop")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_artifact(path: Path) -> dict:
    """Load pickled L0 artifact, reconstruct Booster from string."""
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    booster_str = artifact["booster_bytes"].decode("utf-8")
    booster = lgb.Booster(model_str=booster_str)
    artifact["booster"] = booster
    return artifact


def _default_artifact_path(sym: str, bar_threshold: float) -> Path:
    model_dir = PROJECT_ROOT / "data" / "storage" / "models"
    base = f"l0_{sym.lower()}_thr{int(bar_threshold*1000):03d}"
    candidates = sorted(model_dir.glob(f"{base}_v*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No L0 artifact found matching {base}_v*.pkl in {model_dir}. "
            f"Run: python -m scripts.train_and_persist_l0"
        )
    return candidates[-1]  # highest version


def _new_bars_since(last_bar_id: int, since_ts: datetime,
                     asset: str, bar_threshold: float) -> pd.DataFrame:
    """Bars with bar_id > last_bar_id AND bar_close_ts >= since_ts.

    The since_ts floor prevents the daemon from making retroactive decisions
    on all historical bars when there are no prior decisions yet (fresh session).
    """
    tbl = bars_table(asset)
    sql = (f"SELECT bar_id, bar_open_ts, bar_close_ts, open, high, low, close, "
           f"       volume, n_trades, cusum_pos, cusum_neg "
           f"FROM {tbl} "
           f"WHERE threshold_pct = {bar_threshold} "
           f"  AND bar_id > %s AND bar_close_ts >= %s "
           f"ORDER BY bar_close_ts ASC, bar_id ASC")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (int(last_bar_id), since_ts))
            return pd.DataFrame(cur.fetchall())


def _all_bars_for_features(asset: str, bar_threshold: float) -> pd.DataFrame:
    """Full bars for feature computation (need warmup for EMAs etc)."""
    tbl = bars_table(asset)
    sql = (f"SELECT bar_id, bar_open_ts, bar_close_ts, open, high, low, close, "
           f"       volume, n_trades, cusum_pos, cusum_neg "
           f"FROM {tbl} "
           f"WHERE threshold_pct = {bar_threshold} "
           f"ORDER BY bar_close_ts ASC, bar_id ASC")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return pd.DataFrame(cur.fetchall())


def _compute_features_for_new_bars(
    all_bars: pd.DataFrame, new_bar_ids: list[int],
) -> pd.DataFrame:
    """Run features.builder on all bars; return only rows for new_bar_ids."""
    from features.builder import build_features
    feats = build_features(all_bars)
    return feats[feats["bar_id"].isin(new_bar_ids)].reset_index(drop=True)


def _decide(probs: np.ndarray, confidence_threshold: float
            ) -> tuple[bool, int, str, str]:
    """Given (p_long, p_short, p_neutral), decide trade.

    Returns (should_trade, direction, argmax_class, reason).
    direction: +1 LONG, -1 SHORT, 0 SKIP.
    """
    p_long, p_short, p_neutral = float(probs[0]), float(probs[1]), float(probs[2])
    max_p = max(p_long, p_short, p_neutral)
    classes = ["LONG", "SHORT", "NEUTRAL"]
    argmax_idx = int(np.argmax([p_long, p_short, p_neutral]))
    argmax_class = classes[argmax_idx]

    if max_p < confidence_threshold:
        return False, 0, argmax_class, f"max_prob={max_p:.3f}<thr={confidence_threshold:.2f}"
    if argmax_class == "NEUTRAL":
        return False, 0, argmax_class, "argmax=NEUTRAL"
    if argmax_class == "LONG":
        return True, +1, "LONG", f"p_long={p_long:.3f}>=thr AND argmax=LONG"
    return True, -1, "SHORT", f"p_short={p_short:.3f}>=thr AND argmax=SHORT"


def _get_max_decided_bar_id(session_id: str) -> int:
    """Latest bar_id we've already made a decision on. 0 if no decisions yet."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(bar_id), 0) AS max_bid "
                "FROM events.paper_decisions WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            return int(row["max_bid"])


def _insert_decision(
    session_id: str, bar_id: int, probs: np.ndarray,
    traded: bool, argmax_class: str, skip_reason: Optional[str],
    trade_id: Optional[int],
) -> None:
    p_long, p_short, p_neutral = float(probs[0]), float(probs[1]), float(probs[2])
    max_prob = float(max(p_long, p_short, p_neutral))
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO events.paper_decisions
                  (session_id, bar_id, p_long, p_short, p_neutral, max_prob,
                   argmax_class, traded, skip_reason, trade_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id, bar_id) DO NOTHING
                """,
                (session_id, int(bar_id), p_long, p_short, p_neutral, max_prob,
                 argmax_class, traded, skip_reason, trade_id),
            )
            conn.commit()


def _has_ticks_in_month(asset: str, month_start: date) -> bool:
    """Check if events.ticks_<sym> has any rows in [month_start, month_end)."""
    tbl = ticks_table(asset)
    next_month = (date(month_start.year + 1, 1, 1) if month_start.month == 12
                  else date(month_start.year, month_start.month + 1, 1))
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {tbl} WHERE ts >= %s AND ts < %s LIMIT 1",
                (month_start, next_month),
            )
            return cur.fetchone() is not None


def _build_or_refresh_bars_current_month(asset: str, bar_threshold: float) -> dict:
    """Re-run bars/cusum.build_bars for the current month. Returns counts.

    Pre-check: if no ticks exist in the current month (e.g., Binance proxy
    not yet configured), skip the build (avoids psycopg COPY BINARY error
    on empty queries against TimescaleDB).
    """
    from bars import cusum
    today = datetime.now(timezone.utc).date().replace(day=1)
    if not _has_ticks_in_month(asset, today):
        LOG.info("no ticks in %s for current month — skipping cusum rebuild",
                 ticks_table(asset))
        return {"month": str(today), "wall_seconds": 0.0, "skipped": True,
                "reason": "no_ticks_in_current_month"}
    t0 = time.perf_counter()
    stats = cusum.build_bars(
        threshold=bar_threshold,
        max_duration_h=168.0,
        month_filter=today,
        dry_run=False,
        symbol=asset,
    )
    return {"month": str(today), "wall_seconds": time.perf_counter() - t0, **stats}


# ─────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────

_STOP = False


def _signal_handler(signum, frame):
    global _STOP
    _STOP = True
    LOG.info("received signal %d → graceful shutdown after current iteration", signum)


def loop_once(
    artifact: dict,
    mgr: PaperTradeManager,
    audit: AuditLog,
    cfg: dict,
    asset: str,
    bar_threshold: float,
    confidence_threshold: float,
    max_daily_loss_pct: float,
    halt_path: Path,
    session_started_at: datetime,
) -> Optional[str]:
    """One iteration of the loop. Returns 'halt' if we should stop, else None."""
    iter_t0 = time.perf_counter()
    audit.write("iteration_start")

    # 1. Halt checks
    if halt_path.exists():
        audit.write("halt", reason="manual_flag", path=str(halt_path))
        return "halt"
    daily_loss_frac = mgr.todays_realized_loss_pct()
    if daily_loss_frac <= -(max_daily_loss_pct / 100.0):
        audit.write("halt", reason="dd_kill", daily_loss_frac=daily_loss_frac,
                    max_pct=max_daily_loss_pct)
        return "halt"

    # 2. Poll Binance for new ticks
    try:
        res = binance_rest.poll_and_insert(
            symbol_for_db=asset, symbol_for_binance=f"{asset}USDT",
        )
        audit.write("tick_poll", **res)
    except Exception as e:
        audit.write("error", phase="tick_poll", error=str(e))
        LOG.exception("tick poll failed")
        # Continue iteration with whatever ticks we have

    # 3. Rebuild bars for current month
    try:
        bar_stats = _build_or_refresh_bars_current_month(asset, bar_threshold)
        audit.write("bars_built", **{k: v for k, v in bar_stats.items()
                                     if isinstance(v, (int, float, str))})
    except Exception as e:
        audit.write("error", phase="bars_build", error=str(e))
        LOG.exception("bars build failed")

    # 4. Find new bars since last decision (or session start floor)
    last_bid = _get_max_decided_bar_id(mgr.session_id)
    new_bars = _new_bars_since(last_bid, session_started_at, asset, bar_threshold)
    if len(new_bars) == 0:
        audit.write("no_new_bars", last_decided_bar_id=last_bid)
    else:
        audit.write("new_bars_found", n=len(new_bars),
                    first_bar_id=int(new_bars["bar_id"].iloc[0]),
                    last_bar_id=int(new_bars["bar_id"].iloc[-1]))

        # 5. Compute features for new bars (needs full bars for warmup)
        all_bars = _all_bars_for_features(asset, bar_threshold)
        new_bar_ids = new_bars["bar_id"].astype(int).tolist()
        feat_df = _compute_features_for_new_bars(all_bars, new_bar_ids)

        feature_cols = artifact["feature_columns"]
        booster = artifact["booster"]
        platt = artifact["platt"]

        # 6. Iterate new bars: predict + decide
        for _, row in feat_df.iterrows():
            bar_id = int(row["bar_id"])
            # Check if feature row has NaN (warmup not satisfied)
            X = row[feature_cols].astype("float64").values.reshape(1, -1)
            if np.isnan(X).any():
                audit.write("skip_nan_features", bar_id=bar_id)
                _insert_decision(mgr.session_id, bar_id, np.array([0., 0., 1.]),
                                 traded=False, argmax_class="NEUTRAL",
                                 skip_reason="nan_features", trade_id=None)
                continue
            raw = booster.predict(X)  # shape (1, 3)
            cal = apply_platt(raw, platt)  # (1, 3)
            probs = cal[0]  # (3,)
            should_trade, direction, argmax_class, reason = _decide(
                probs, confidence_threshold,
            )
            audit.write("l0_prediction", bar_id=bar_id,
                        p_long=float(probs[0]), p_short=float(probs[1]), p_neutral=float(probs[2]),
                        argmax=argmax_class, decision=("TRADE" if should_trade else "SKIP"),
                        reason=reason)
            trade_id: Optional[int] = None
            if should_trade:
                # Get entry price = current bar's close
                bar_row = new_bars[new_bars["bar_id"] == bar_id].iloc[0]
                tid = mgr.open_trade(
                    bar_id=bar_id,
                    entry_ts=bar_row["bar_close_ts"],
                    entry_price=float(bar_row["close"]),
                    direction=direction,
                    p_long=float(probs[0]), p_short=float(probs[1]), p_neutral=float(probs[2]),
                    entry_reason=reason,
                )
                if tid is None:
                    audit.write("trade_skip_concurrent", bar_id=bar_id, reason="max_concurrent")
                    _insert_decision(mgr.session_id, bar_id, probs, traded=False,
                                     argmax_class=argmax_class, skip_reason="concurrent_block",
                                     trade_id=None)
                    continue
                trade_id = tid
                audit.write("trade_entry", trade_id=trade_id, bar_id=bar_id,
                            direction=direction, entry_price=float(bar_row["close"]),
                            reason=reason)
                _insert_decision(mgr.session_id, bar_id, probs, traded=True,
                                 argmax_class=argmax_class, skip_reason=None,
                                 trade_id=trade_id)
            else:
                _insert_decision(mgr.session_id, bar_id, probs, traded=False,
                                 argmax_class=argmax_class, skip_reason=reason,
                                 trade_id=None)

    # 7. Check open trades for exits
    try:
        closed = mgr.check_open_trades()
        for c in closed:
            audit.write("trade_exit", **{k: v for k, v in c.items()
                                          if not isinstance(v, datetime)},
                        exit_ts=str(c.get("exit_ts")))
    except Exception as e:
        audit.write("error", phase="check_exits", error=str(e))
        LOG.exception("check exits failed")

    iter_wall = time.perf_counter() - iter_t0
    summary = mgr.session_summary()
    audit.write("iteration_end", wall_seconds=iter_wall,
                **{k: (str(v) if isinstance(v, datetime) else v)
                   for k, v in summary.items()})
    LOG.info("loop iter done in %.1fs | trades: total=%d open=%d exited=%d win%%=%.1f mPnL=%+.1f total=%+.1f",
             iter_wall, summary.get("n_total", 0), summary.get("n_open", 0),
             summary.get("n_exited", 0), summary.get("win_pct", 0.0),
             summary.get("mean_pnl_bps_net", 0.0), summary.get("total_pnl_bps", 0.0))
    return None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="run_paper_trading_loop")
    p.add_argument("--asset", default="BTC")
    p.add_argument("--bar-threshold", type=float, default=0.015)
    p.add_argument("--session-id", default=None,
                   help="Default: btc_thr015_<YYYYMMDD>")
    p.add_argument("--artifact", default=None,
                   help="Default: latest l0_<sym>_thr<NNN>_v*.pkl in data/storage/models/")
    p.add_argument("--poll-seconds", type=int, default=600, help="Poll interval (default 600 = 10min)")
    p.add_argument("--halt-file", default="/tmp/paper_trading_HALT",
                   help="Touch this file to gracefully halt")
    p.add_argument("--log-dir", default="data/storage/paper_trading",
                   help="Audit log directory (relative to project root)")
    p.add_argument("--notes", default="", help="Free-text session notes")
    p.add_argument("--one-shot", action="store_true", help="Run one iteration then exit (smoke)")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = _load_config()
    from data.db import symbol_short
    sym = symbol_short(args.asset).upper()
    bar_threshold = args.bar_threshold
    confidence_threshold = cfg["model"]["signal_threshold"]
    tp_pct = cfg["labeling"]["tp_pct"].get(sym, cfg["labeling"]["tp_pct"]["BTC"])
    sl_pct = cfg["labeling"]["sl_pct"].get(sym, cfg["labeling"]["sl_pct"]["BTC"])
    vertical_bars = cfg["labeling"]["vertical_bars"]
    position_size_usd = cfg["backtest"]["position_size_usd"]
    cost_bps_round_trip = cfg["costs"]["scenarios"]["realistic"]  # 7 bps RT (Hyperliquid)
    max_daily_loss_pct = cfg["risk"]["max_daily_loss_pct"]

    # Artifact
    artifact_path = Path(args.artifact) if args.artifact else _default_artifact_path(sym, bar_threshold)
    LOG.info("loading artifact: %s", artifact_path)
    artifact = _load_artifact(artifact_path)
    meta = artifact["training_metadata"]
    LOG.info("  trained_at=%s  data_max=%s  ratio=%.4f  n_trees=%d",
             meta["trained_at"], meta["data_max_ts"], meta["ratio"], meta["n_trees"])

    # Session
    init_paper_schema()
    if args.session_id:
        session_id = args.session_id
    else:
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        session_id = f"{sym.lower()}_thr{int(bar_threshold*1000):03d}_{today_str}"
    LOG.info("session_id: %s", session_id)
    start_session(
        session_id=session_id, asset=sym, model_artifact_path=str(artifact_path),
        bar_threshold=bar_threshold, tp_pct=tp_pct, sl_pct=sl_pct,
        vertical_bars=vertical_bars, confidence_threshold=confidence_threshold,
        position_size_usd=position_size_usd, cost_bps_round_trip=cost_bps_round_trip,
        max_daily_loss_pct=max_daily_loss_pct, notes=args.notes,
    )

    # Manager + audit log
    mgr = PaperTradeManager(
        session_id=session_id, asset=sym, bar_threshold=bar_threshold,
        tp_pct=tp_pct, sl_pct=sl_pct, vertical_bars=vertical_bars,
        cost_bps_round_trip=cost_bps_round_trip,
        position_size_usd=position_size_usd, max_concurrent=1,
    )
    log_dir = PROJECT_ROOT / args.log_dir
    audit = AuditLog(log_dir, session_id)
    audit.write("session_start", session_id=session_id, asset=sym,
                bar_threshold=bar_threshold, tp_pct=tp_pct, sl_pct=sl_pct,
                vertical_bars=vertical_bars, confidence_threshold=confidence_threshold,
                cost_bps_round_trip=cost_bps_round_trip,
                max_daily_loss_pct=max_daily_loss_pct,
                artifact_path=str(artifact_path),
                model_metadata=meta)

    halt_path = Path(args.halt_file)
    end_reason = "normal"
    session_started_at = datetime.now(timezone.utc)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while not _STOP:
            result = loop_once(
                artifact=artifact, mgr=mgr, audit=audit, cfg=cfg, asset=sym,
                bar_threshold=bar_threshold,
                confidence_threshold=confidence_threshold,
                max_daily_loss_pct=max_daily_loss_pct,
                halt_path=halt_path,
                session_started_at=session_started_at,
            )
            if result == "halt":
                end_reason = "halt"
                break
            if args.one_shot:
                end_reason = "one_shot_complete"
                break
            time.sleep(args.poll_seconds)
    except Exception:
        LOG.exception("fatal in main loop")
        audit.write("error", phase="main_loop", error="see logs")
        end_reason = "crashed"
        raise
    finally:
        end_session(session_id, end_reason=end_reason)
        audit.write("session_end", end_reason=end_reason,
                    **{k: (str(v) if isinstance(v, datetime) else v)
                       for k, v in mgr.session_summary().items()})
        audit.close()
        close_pool()
        LOG.info("session %s ended (reason=%s)", session_id, end_reason)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
