"""Backfill backtest trades into events.backtest_trades (DR v3.0.25).

Re-runs the L0 walk-forward at a given config (defaulting to the locked
champion: 1.5% bars + TB=0.03 + thr=0.58) and persists every simulated
trade into events.backtest_trades so the dashboard can show them
alongside paper trades.

The walk-forward training is identical to scripts.run_phase_1_lgbm's
joint sweep, but instead of just aggregating metrics we persist each
Trade row.

CLI:
    python -m scripts.backfill_backtest_trades \\
        --asset BTC --bar-threshold 0.015 --tb 0.03 --threshold 0.58 \\
        --run-id champion_btc_thr015_tb03_t058 \\
        --notes "Champion baseline from DR v3.0.20"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cv.walk_forward import generate_folds, split_fold
from model.lgbm import train_lgbm, fit_platt, apply_platt
from backtest.runner import simulate_trades, build_equity_curve, compute_metrics
from data.db import (
    close_pool, bars_table, symbol_short, get_engine, get_connection,
    init_backtest_trades_schema,
)

LOG = logging.getLogger("scripts.backfill_backtest_trades")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

KEY_COLS = ["bar_id", "bar_close_ts"]
LABEL_COL = "label"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_features(features_path: str) -> pd.DataFrame:
    return pd.read_parquet(features_path)


def _load_bars_ohlc(asset: str, bar_threshold: float) -> pd.DataFrame:
    sql = (f"SELECT bar_id, bar_open_ts, bar_close_ts, open, high, low, close "
           f"FROM {bars_table(asset)} WHERE threshold_pct = {bar_threshold} "
           f"ORDER BY bar_close_ts, bar_id")
    return pd.read_sql_query(sql, get_engine())


def upsert_run(
    run_id: str, asset: str, bar_threshold: float, tb_pct: float,
    confidence_threshold: float, vertical_bars: int,
    cost_bps_round_trip: float, feature_set: str,
    notes: str, git_commit: Optional[str],
) -> None:
    """Insert/replace the run metadata row."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Delete prior data for clean re-runs of same run_id
            cur.execute("DELETE FROM events.backtest_trades WHERE run_id = %s", (run_id,))
            cur.execute("DELETE FROM events.backtest_runs WHERE run_id = %s", (run_id,))
            cur.execute(
                """
                INSERT INTO events.backtest_runs
                  (run_id, asset, bar_threshold, tb_pct, confidence_threshold,
                   vertical_bars, cost_bps_round_trip, feature_set, git_commit, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, asset, float(bar_threshold), float(tb_pct),
                 float(confidence_threshold), int(vertical_bars),
                 float(cost_bps_round_trip), feature_set, git_commit, notes),
            )
            conn.commit()


def update_run_summary(run_id: str, summary: dict) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE events.backtest_runs SET
                  n_folds_total = %s, n_folds_evaluated = %s,
                  sharpe_all_folds = %s, sharpe_nonzero = %s,
                  n_trades_total = %s, win_pct_mean = %s
                WHERE run_id = %s
                """,
                (
                    summary.get("n_folds_total"),
                    summary.get("n_folds_evaluated"),
                    summary.get("sharpe_all_folds"),
                    summary.get("sharpe_nonzero"),
                    summary.get("n_trades_total"),
                    summary.get("win_pct_mean"),
                    run_id,
                ),
            )
            conn.commit()


def insert_trades(run_id: str, rows: list[dict]) -> None:
    """Bulk insert backtest trades."""
    if not rows:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO events.backtest_trades
                  (run_id, fold_id, bar_id_entry, entry_ts, entry_price,
                   direction, p_long, p_short, p_neutral,
                   exit_bar_id, exit_ts, exit_price, exit_reason, holding_bars,
                   pnl_bps_gross, pnl_bps_net, label)
                VALUES (%(run_id)s, %(fold_id)s, %(bar_id_entry)s, %(entry_ts)s, %(entry_price)s,
                        %(direction)s, %(p_long)s, %(p_short)s, %(p_neutral)s,
                        %(exit_bar_id)s, %(exit_ts)s, %(exit_price)s, %(exit_reason)s, %(holding_bars)s,
                        %(pnl_bps_gross)s, %(pnl_bps_net)s, %(label)s)
                """,
                rows,
            )
            conn.commit()


def run_backfill(
    run_id: str,
    asset: str = "BTC",
    bar_threshold: float = 0.015,
    tb_pct: float = 0.03,
    confidence_threshold: float = 0.58,
    cost_bps_round_trip: float = 11.0,
    feature_set: str = "v1_33",
    notes: str = "",
    features_path: Optional[str] = None,
) -> dict:
    from labels.triple_barrier import apply_triple_barrier

    cfg = _load_config()
    wf = cfg["walk_forward"]
    lgbm_params = cfg["model"]["L0_lightgbm"]
    vertical_bars = cfg["labeling"]["vertical_bars"]

    sym = symbol_short(asset)
    if features_path is None:
        thr_suffix = f"_thr{int(bar_threshold*1000):03d}"
        features_path = str(
            PROJECT_ROOT / f"data/storage/features/features_{sym}{thr_suffix}.parquet"
        )

    LOG.info("loading features (%s)...", features_path)
    feats = _load_features(features_path)
    LOG.info("loading bars (asset=%s, bar_thr=%s)...", asset, bar_threshold)
    bars_full = _load_bars_ohlc(asset, bar_threshold)
    feature_cols = [c for c in feats.columns if c not in KEY_COLS]

    LOG.info("relabeling at TB=tp=sl=%.3f...", tb_pct)
    labels_df = apply_triple_barrier(
        bars_full[["bar_id", "bar_open_ts", "bar_close_ts", "close", "high", "low"]],
        tp_pct=tb_pct, sl_pct=tb_pct, vertical_bars=vertical_bars,
    )
    df_full = feats.merge(labels_df, on="bar_id", how="inner")
    df_full = df_full[df_full[LABEL_COL] != -1].copy()
    df_full["label"] = df_full["label"].astype("int64")
    df_full = df_full.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)
    LOG.info("merged: %d labelable bars", len(df_full))

    data_start = date(df_full["bar_close_ts"].min().year,
                      df_full["bar_close_ts"].min().month, 1)
    data_end_ts = df_full["bar_close_ts"].max()
    if data_end_ts.month == 12:
        data_end = date(data_end_ts.year + 1, 1, 1)
    else:
        data_end = date(data_end_ts.year, data_end_ts.month + 1, 1)

    folds = generate_folds(
        data_start=data_start, data_end=data_end,
        initial_train_months=wf["initial_train_months"],
        val_months=wf["val_months"], oot_months=wf["oot_months"],
        step_months=wf["step_months"],
    )
    LOG.info("generated %d folds", len(folds))

    # Initialize the run row (idempotent: delete prior then insert)
    git_commit = None
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            timeout=2,
        ).decode().strip()
    except Exception:
        pass
    init_backtest_trades_schema()
    upsert_run(run_id, asset, bar_threshold, tb_pct, confidence_threshold,
               vertical_bars, cost_bps_round_trip, feature_set, notes, git_commit)
    LOG.info("registered run_id=%s, git=%s", run_id, (git_commit or "?")[:8])

    n_evaluated = 0
    fold_sharpes = []
    all_trade_rows = []
    n_winners_total = 0
    n_total = 0

    t0_total = time.perf_counter()
    for fold in folds:
        parts = split_fold(df_full, fold, purge_bars=wf["purge_bars"],
                            embargo_bars=wf["embargo_bars"], ts_col="bar_close_ts")
        train, val, oot = parts["train"], parts["val"], parts["oot"]
        if len(val) < 100 or len(oot) < 100:
            LOG.warning("Fold %d: SKIPPED (n_val=%d, n_oot=%d)",
                        fold.fold_id, len(val), len(oot))
            continue
        n_evaluated += 1

        booster = train_lgbm(
            train[feature_cols], train[LABEL_COL].astype(int).values,
            val[feature_cols], val[LABEL_COL].astype(int).values,
            dict(lgbm_params),
        )
        val_raw = booster.predict(val[feature_cols])
        cal = fit_platt(val_raw, val[LABEL_COL].astype(int).values, n_classes=3)
        oot_raw = booster.predict(oot[feature_cols])
        oot_cal = apply_platt(oot_raw, cal)

        preds = pd.DataFrame({
            "bar_id": oot["bar_id"].astype("int64").values,
            "p_long":   oot_cal[:, 0],
            "p_short":  oot_cal[:, 1],
            "p_neutral": oot_cal[:, 2],
        })
        labels_oot = oot[["bar_id", "exit_bar_id", "exit_price",
                          "exit_reason", "holding_bars", "label"]].copy()

        trades = simulate_trades(
            predictions=preds,
            bars_df=bars_full[["bar_id", "bar_close_ts", "close"]],
            labels_df=labels_oot,
            confidence_threshold=confidence_threshold,
            cost_bps_round_trip=cost_bps_round_trip,
            max_concurrent=1,
        )
        eq = build_equity_curve(trades, starting_equity=10_000.0)
        metrics = compute_metrics(trades, eq, oot_n_bars=len(oot))
        fold_sharpes.append(float(metrics["oot_sharpe"]))

        # Build pred + label lookup for richer trade rows
        # (bar_id → p_long/p_short/p_neutral and bar_id → label)
        pred_by_bar = {int(b): (float(pl), float(ps), float(pn))
                       for b, pl, ps, pn in zip(
                           preds["bar_id"], preds["p_long"], preds["p_short"], preds["p_neutral"]
                       )}
        label_by_bar = {int(b): int(lb) for b, lb in zip(
            oot["bar_id"].astype("int64").values, oot["label"].astype("int64").values
        )}

        for t in trades:
            n_total += 1
            if t.pnl_bps_net > 0:
                n_winners_total += 1
            pl, ps, pn = pred_by_bar.get(int(t.bar_id), (None, None, None))
            all_trade_rows.append({
                "run_id": run_id,
                "fold_id": int(fold.fold_id),
                "bar_id_entry": int(t.bar_id),
                "entry_ts": t.entry_ts,
                "entry_price": float(t.entry_price),
                "direction": int(t.direction),
                "p_long": pl, "p_short": ps, "p_neutral": pn,
                "exit_bar_id": None,  # backtest Trade dataclass doesn't expose exit_bar_id; left null
                "exit_ts": t.exit_ts,
                "exit_price": float(t.exit_price),
                "exit_reason": str(t.exit_reason),
                "holding_bars": int(t.holding_bars),
                "pnl_bps_gross": float(t.pnl_bps_gross),
                "pnl_bps_net": float(t.pnl_bps_net),
                "label": label_by_bar.get(int(t.bar_id)),
            })

        LOG.info("Fold %d: %d trades, fold_sharpe=%.3f", fold.fold_id, len(trades), metrics["oot_sharpe"])

    total_s = time.perf_counter() - t0_total

    # Bulk insert all trades in one go (small enough — ~600 rows for champion)
    LOG.info("inserting %d trades into events.backtest_trades...", len(all_trade_rows))
    insert_trades(run_id, all_trade_rows)

    # Update run summary
    sharpe_all = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
    nz = [s for s in fold_sharpes if abs(s) > 1e-9]
    sharpe_nz = float(np.mean(nz)) if nz else 0.0
    win_pct = 100.0 * n_winners_total / max(n_total, 1)
    update_run_summary(run_id, {
        "n_folds_total": len(folds),
        "n_folds_evaluated": n_evaluated,
        "sharpe_all_folds": sharpe_all,
        "sharpe_nonzero": sharpe_nz,
        "n_trades_total": n_total,
        "win_pct_mean": win_pct,
    })

    summary = {
        "run_id": run_id,
        "n_folds_total": len(folds), "n_folds_evaluated": n_evaluated,
        "sharpe_all_folds": sharpe_all, "sharpe_nonzero": sharpe_nz,
        "n_trades_total": n_total, "win_pct": win_pct,
        "wall_seconds": total_s,
    }
    LOG.info("DONE in %.1fs: %s", total_s, summary)
    return summary


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="backfill_backtest_trades")
    p.add_argument("--asset", default="BTC")
    p.add_argument("--bar-threshold", type=float, default=0.015,
                   help="Locked champion: 1.5% bars")
    p.add_argument("--tb", type=float, default=0.03,
                   help="Locked champion: TB 0.03")
    p.add_argument("--threshold", type=float, default=0.58,
                   help="Locked champion: confidence 0.58")
    p.add_argument("--cost-bps", type=float, default=11.0,
                   help="Round-trip bps cost (default 11.0)")
    p.add_argument("--feature-set", default="v1_33",
                   help="Identifier for the feature set used")
    p.add_argument("--run-id", default=None,
                   help="Defaults to a deterministic name based on config")
    p.add_argument("--notes", default="")
    p.add_argument("--features-path", default=None,
                   help="Override features parquet path")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sym = symbol_short(args.asset)
    if args.run_id is None:
        args.run_id = (
            f"champion_{sym}_thr{int(args.bar_threshold*1000):03d}"
            f"_tb{int(args.tb*1000):03d}_t{int(args.threshold*100):02d}"
        )

    try:
        run_backfill(
            run_id=args.run_id,
            asset=sym.upper(),
            bar_threshold=args.bar_threshold,
            tb_pct=args.tb,
            confidence_threshold=args.threshold,
            cost_bps_round_trip=args.cost_bps,
            feature_set=args.feature_set,
            notes=args.notes,
            features_path=args.features_path,
        )
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
