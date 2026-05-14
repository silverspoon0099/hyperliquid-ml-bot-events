"""Extract per-fold L0 OOT predictions for meta-labeling (DR v3.0.19).

Re-runs the walk-forward training at TB=0.03 (matching DR v3.0.12 best
operating point) and SAVES per-fold OOT predictions to parquet for
use by `scripts/run_meta_labeling.py`. Training is the same as joint
sweep — we just persist the probabilities + the realized trade outcomes.

For each fold (skipping early-stopped folds):
  - Train LightGBM on (train, val) at TB=0.03 relabel
  - Predict OOT probabilities (3-class)
  - For each primary threshold {0.62, 0.65}, simulate trades and record:
      bar_id, bar_close_ts, fold_id, primary_thr,
      p_long, p_short, p_neutral, direction (LONG=1 / SHORT=-1),
      pnl_bps_net, win (1 if pnl > 0 else 0), true_label (0/1/2)

Output: `reports/phase_1/l0_predictions_thr{0.62,0.65}.parquet`

CLI:
    python -m scripts.extract_l0_predictions --tb 0.03 --thresholds 0.62,0.65
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cv.walk_forward import generate_folds, split_fold
from model.lgbm import train_lgbm, fit_platt, apply_platt
from backtest.runner import simulate_trades
from data.db import close_pool, bars_table, symbol_short, get_engine

LOG = logging.getLogger("scripts.extract_l0_predictions")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

KEY_COLS = ["bar_id", "bar_close_ts"]
LABEL_COL = "label"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_features(asset: str = "BTC") -> pd.DataFrame:
    sym = symbol_short(asset)
    return pd.read_parquet(PROJECT_ROOT / f"data/storage/features/features_{sym}.parquet")


def _load_bars_ohlc(asset: str = "BTC") -> pd.DataFrame:
    sql = (f"SELECT bar_id, bar_open_ts, bar_close_ts, open, high, low, close "
           f"FROM {bars_table(asset)} WHERE threshold_pct=0.02 ORDER BY bar_close_ts, bar_id")
    return pd.read_sql_query(sql, get_engine())


def run_extract(
    tb: float = 0.03,
    primary_thresholds: tuple = (0.62, 0.65),
    cost_bps_round_trip: float = 11.0,
    asset: str = "BTC",
) -> dict:
    """Train L0 per fold at TB=tb, persist OOT predictions + trades per primary threshold."""
    from labels.triple_barrier import apply_triple_barrier

    cfg = _load_config()
    wf = cfg["walk_forward"]
    lgbm_params = cfg["model"]["L0_lightgbm"]
    vertical_bars = cfg["labeling"]["vertical_bars"]

    LOG.info("loading features + bars (asset=%s, with OHLC)...", asset)
    feats = _load_features(asset=asset)
    bars_full = _load_bars_ohlc(asset=asset)
    feature_cols = [c for c in feats.columns if c not in KEY_COLS]

    LOG.info("relabeling at TB=tp=sl=%.3f...", tb)
    labels_df = apply_triple_barrier(
        bars_full[["bar_id", "bar_open_ts", "bar_close_ts", "close", "high", "low"]],
        tp_pct=tb, sl_pct=tb, vertical_bars=vertical_bars,
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

    # Per primary threshold: accumulate rows from all folds
    rows_by_thr: dict[float, list[dict]] = {thr: [] for thr in primary_thresholds}

    t0_total = time.perf_counter()
    for fold in folds:
        parts = split_fold(df_full, fold, purge_bars=wf["purge_bars"],
                            embargo_bars=wf["embargo_bars"], ts_col="bar_close_ts")
        train, val, oot = parts["train"], parts["val"], parts["oot"]
        if len(val) < 100 or len(oot) < 100:
            LOG.warning("Fold %d: SKIPPED (n_val=%d, n_oot=%d)", fold.fold_id, len(val), len(oot))
            continue

        booster = train_lgbm(
            train[feature_cols], train[LABEL_COL].astype(int).values,
            val[feature_cols], val[LABEL_COL].astype(int).values,
            dict(lgbm_params),
        )
        val_raw = booster.predict(val[feature_cols])
        cal = fit_platt(val_raw, val[LABEL_COL].astype(int).values, n_classes=3)
        oot_raw = booster.predict(oot[feature_cols])
        oot_cal = apply_platt(oot_raw, cal)

        # OOT bar metadata
        oot_bar_ids = oot["bar_id"].astype("int64").values
        oot_ts = pd.to_datetime(oot["bar_close_ts"].values, utc=True)
        oot_true_labels = oot[LABEL_COL].astype("int64").values

        preds = pd.DataFrame({
            "bar_id": oot_bar_ids,
            "p_long":   oot_cal[:, 0],
            "p_short":  oot_cal[:, 1],
            "p_neutral": oot_cal[:, 2],
        })
        labels_oot = oot[["bar_id", "exit_bar_id", "exit_price",
                          "exit_reason", "holding_bars", "label"]].copy()

        # For each primary threshold, simulate trades and record per-bar outcome
        for thr in primary_thresholds:
            trades = simulate_trades(
                predictions=preds,
                bars_df=bars_full[["bar_id", "bar_close_ts", "close"]],
                labels_df=labels_oot,
                confidence_threshold=thr,
                cost_bps_round_trip=cost_bps_round_trip,
                max_concurrent=1,
            )
            # Index trades by bar_id (entry bar) for fast lookup
            trade_by_bar = {t.bar_id: t for t in trades}
            # For each OOT bar where primary fires, record the row
            # Determine which bars the primary "signaled" at this threshold
            for i, bid in enumerate(oot_bar_ids):
                p_long = float(oot_cal[i, 0])
                p_short = float(oot_cal[i, 1])
                p_neutral = float(oot_cal[i, 2])
                max_p = max(p_long, p_short, p_neutral)
                # Primary signals only if max > thr AND the max class is not NEUTRAL
                # AND the trade actually got simulated (not blocked by max_concurrent)
                if bid not in trade_by_bar:
                    continue  # didn't fire / blocked by concurrency
                t = trade_by_bar[bid]
                rows_by_thr[thr].append({
                    "bar_id": int(bid),
                    "bar_close_ts": oot_ts[i],
                    "fold_id": int(fold.fold_id),
                    "primary_thr": float(thr),
                    "p_long": p_long,
                    "p_short": p_short,
                    "p_neutral": p_neutral,
                    "direction": int(t.direction),     # +1 LONG, -1 SHORT
                    "entry_ts": t.entry_ts,
                    "exit_ts": t.exit_ts,
                    "entry_price": float(t.entry_price),
                    "exit_price": float(t.exit_price),
                    "exit_reason": str(t.exit_reason),
                    "holding_bars": int(t.holding_bars),
                    "pnl_bps_gross": float(t.pnl_bps_gross),
                    "pnl_bps_net": float(t.pnl_bps_net),
                    "win": int(t.pnl_bps_net > 0),
                    "true_label": int(oot_true_labels[i]),
                })

        LOG.info("Fold %d done (per-thr trades: %s)",
                 fold.fold_id,
                 ", ".join(f"{thr:.2f}={sum(1 for r in rows_by_thr[thr] if r['fold_id']==fold.fold_id)}"
                           for thr in primary_thresholds))

    total_s = time.perf_counter() - t0_total
    LOG.info("extract done in %.1fs (%.1f min)", total_s, total_s / 60.0)

    out_paths = {}
    for thr in primary_thresholds:
        df = pd.DataFrame(rows_by_thr[thr])
        thr_label = f"{int(thr * 100):03d}"  # 0.62 -> "062"
        out_path = PROJECT_ROOT / "reports" / "phase_1" / f"l0_predictions_thr{thr_label}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, compression="snappy")
        LOG.info("  thr=%.2f: %d trade signals → %s (%.2f MB)",
                 thr, len(df), out_path, out_path.stat().st_size / 1e6)
        out_paths[thr] = str(out_path)

    return {
        "tb": tb,
        "primary_thresholds": list(primary_thresholds),
        "n_folds": len(folds),
        "wall_clock_seconds": total_s,
        "output_paths": out_paths,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="extract_l0_predictions")
    p.add_argument("--asset", default="BTC")
    p.add_argument("--tb", type=float, default=0.03)
    p.add_argument("--thresholds", default="0.62,0.65",
                   help="Comma-separated primary thresholds (default: 0.62,0.65)")
    p.add_argument("--cost-bps", type=float, default=11.0,
                   help="Round-trip cost in bps for trade simulation")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    thresholds = tuple(float(x) for x in args.thresholds.split(","))
    try:
        run_extract(tb=args.tb, primary_thresholds=thresholds,
                     cost_bps_round_trip=args.cost_bps, asset=args.asset)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
