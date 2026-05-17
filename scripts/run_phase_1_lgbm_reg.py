"""Phase 1.0 L0 LightGBM REGRESSION walk-forward (DR v3.0.21 Phase A).

Tests the "continuous targets" hypothesis: instead of 3-class softmax
on triple-barrier labels, train regression on the realized signed log-
return at TB exit. Sign of prediction → trade direction. |prediction|
→ confidence (sweep magnitude thresholds).

Substrate: 1.5% bars + 33-feature parquet (DR v3.0.20 champion baseline,
locked as tag `v3.0.20-champion-baseline`).

Target: y = log(exit_price / entry_close)
  - LONG triple-barrier hit: y ≈ +log(1.05) ≈ +0.0488
  - SHORT triple-barrier hit: y ≈ −log(1.05) ≈ −0.0488
  - vertical/timeout exit: y = log(actual_exit / entry_close), continuous

Loss: Huber (robust to crypto's heavy tails; the ±5% TB spikes are
naturally outlier-ish).

Trade construction (re-uses existing simulate_trades):
  - if pred_ret > +mag_thr: LONG signal (pseudo p_long=1)
  - if pred_ret < −mag_thr: SHORT signal (pseudo p_short=1)
  - else: skip (p_neutral=1)
  Then simulate_trades with confidence_threshold=0.5 fires the LONG/SHORT
  trade and the existing pipeline computes Sharpe/win%/etc.

Magnitude threshold sweep: {0.005, 0.010, 0.015, 0.020, 0.025, 0.030}
log-return units (≈ 0.5–3.0% predicted move).

Decision tree vs v3.0.20 champion (+1.204 at thr=0.58):
  - Lift ≥ +0.20  → adopt as new champion, proceed to Phase B
  - Lift ∈ [0, +0.20] → marginal, ship categorical
  - Lift < 0  → regression unhelpful, ship categorical

CLI:
    python -m scripts.run_phase_1_lgbm_reg --asset BTC
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

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cv.walk_forward import generate_folds, split_fold
from backtest.runner import simulate_trades, build_equity_curve, compute_metrics
from data.db import close_pool, bars_table, symbol_short, get_engine

LOG = logging.getLogger("scripts.run_phase_1_lgbm_reg")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

KEY_COLS = ["bar_id", "bar_close_ts"]


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_features(features_path: str) -> pd.DataFrame:
    return pd.read_parquet(features_path)


def _load_labels(labels_path: str) -> pd.DataFrame:
    return pd.read_parquet(labels_path)


def _load_bars_ohlc(asset: str, bar_threshold: float) -> pd.DataFrame:
    sql = (f"SELECT bar_id, bar_open_ts, bar_close_ts, open, high, low, close "
           f"FROM {bars_table(asset)} WHERE threshold_pct = {bar_threshold} "
           f"ORDER BY bar_close_ts, bar_id")
    return pd.read_sql_query(sql, get_engine())


def build_regression_target(labels: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """For each labeled bar, compute target = log(exit_price / entry_close)."""
    merged = labels.merge(
        bars[["bar_id", "close"]].rename(columns={"close": "entry_close"}),
        on="bar_id", how="inner",
    )
    merged["y"] = np.log(merged["exit_price"].astype(float) / merged["entry_close"].astype(float))
    return merged[["bar_id", "y"]]


def train_lgbm_regression(
    X_train, y_train, X_val, y_val, lgbm_params: dict,
) -> lgb.Booster:
    """Train LightGBM regression (Huber loss). Mirrors model.lgbm.train_lgbm
    structure but with regression objective."""
    params = {
        "objective": "huber",
        "metric": "huber",
        "alpha": 0.9,           # Huber transition point (default-ish)
        "boosting_type": lgbm_params.get("boosting_type", "gbdt"),
        "num_leaves": lgbm_params.get("num_leaves", 63),
        "learning_rate": lgbm_params.get("learning_rate", 0.05),
        "feature_fraction": lgbm_params.get("feature_fraction", 0.8),
        "bagging_fraction": lgbm_params.get("bagging_fraction", 0.8),
        "bagging_freq": lgbm_params.get("bagging_freq", 5),
        "min_child_samples": lgbm_params.get("min_child_samples", 50),
        "lambda_l1": lgbm_params.get("lambda_l1", 0.1),
        "lambda_l2": lgbm_params.get("lambda_l2", 0.1),
        "verbose": -1,
        "seed": lgbm_params.get("seed", 42),
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=lgbm_params.get("num_boost_round", 1000),
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=lgbm_params.get("early_stopping_rounds", 50), verbose=False)],
    )
    return booster


def regression_to_pseudo_probs(
    bar_ids: np.ndarray, pred_ret: np.ndarray, mag_thr: float,
) -> pd.DataFrame:
    """Convert regression predictions to (p_long, p_short, p_neutral) for simulate_trades.
    Trade if |pred| > mag_thr, direction = sign(pred). Pseudo-prob = 1 for active class."""
    p_long = (pred_ret > mag_thr).astype(float)
    p_short = (pred_ret < -mag_thr).astype(float)
    p_neutral = 1.0 - p_long - p_short
    return pd.DataFrame({
        "bar_id": bar_ids.astype("int64"),
        "p_long": p_long,
        "p_short": p_short,
        "p_neutral": p_neutral,
    })


def run_regression_sweep(
    features_path: str,
    labels_path: str,
    asset: str = "BTC",
    bar_threshold: float = 0.015,
    magnitude_thresholds: tuple = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030),
    cost_bps_round_trip: float = 11.0,
    out_suffix: str = "_thr015_reg",
) -> dict:
    cfg = _load_config()
    wf = cfg["walk_forward"]
    lgbm_params = cfg["model"]["L0_lightgbm"]

    LOG.info("loading features (%s)...", features_path)
    feats = _load_features(features_path)
    LOG.info("loading labels (%s)...", labels_path)
    labels = _load_labels(labels_path)
    LOG.info("loading bars (asset=%s, bar_thr=%s)...", asset, bar_threshold)
    bars_full = _load_bars_ohlc(asset, bar_threshold)
    feature_cols = [c for c in feats.columns if c not in KEY_COLS]
    LOG.info("loaded: %d features, %d bars, %d labeled", len(feats), len(bars_full), len(labels))

    LOG.info("building regression target y = log(exit_price / entry_close)...")
    y_df = build_regression_target(labels, bars_full)
    LOG.info("target: %d rows, y_min=%.4f y_max=%.4f y_mean=%.5f y_std=%.5f",
             len(y_df), y_df["y"].min(), y_df["y"].max(),
             y_df["y"].mean(), y_df["y"].std())

    # Merge feats + labels + target. Note: labels.parquet was filtered to
    # label != -1 already (during labels.triple_barrier build), so this join
    # only keeps valid labeled bars.
    df_full = feats.merge(labels[["bar_id"]], on="bar_id", how="inner")
    df_full = df_full.merge(y_df, on="bar_id", how="inner")
    df_full = df_full.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)
    LOG.info("merged: %d rows × %d cols (incl. target y)", len(df_full), df_full.shape[1])

    # Generate folds
    data_start = date(df_full["bar_close_ts"].min().year, df_full["bar_close_ts"].min().month, 1)
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
    LOG.info("generated %d folds, magnitude thresholds=%s", len(folds), list(magnitude_thresholds))

    # by_mag_thr[thr_str] -> list of per-fold rows
    by_mag: dict[str, list] = {f"{m:.4f}": [] for m in magnitude_thresholds}

    t0_total = time.perf_counter()
    n_evaluated = 0

    for fold in folds:
        parts = split_fold(df_full, fold, purge_bars=wf["purge_bars"],
                            embargo_bars=wf["embargo_bars"], ts_col="bar_close_ts")
        train, val, oot = parts["train"], parts["val"], parts["oot"]
        if len(val) < 100 or len(oot) < 100:
            LOG.warning("Fold %d: SKIPPED (n_val=%d, n_oot=%d)", fold.fold_id, len(val), len(oot))
            continue
        n_evaluated += 1

        booster = train_lgbm_regression(
            train[feature_cols], train["y"].values,
            val[feature_cols], val["y"].values,
            dict(lgbm_params),
        )
        oot_pred = booster.predict(oot[feature_cols])
        oot_bar_ids = oot["bar_id"].astype("int64").values

        labels_oot = labels[labels["bar_id"].isin(oot_bar_ids)][
            ["bar_id", "exit_bar_id", "exit_price", "exit_reason", "holding_bars", "label"]
        ].copy()

        for mag_thr in magnitude_thresholds:
            preds = regression_to_pseudo_probs(oot_bar_ids, oot_pred, mag_thr)
            trades = simulate_trades(
                predictions=preds,
                bars_df=bars_full[["bar_id", "bar_close_ts", "close"]],
                labels_df=labels_oot,
                confidence_threshold=0.5,   # always 0.5 since pseudo-probs are 0 or 1
                cost_bps_round_trip=cost_bps_round_trip,
                max_concurrent=1,
            )
            eq = build_equity_curve(trades, starting_equity=10_000.0)
            metrics = compute_metrics(trades, eq, oot_n_bars=len(oot))
            mean_pnl = float(np.mean([t.pnl_bps_net for t in trades])) if trades else 0.0
            median_pnl = float(np.median([t.pnl_bps_net for t in trades])) if trades else 0.0
            n_long = sum(1 for t in trades if t.direction == 1)
            n_short = sum(1 for t in trades if t.direction == -1)
            n_long_win = sum(1 for t in trades if t.direction == 1 and t.pnl_bps_net > 0)
            n_short_win = sum(1 for t in trades if t.direction == -1 and t.pnl_bps_net > 0)

            by_mag[f"{mag_thr:.4f}"].append({
                "fold": fold.fold_id,
                "n_oot": len(oot),
                "n_trades": metrics["oot_n_trades"],
                "n_long": n_long, "n_short": n_short,
                "mean_pnl_bps_net": mean_pnl,
                "median_pnl_bps_net": median_pnl,
                "win_pct": metrics["oot_profitable_trade_pct"],
                "long_win_pct": (n_long_win / n_long * 100.0) if n_long > 0 else 0.0,
                "short_win_pct": (n_short_win / n_short * 100.0) if n_short > 0 else 0.0,
                "sharpe": metrics["oot_sharpe"],
                "max_dd": metrics["oot_max_dd"],
                "annual_return": metrics["oot_annual_return"],
                "pct_time_in_market": metrics["oot_pct_time_in_market"],
            })
        LOG.info("Fold %d: trained; backtested %d mag thresholds (pred_min=%.4f max=%.4f)",
                 fold.fold_id, len(magnitude_thresholds), oot_pred.min(), oot_pred.max())

    total_s = time.perf_counter() - t0_total

    results = {}
    for mag_str, fold_rows in by_mag.items():
        if not fold_rows:
            results[mag_str] = {"aggregate": {}, "per_fold": []}
            continue
        sharpes = [r["sharpe"] for r in fold_rows]
        nz = [s for s in sharpes if s != 0.0]
        any_trade = [r for r in fold_rows if r["n_trades"] > 0]
        any_long = [r for r in fold_rows if r["n_long"] > 0]
        any_short = [r for r in fold_rows if r["n_short"] > 0]
        n_total = sum(r["n_trades"] for r in fold_rows)
        results[mag_str] = {
            "aggregate": {
                "n_trades_total": n_total,
                "trades_per_fold_mean": n_total / len(fold_rows),
                "n_folds_with_trades": len(any_trade),
                "n_folds_zero_trades": sum(1 for r in fold_rows if r["n_trades"] == 0),
                "mean_pnl_bps_net": float(np.mean([r["mean_pnl_bps_net"] for r in any_trade])) if any_trade else 0.0,
                "median_pnl_bps_net": float(np.median([r["median_pnl_bps_net"] for r in any_trade])) if any_trade else 0.0,
                "win_pct_mean": float(np.mean([r["win_pct"] for r in any_trade])) if any_trade else 0.0,
                "long_win_pct_mean": float(np.mean([r["long_win_pct"] for r in any_long])) if any_long else 0.0,
                "short_win_pct_mean": float(np.mean([r["short_win_pct"] for r in any_short])) if any_short else 0.0,
                "sharpe_mean_across_folds": float(np.mean(sharpes)),
                "sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
                "sharpe_mean_nonzero_folds": float(np.mean(nz)) if nz else 0.0,
                "annual_return_mean": float(np.mean([r["annual_return"] for r in fold_rows])),
            },
            "per_fold": fold_rows,
        }

    out = {
        "objective": "huber_regression",
        "target": "log(exit_price / entry_close)",
        "asset": asset,
        "bar_threshold": bar_threshold,
        "magnitude_thresholds_swept": list(magnitude_thresholds),
        "n_folds_total": len(folds),
        "n_folds_evaluated": n_evaluated,
        "wall_clock_seconds": total_s,
        "by_magnitude_threshold": results,
    }
    out_path = PROJECT_ROOT / "reports" / "phase_1" / f"regression_sweep{out_suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    LOG.info("wrote %s (wall=%.1fmin)", out_path, total_s / 60.0)
    return out


def print_report(out: dict) -> None:
    print(f"\n========== Phase 1.0 — L0 Regression Sweep (DR v3.0.21 Phase A) ==========")
    print(f"Asset: {out['asset']}, bar_threshold: {out['bar_threshold']}")
    print(f"Folds total / evaluated: {out['n_folds_total']} / {out['n_folds_evaluated']}")
    print(f"Wall clock: {out['wall_clock_seconds']:.1f}s ({out['wall_clock_seconds']/60:.1f}min)")
    print(f"Magnitude thresholds: {out['magnitude_thresholds_swept']}")
    print(f"Target: {out['target']}\n")
    print("--- Side-by-side aggregates (vs v3.0.20 champion +1.204) ---")
    header = (f"  {'mag_thr':>8}  {'n_trd':>6}  {'tr/fld':>7}  {'0-tr':>5}  "
              f"{'mPnL':>9}  {'win%':>6}  {'L_win%':>7}  {'S_win%':>7}  "
              f"{'Shp_all':>8}  {'Shp_!=0':>8}  {'annret':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mag in out["magnitude_thresholds_swept"]:
        key = f"{mag:.4f}"
        a = out["by_magnitude_threshold"][key]["aggregate"]
        print(f"  {mag:>8.4f}  "
              f"{a.get('n_trades_total', 0):>6}  "
              f"{a.get('trades_per_fold_mean', 0):>7.1f}  "
              f"{a.get('n_folds_zero_trades', 0):>5}  "
              f"{a.get('mean_pnl_bps_net', 0):>+9.2f}  "
              f"{a.get('win_pct_mean', 0):>6.1f}  "
              f"{a.get('long_win_pct_mean', 0):>7.1f}  "
              f"{a.get('short_win_pct_mean', 0):>7.1f}  "
              f"{a.get('sharpe_mean_across_folds', 0):>+8.3f}  "
              f"{a.get('sharpe_mean_nonzero_folds', 0):>+8.3f}  "
              f"{a.get('annual_return_mean', 0):>+7.3f}")
    print()
    print("--- v3.0.20 categorical champion (TB=0.03/thr=0.58 @ 1.5% bars) ---")
    print("  Sharpe(all) = +1.204, n_trades=631, win%=69.5, mPnL=+87")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="run_phase_1_lgbm_reg")
    p.add_argument("--asset", default="BTC")
    p.add_argument("--bar-threshold", type=float, default=0.015,
                   help="DR v3.0.20 champion: 1.5% bars")
    p.add_argument("--features-path", default=None,
                   help="Override; default = features_btc_thr015.parquet")
    p.add_argument("--labels-path", default=None,
                   help="Override; default = labels_btc_thr015.parquet")
    p.add_argument("--mag-thresholds", default="0.005,0.010,0.015,0.020,0.025,0.030",
                   help="Comma-separated magnitude thresholds in log-return units")
    p.add_argument("--out-suffix", default="_thr015_reg")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sym = symbol_short(args.asset).upper()
    feats_path = args.features_path or str(
        PROJECT_ROOT / f"data/storage/features/features_{sym.lower()}_thr{int(args.bar_threshold*1000):03d}.parquet"
    )
    labels_path = args.labels_path or str(
        PROJECT_ROOT / f"data/storage/labels/labels_{sym.lower()}_thr{int(args.bar_threshold*1000):03d}.parquet"
    )
    mag_thrs = tuple(float(x) for x in args.mag_thresholds.split(","))

    try:
        out = run_regression_sweep(
            features_path=feats_path,
            labels_path=labels_path,
            asset=sym,
            bar_threshold=args.bar_threshold,
            magnitude_thresholds=mag_thrs,
            out_suffix=args.out_suffix,
        )
        print_report(out)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
