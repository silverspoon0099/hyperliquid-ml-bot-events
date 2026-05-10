"""Phase 1.0 L0 LightGBM walk-forward orchestrator (DR v3.0.9 + v3.0.10).

Reads features + labels parquets, joins with bars (for close prices),
generates folds, trains LightGBM per fold with Platt calibration on val,
runs OOT backtest, writes per-fold artifacts + lgbm_results.json.

DR v3.0.10 adds `--threshold-sweep` for sensitivity analysis across
{0.50, 0.52, 0.55, 0.58, 0.60} confidence thresholds. Training is shared
per fold; only the backtest re-runs per threshold.

NO L1 ResNet-LSTM in this phase.

CLI:
    python -m scripts.run_phase_1_lgbm                    # full sweep, default threshold
    python -m scripts.run_phase_1_lgbm --first-n 3        # smoke (first 3 folds)
    python -m scripts.run_phase_1_lgbm --dry-run          # build folds, no train
    python -m scripts.run_phase_1_lgbm --threshold-sweep  # DR v3.0.10 sensitivity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cv.walk_forward import generate_folds, split_fold, Fold
from cv.pre_gate import class_prior_entropy, pre_gate_ratio, aggregate_pre_gate
from model.lgbm import train_lgbm, fit_platt, apply_platt, feature_importance_top_k
from backtest.runner import (
    simulate_trades, build_equity_curve, compute_metrics, trades_to_dataframe,
)
from data.db import close_pool, get_engine

LOG = logging.getLogger("scripts.run_phase_1_lgbm")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

KEY_COLS = ["bar_id", "bar_close_ts"]
LABEL_COL = "label"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_features(tier1: bool = False) -> pd.DataFrame:
    """Read features parquet. tier1=True loads the DR v3.0.13 extended set."""
    if tier1:
        path = PROJECT_ROOT / "data/storage/features/features_btc_tier1.parquet"
    else:
        path = PROJECT_ROOT / "data/storage/features/features_btc.parquet"
    return pd.read_parquet(path)


def _load_labels() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "data/storage/labels/labels_btc.parquet")


def _load_bars_close() -> pd.DataFrame:
    sql = "SELECT bar_id, bar_close_ts, close FROM events.bars_btc_cusum ORDER BY bar_close_ts, bar_id"
    return pd.read_sql_query(sql, get_engine())


def _load_bars_ohlc() -> pd.DataFrame:
    """Load full OHLC needed for in-memory relabeling (DR v3.0.11)."""
    sql = """
        SELECT bar_id, bar_open_ts, bar_close_ts, close, high, low
        FROM events.bars_btc_cusum
        ORDER BY bar_close_ts, bar_id
    """
    return pd.read_sql_query(sql, get_engine())


def _md5_array(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _equity_curve_with_position(
    trades: list, oot_bars: pd.DataFrame
) -> pd.DataFrame:
    """Build a sparse (ts, equity, position, signal) frame for §11.5."""
    if not trades:
        return pd.DataFrame(columns=["ts", "equity", "position", "signal"])
    rows = []
    starting_equity = 10_000.0
    cur = starting_equity
    rows.append({"ts": trades[0].entry_ts, "equity": cur,
                 "position": 0, "signal": "init"})
    for t in trades:
        rows.append({"ts": t.entry_ts, "equity": cur,
                     "position": t.direction,
                     "signal": "LONG" if t.direction == 1 else "SHORT"})
        cur *= 1.0 + t.pnl_bps_net / 10000.0
        rows.append({"ts": t.exit_ts, "equity": cur,
                     "position": 0, "signal": f"EXIT_{t.exit_reason}"})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
def run_fold(
    fold: Fold,
    df_full: pd.DataFrame,
    bars_close: pd.DataFrame,
    feature_cols: list[str],
    lgbm_params: dict,
    purge_bars: int,
    embargo_bars: int,
    confidence_threshold: float,
    cost_bps_round_trip: float,
    out_dir: Path,
    dry_run: bool,
) -> dict:
    """Train + OOT backtest for one fold; returns the fold metrics dict."""
    parts = split_fold(df_full, fold,
                       purge_bars=purge_bars, embargo_bars=embargo_bars,
                       ts_col="bar_close_ts")
    train, val, oot = parts["train"], parts["val"], parts["oot"]
    n_train, n_val, n_oot = len(train), len(val), len(oot)

    res: dict = {
        "fold": fold.fold_id,
        "train_start": str(fold.train_start),
        "val_start": str(fold.val_start),
        "val_end": str(fold.val_end),
        "oot_end": str(fold.oot_end),
        "n_train": n_train, "n_val": n_val, "n_oot": n_oot,
        "skipped": False,
        "skip_reason": None,
    }

    if n_val < 100 or n_oot < 100:
        res["skipped"] = True
        res["skip_reason"] = f"n_val={n_val} or n_oot={n_oot} < 100 bars"
        LOG.warning("Fold %d: SKIPPED (%s)", fold.fold_id, res["skip_reason"])
        return res

    if dry_run:
        LOG.info("Fold %d: dry-run (would train on %d / val %d / oot %d)",
                 fold.fold_id, n_train, n_val, n_oot)
        return res

    # ────── Train ──────
    t0 = time.perf_counter()
    booster = train_lgbm(
        train[feature_cols], train[LABEL_COL].astype(int).values,
        val[feature_cols], val[LABEL_COL].astype(int).values,
        dict(lgbm_params),
    )
    train_s = time.perf_counter() - t0

    # ────── Pre-gate (val_logloss / H(p_train)) ──────
    val_raw = booster.predict(val[feature_cols])
    val_ll = log_loss(val[LABEL_COL].astype(int).values, val_raw,
                      labels=[0, 1, 2])
    H_p = class_prior_entropy(train[LABEL_COL].astype(int).values, n_classes=3)
    ratio = val_ll / H_p

    # ────── Calibrate (Platt) on val raw → predict OOT ──────
    cal = fit_platt(val_raw, val[LABEL_COL].astype(int).values, n_classes=3)
    oot_raw = booster.predict(oot[feature_cols])
    oot_cal = apply_platt(oot_raw, cal)

    # ────── Backtest ──────
    preds = pd.DataFrame({
        "bar_id": oot["bar_id"].astype("int64").values,
        "p_long":   oot_cal[:, 0],
        "p_short":  oot_cal[:, 1],
        "p_neutral": oot_cal[:, 2],
    })

    # labels_df subset for OOT bars
    labels_oot = oot[["bar_id", "exit_bar_id", "exit_price",
                      "exit_reason", "holding_bars", "label"]].copy()

    trades = simulate_trades(
        predictions=preds, bars_df=bars_close, labels_df=labels_oot,
        confidence_threshold=confidence_threshold,
        cost_bps_round_trip=cost_bps_round_trip,
        max_concurrent=1,
    )
    eq_for_metrics = build_equity_curve(trades, starting_equity=10_000.0)
    metrics = compute_metrics(trades, eq_for_metrics, oot_n_bars=n_oot)

    # ────── Per-fold artifacts ──────
    fold_dir = out_dir / f"fold_{fold.fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    eq_full = _equity_curve_with_position(trades, oot)
    eq_full.to_csv(fold_dir / "equity_curve.csv", index=False)
    trades_df = trades_to_dataframe(trades)
    trades_df.to_csv(fold_dir / "trades.csv", index=False)
    preds.to_parquet(fold_dir / "predictions.parquet", index=False)

    # ────── Feature importance top-10 (gain) ──────
    fi_top10 = feature_importance_top_k(booster, feature_cols, k=10)

    res.update({
        "val_logloss": float(val_ll),
        "H_p_train": float(H_p),
        "ratio": float(ratio),
        "pre_gate_pass": bool(ratio < 0.99),
        "trees_used": int(booster.best_iteration or booster.num_trees()),
        "train_seconds": train_s,
        **metrics,
        "feature_importance_top10": fi_top10,
        "oot_md5": _md5_array(oot_cal),
    })
    LOG.info(
        "Fold %d: ratio=%.4f pass=%s sharpe=%.3f n_trades=%d trees=%d",
        fold.fold_id, ratio, res["pre_gate_pass"],
        metrics["oot_sharpe"], metrics["oot_n_trades"], res["trees_used"],
    )
    return res


# ─────────────────────────────────────────────────────────────────────
def run(
    first_n: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    cfg = _load_config()
    wf = cfg["walk_forward"]
    pg = cfg["pre_gate"]
    bt = cfg["backtest"]
    lgbm_params = cfg["model"]["L0_lightgbm"]
    confidence_threshold = cfg["model"]["signal_threshold"]
    cost_bps_round_trip = bt["costs_bps_round_trip"]

    LOG.info("loading features + labels + bars...")
    feats = _load_features()
    labels = _load_labels()
    bars_close = _load_bars_close()

    df_full = feats.merge(labels, on="bar_id", how="inner")
    df_full = df_full[df_full[LABEL_COL] != -1].copy()
    df_full["label"] = df_full["label"].astype("int64")
    df_full = df_full.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)
    LOG.info("features+labels merged: %d labelable bars × %d cols",
             len(df_full), len(df_full.columns))

    feature_cols = [c for c in feats.columns if c not in KEY_COLS]
    LOG.info("feature columns: %d (e.g. %s)", len(feature_cols), feature_cols[:3])

    # Generate folds
    data_start = date(df_full["bar_close_ts"].min().year,
                      df_full["bar_close_ts"].min().month, 1)
    data_end_ts = df_full["bar_close_ts"].max()
    # Use the start of the month AFTER the last bar's month as data_end
    if data_end_ts.month == 12:
        data_end = date(data_end_ts.year + 1, 1, 1)
    else:
        data_end = date(data_end_ts.year, data_end_ts.month + 1, 1)

    folds = generate_folds(
        data_start=data_start, data_end=data_end,
        initial_train_months=wf["initial_train_months"],
        val_months=wf["val_months"],
        oot_months=wf["oot_months"],
        step_months=wf["step_months"],
    )
    LOG.info("generated %d folds (data_start=%s, data_end=%s)",
             len(folds), data_start, data_end)
    if first_n is not None:
        folds = folds[:first_n]
        LOG.info("first_n=%d → running %d folds", first_n, len(folds))

    out_dir = PROJECT_ROOT / "reports" / "phase_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict] = []
    t0_total = time.perf_counter()
    for fold in folds:
        try:
            res = run_fold(
                fold=fold, df_full=df_full, bars_close=bars_close,
                feature_cols=feature_cols, lgbm_params=lgbm_params,
                purge_bars=wf["purge_bars"], embargo_bars=wf["embargo_bars"],
                confidence_threshold=confidence_threshold,
                cost_bps_round_trip=cost_bps_round_trip,
                out_dir=out_dir, dry_run=dry_run,
            )
        except Exception:
            LOG.exception("fold %d raised; recording skip", fold.fold_id)
            res = {"fold": fold.fold_id, "skipped": True,
                   "skip_reason": "exception (see log)"}
        fold_results.append(res)
    total_s = time.perf_counter() - t0_total

    # ────── Aggregate ──────
    evaluated = [r for r in fold_results if not r.get("skipped")]
    if dry_run or not evaluated:
        agg = {}
        pg_summary = {}
    else:
        agg = {}
        for k in ("oot_sharpe", "oot_sortino", "oot_max_dd",
                 "oot_pct_time_in_market", "oot_n_trades",
                 "oot_profitable_trade_pct", "oot_annual_return", "ratio"):
            vals = [r[k] for r in evaluated if k in r]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        ratios = [r["ratio"] for r in evaluated if "ratio" in r]
        pg_summary = aggregate_pre_gate(
            ratios, threshold=pg["threshold"],
            required_pass=pg["required_pass_folds"], n_first=6,
        )

    out = {
        "n_folds_total": len(folds),
        "n_folds_evaluated": len(evaluated),
        "n_folds_skipped": len(folds) - len(evaluated),
        "wall_clock_seconds": total_s,
        "pre_gate": pg_summary,
        "aggregate": agg,
        "per_fold": fold_results,
    }
    out_path = out_dir / "lgbm_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    LOG.info("wrote %s", out_path)

    return out


# ─────────────────────────────────────────────────────────────────────
def print_sanity_report(out: dict) -> None:
    print("\n========== Phase 1.0 L0 LightGBM Sanity Report ==========")
    print(f"Folds total / evaluated / skipped: "
          f"{out['n_folds_total']} / {out['n_folds_evaluated']} / {out['n_folds_skipped']}")
    print(f"Wall clock: {out['wall_clock_seconds']:.1f}s")

    pg = out.get("pre_gate") or {}
    if pg:
        print(f"\n--- Pre-gate (first 6 folds) ---")
        print(f"  passed: {pg['n_passed']} / {pg['n_evaluated']}  "
              f"(required ≥ {pg['required_pass']})  → "
              f"{'✓ PASSED' if pg.get('passed') else '✗ FAILED'}")

    print(f"\n--- Per-fold metrics ---")
    print(f"{'fold':>4}  {'val_start':>10}  {'oot_end':>10}  "
          f"{'n_oot':>5}  {'ratio':>6}  {'PG':>3}  "
          f"{'sharpe':>7}  {'sortino':>7}  {'max_dd':>7}  "
          f"{'n_tr':>5}  {'win%':>5}")
    for r in out["per_fold"]:
        if r.get("skipped"):
            print(f"{r['fold']:>4}  SKIPPED ({r.get('skip_reason','')})")
            continue
        print(
            f"{r['fold']:>4}  {r['val_start']:>10}  {r['oot_end']:>10}  "
            f"{r['n_oot']:>5}  {r['ratio']:>6.4f}  "
            f"{'✓' if r['pre_gate_pass'] else '✗':>3}  "
            f"{r['oot_sharpe']:>7.3f}  {r['oot_sortino']:>7.3f}  "
            f"{r['oot_max_dd']:>7.3f}  {r['oot_n_trades']:>5}  "
            f"{r['oot_profitable_trade_pct']:>5.1f}"
        )

    agg = out.get("aggregate") or {}
    if agg:
        print(f"\n--- Aggregate (mean ± std across evaluated folds) ---")
        for k in ("oot_sharpe", "oot_sortino", "oot_max_dd",
                  "oot_pct_time_in_market", "oot_n_trades",
                  "oot_profitable_trade_pct", "oot_annual_return", "ratio"):
            mean = agg.get(f"{k}_mean")
            std  = agg.get(f"{k}_std")
            if mean is not None:
                print(f"  {k:<28} {mean:>9.4f} ± {std:.4f}")

    # Top-10 feature importance from first fold (representative)
    first_eval = next((r for r in out["per_fold"] if not r.get("skipped")), None)
    if first_eval and first_eval.get("feature_importance_top10"):
        print(f"\n--- Top-10 features (gain) — fold {first_eval['fold']} ---")
        for f in first_eval["feature_importance_top10"]:
            print(f"  {f['feature']:<22} gain={f['gain']:>14.2f}  split={f['split']:>5}")

    # Per-fold Sharpe variance interpretation note (DR §15 fold)
    print("\n--- Interpretation ---")
    print("  Per-fold Sharpe is high-variance for thin OOT (~90 daily returns")
    print("  per 3-month window). Mean across folds is the meaningful aggregate;")
    print("  do not over-interpret individual fold swings.")


# ─────────────────────────────────────────────────────────────────────
def run_threshold_sweep(
    first_n: Optional[int] = None,
    thresholds: tuple = (0.50, 0.52, 0.55, 0.58, 0.60),
) -> dict:
    """DR v3.0.10: train once per fold, backtest 5 times across thresholds.

    Output schema mirrors DR §"Output". Training shared per fold to avoid
    re-fitting LightGBM 5×; only `simulate_trades` + metrics re-run per
    threshold.
    """
    cfg = _load_config()
    wf = cfg["walk_forward"]
    bt = cfg["backtest"]
    lgbm_params = cfg["model"]["L0_lightgbm"]
    cost_bps_round_trip = bt["costs_bps_round_trip"]

    LOG.info("loading features + labels + bars...")
    feats = _load_features()
    labels = _load_labels()
    bars_close = _load_bars_close()

    df_full = feats.merge(labels, on="bar_id", how="inner")
    df_full = df_full[df_full[LABEL_COL] != -1].copy()
    df_full["label"] = df_full["label"].astype("int64")
    df_full = df_full.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)

    feature_cols = [c for c in feats.columns if c not in KEY_COLS]

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
    if first_n is not None:
        folds = folds[:first_n]
    LOG.info("folds=%d  thresholds=%s", len(folds), list(thresholds))

    # Per-threshold accumulator: {thr: [per-fold dicts]}
    by_threshold: dict[str, list[dict]] = {f"{t:.2f}": [] for t in thresholds}
    n_evaluated = 0

    t0_total = time.perf_counter()
    for fold in folds:
        parts = split_fold(df_full, fold,
                           purge_bars=wf["purge_bars"],
                           embargo_bars=wf["embargo_bars"],
                           ts_col="bar_close_ts")
        train, val, oot = parts["train"], parts["val"], parts["oot"]
        n_train, n_val, n_oot = len(train), len(val), len(oot)

        if n_val < 100 or n_oot < 100:
            LOG.warning("Fold %d: SKIPPED (n_val=%d or n_oot=%d < 100)",
                        fold.fold_id, n_val, n_oot)
            continue
        n_evaluated += 1

        # Train + calibrate (shared across thresholds)
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

        for thr in thresholds:
            trades = simulate_trades(
                predictions=preds, bars_df=bars_close, labels_df=labels_oot,
                confidence_threshold=thr,
                cost_bps_round_trip=cost_bps_round_trip,
                max_concurrent=1,
            )
            eq = build_equity_curve(trades, starting_equity=10_000.0)
            metrics = compute_metrics(trades, eq, oot_n_bars=n_oot)
            n_trades = metrics["oot_n_trades"]
            mean_pnl = float(np.mean([t.pnl_bps_net for t in trades])) if trades else 0.0
            median_pnl = float(np.median([t.pnl_bps_net for t in trades])) if trades else 0.0
            n_long = sum(1 for t in trades if t.direction == 1)
            n_short = sum(1 for t in trades if t.direction == -1)
            n_long_win = sum(1 for t in trades if t.direction == 1 and t.pnl_bps_net > 0)
            n_short_win = sum(1 for t in trades if t.direction == -1 and t.pnl_bps_net > 0)

            by_threshold[f"{thr:.2f}"].append({
                "fold": fold.fold_id,
                "n_oot": n_oot,
                "n_trades": n_trades,
                "n_long": n_long, "n_short": n_short,
                "mean_pnl_bps_net": mean_pnl,
                "median_pnl_bps_net": median_pnl,
                "win_pct": metrics["oot_profitable_trade_pct"],
                "long_win_pct": (n_long_win / n_long * 100.0) if n_long > 0 else 0.0,
                "short_win_pct": (n_short_win / n_short * 100.0) if n_short > 0 else 0.0,
                "sharpe": metrics["oot_sharpe"],
                "sortino": metrics["oot_sortino"],
                "max_dd": metrics["oot_max_dd"],
                "annual_return": metrics["oot_annual_return"],
                "pct_time_in_market": metrics["oot_pct_time_in_market"],
            })
        LOG.info("Fold %d: trained; backtested %d thresholds",
                 fold.fold_id, len(thresholds))

    total_s = time.perf_counter() - t0_total

    # Per-threshold aggregates
    results = {}
    for thr_str, fold_rows in by_threshold.items():
        if not fold_rows:
            results[thr_str] = {"aggregate": {}, "per_fold": []}
            continue
        n_total_trades = sum(r["n_trades"] for r in fold_rows)
        # Pool all per-trade PnLs (weighted by occurrence, not by fold)
        all_pnl = []
        for r in fold_rows:
            if r["n_trades"] > 0:
                all_pnl.extend([r["mean_pnl_bps_net"]] * r["n_trades"])
        # Per-fold sharpe distribution
        sharpes = [r["sharpe"] for r in fold_rows]
        nonzero_sharpes = [s for s in sharpes if s != 0.0]
        results[thr_str] = {
            "aggregate": {
                "n_trades_total": n_total_trades,
                "trades_per_fold_mean": n_total_trades / len(fold_rows),
                "trades_per_fold_median": float(np.median([r["n_trades"] for r in fold_rows])),
                "n_folds_with_trades": sum(1 for r in fold_rows if r["n_trades"] > 0),
                "mean_pnl_bps_net": float(np.mean([r["mean_pnl_bps_net"] for r in fold_rows
                                                    if r["n_trades"] > 0])) if any(r["n_trades"] for r in fold_rows) else 0.0,
                "median_pnl_bps_net": float(np.median([r["median_pnl_bps_net"] for r in fold_rows
                                                        if r["n_trades"] > 0])) if any(r["n_trades"] for r in fold_rows) else 0.0,
                "win_pct_mean": float(np.mean([r["win_pct"] for r in fold_rows
                                                if r["n_trades"] > 0])) if any(r["n_trades"] for r in fold_rows) else 0.0,
                "long_win_pct_mean": float(np.mean([r["long_win_pct"] for r in fold_rows
                                                     if r["n_long"] > 0])) if any(r["n_long"] for r in fold_rows) else 0.0,
                "short_win_pct_mean": float(np.mean([r["short_win_pct"] for r in fold_rows
                                                      if r["n_short"] > 0])) if any(r["n_short"] for r in fold_rows) else 0.0,
                "sharpe_mean_across_folds": float(np.mean(sharpes)),
                "sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
                "sharpe_mean_nonzero_folds": float(np.mean(nonzero_sharpes)) if nonzero_sharpes else 0.0,
                "n_folds_zero_trades": sum(1 for r in fold_rows if r["n_trades"] == 0),
                "annual_return_mean": float(np.mean([r["annual_return"] for r in fold_rows])),
            },
            "per_fold": fold_rows,
        }

    out = {
        "thresholds_swept": list(thresholds),
        "n_folds_total": len(folds),
        "n_folds_evaluated": n_evaluated,
        "wall_clock_seconds": total_s,
        "by_threshold": results,
    }
    out_path = PROJECT_ROOT / "reports" / "phase_1" / "threshold_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    LOG.info("wrote %s", out_path)
    return out


def print_threshold_sweep_report(out: dict) -> None:
    print("\n========== Phase 1.0 — Threshold Sweep (DR v3.0.10) ==========")
    print(f"Folds total / evaluated: {out['n_folds_total']} / {out['n_folds_evaluated']}")
    print(f"Wall clock: {out['wall_clock_seconds']:.1f}s")
    print(f"Thresholds swept: {out['thresholds_swept']}")

    print("\n--- Side-by-side aggregates ---")
    cols = [
        ("threshold",            lambda thr, a: f"{float(thr):.2f}"),
        ("n_trades",             lambda thr, a: f"{a.get('n_trades_total', 0):>5}"),
        ("trades/fold",          lambda thr, a: f"{a.get('trades_per_fold_mean', 0):>6.1f}"),
        ("zero-trade folds",     lambda thr, a: f"{a.get('n_folds_zero_trades', 0):>4}"),
        ("mean PnL (bps)",       lambda thr, a: f"{a.get('mean_pnl_bps_net', 0):>+8.2f}"),
        ("median PnL (bps)",     lambda thr, a: f"{a.get('median_pnl_bps_net', 0):>+8.2f}"),
        ("win%",                 lambda thr, a: f"{a.get('win_pct_mean', 0):>5.1f}"),
        ("LONG win%",            lambda thr, a: f"{a.get('long_win_pct_mean', 0):>5.1f}"),
        ("SHORT win%",           lambda thr, a: f"{a.get('short_win_pct_mean', 0):>5.1f}"),
        ("Sharpe (all folds)",   lambda thr, a: f"{a.get('sharpe_mean_across_folds', 0):>+6.3f}"),
        ("Sharpe (nonzero)",     lambda thr, a: f"{a.get('sharpe_mean_nonzero_folds', 0):>+6.3f}"),
        ("ann. ret",             lambda thr, a: f"{a.get('annual_return_mean', 0):>+6.3f}"),
    ]
    header = f"  {'thr':>5}  {'n_trd':>6}  {'tr/fld':>7}  {'0-tr':>5}  {'mPnL':>9}  {'medPnL':>9}  {'win%':>6}  {'L_win%':>7}  {'S_win%':>7}  {'Shp_all':>8}  {'Shp_!=0':>8}  {'annret':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for thr in out["thresholds_swept"]:
        a = out["by_threshold"][f"{thr:.2f}"]["aggregate"]
        print(f"  {thr:>5.2f}  "
              f"{a.get('n_trades_total', 0):>6}  "
              f"{a.get('trades_per_fold_mean', 0):>7.1f}  "
              f"{a.get('n_folds_zero_trades', 0):>5}  "
              f"{a.get('mean_pnl_bps_net', 0):>+9.2f}  "
              f"{a.get('median_pnl_bps_net', 0):>+9.2f}  "
              f"{a.get('win_pct_mean', 0):>6.1f}  "
              f"{a.get('long_win_pct_mean', 0):>7.1f}  "
              f"{a.get('short_win_pct_mean', 0):>7.1f}  "
              f"{a.get('sharpe_mean_across_folds', 0):>+8.3f}  "
              f"{a.get('sharpe_mean_nonzero_folds', 0):>+8.3f}  "
              f"{a.get('annual_return_mean', 0):>+7.3f}")

    print("\n--- Per-fold n_trades by threshold ---")
    fold_ids = [r["fold"] for r in out["by_threshold"][f"{out['thresholds_swept'][0]:.2f}"]["per_fold"]]
    print(f"  fold  " + "  ".join(f"{thr:>5.2f}" for thr in out["thresholds_swept"]))
    for i, fold_id in enumerate(fold_ids):
        row = []
        for thr in out["thresholds_swept"]:
            n = out["by_threshold"][f"{thr:.2f}"]["per_fold"][i]["n_trades"]
            row.append(f"{n:>5}")
        print(f"  {fold_id:>4}  " + "  ".join(row))

    print("\n--- Interpretation note ---")
    print("  Per-fold Sharpe is high-variance for thin OOT (~90 daily returns")
    print("  per 3-month window). Mean across folds is the meaningful aggregate.")
    print("  'Sharpe (nonzero)' excludes folds where threshold filtered all signals.")


def run_tb_sweep(
    first_n: Optional[int] = None,
    tb_values: tuple = (0.03, 0.04, 0.05, 0.06, 0.07),
) -> dict:
    """DR v3.0.11: §16.4 step (1) TB sweep. In-memory relabel per TB,
    full L0 walk-forward at default 0.60 confidence threshold.
    """
    from labels.triple_barrier import apply_triple_barrier

    cfg = _load_config()
    wf = cfg["walk_forward"]
    bt = cfg["backtest"]
    lgbm_params = cfg["model"]["L0_lightgbm"]
    confidence_threshold = cfg["model"]["signal_threshold"]  # default 0.60
    cost_bps_round_trip = bt["costs_bps_round_trip"]
    vertical_bars = cfg["labeling"]["vertical_bars"]

    LOG.info("loading features + bars (with OHLC)...")
    feats = _load_features()
    bars_full = _load_bars_ohlc()
    feature_cols = [c for c in feats.columns if c not in KEY_COLS]

    data_start = date(bars_full["bar_close_ts"].min().year,
                      bars_full["bar_close_ts"].min().month, 1)
    data_end_ts = bars_full["bar_close_ts"].max()
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
    if first_n is not None:
        folds = folds[:first_n]
    LOG.info("folds=%d  TB values=%s", len(folds), list(tb_values))

    by_tb: dict[str, dict] = {}
    t0_total = time.perf_counter()

    for tb in tb_values:
        LOG.info("===== TB sweep value: tp=sl=%.2f =====", tb)
        t_tb = time.perf_counter()

        # In-memory relabel
        labels_df = apply_triple_barrier(
            bars_full[["bar_id", "bar_open_ts", "bar_close_ts", "close", "high", "low"]],
            tp_pct=tb, sl_pct=tb, vertical_bars=vertical_bars,
        )
        df_full = feats.merge(labels_df, on="bar_id", how="inner")
        df_full = df_full[df_full[LABEL_COL] != -1].copy()
        df_full["label"] = df_full["label"].astype("int64")
        df_full = df_full.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)

        fold_rows = []
        for fold in folds:
            parts = split_fold(df_full, fold,
                               purge_bars=wf["purge_bars"],
                               embargo_bars=wf["embargo_bars"],
                               ts_col="bar_close_ts")
            train, val, oot = parts["train"], parts["val"], parts["oot"]
            if len(val) < 100 or len(oot) < 100:
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
            mean_pnl = float(np.mean([t.pnl_bps_net for t in trades])) if trades else 0.0
            median_pnl = float(np.median([t.pnl_bps_net for t in trades])) if trades else 0.0
            n_long = sum(1 for t in trades if t.direction == 1)
            n_short = sum(1 for t in trades if t.direction == -1)
            n_long_win = sum(1 for t in trades if t.direction == 1 and t.pnl_bps_net > 0)
            n_short_win = sum(1 for t in trades if t.direction == -1 and t.pnl_bps_net > 0)

            fold_rows.append({
                "fold": fold.fold_id, "n_oot": len(oot),
                "n_trades": metrics["oot_n_trades"],
                "n_long": n_long, "n_short": n_short,
                "mean_pnl_bps_net": mean_pnl,
                "median_pnl_bps_net": median_pnl,
                "win_pct": metrics["oot_profitable_trade_pct"],
                "long_win_pct": (n_long_win / n_long * 100.0) if n_long > 0 else 0.0,
                "short_win_pct": (n_short_win / n_short * 100.0) if n_short > 0 else 0.0,
                "sharpe": metrics["oot_sharpe"],
                "sortino": metrics["oot_sortino"],
                "max_dd": metrics["oot_max_dd"],
                "annual_return": metrics["oot_annual_return"],
            })

        sharpes = [r["sharpe"] for r in fold_rows]
        nonzero_sharpes = [s for s in sharpes if s != 0.0]
        n_total_trades = sum(r["n_trades"] for r in fold_rows)
        any_trade_folds = [r for r in fold_rows if r["n_trades"] > 0]
        any_long_folds = [r for r in fold_rows if r["n_long"] > 0]
        any_short_folds = [r for r in fold_rows if r["n_short"] > 0]

        by_tb[f"{tb:.2f}"] = {
            "aggregate": {
                "n_trades_total": n_total_trades,
                "trades_per_fold_mean": n_total_trades / max(len(fold_rows), 1),
                "n_folds_with_trades": len(any_trade_folds),
                "mean_pnl_bps_net": float(np.mean([r["mean_pnl_bps_net"] for r in any_trade_folds])) if any_trade_folds else 0.0,
                "median_pnl_bps_net": float(np.median([r["median_pnl_bps_net"] for r in any_trade_folds])) if any_trade_folds else 0.0,
                "win_pct_mean": float(np.mean([r["win_pct"] for r in any_trade_folds])) if any_trade_folds else 0.0,
                "long_win_pct_mean": float(np.mean([r["long_win_pct"] for r in any_long_folds])) if any_long_folds else 0.0,
                "short_win_pct_mean": float(np.mean([r["short_win_pct"] for r in any_short_folds])) if any_short_folds else 0.0,
                "sharpe_mean_across_folds": float(np.mean(sharpes)) if sharpes else 0.0,
                "sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
                "sharpe_mean_nonzero_folds": float(np.mean(nonzero_sharpes)) if nonzero_sharpes else 0.0,
                "n_folds_zero_trades": sum(1 for r in fold_rows if r["n_trades"] == 0),
                "annual_return_mean": float(np.mean([r["annual_return"] for r in fold_rows])) if fold_rows else 0.0,
            },
            "per_fold": fold_rows,
        }
        LOG.info("TB %.2f done in %.1fs: n_trades=%d sharpe(all)=%.3f sharpe(!=0)=%.3f",
                 tb, time.perf_counter() - t_tb,
                 n_total_trades,
                 by_tb[f"{tb:.2f}"]["aggregate"]["sharpe_mean_across_folds"],
                 by_tb[f"{tb:.2f}"]["aggregate"]["sharpe_mean_nonzero_folds"])

    total_s = time.perf_counter() - t0_total
    out = {
        "tb_values_swept": list(tb_values),
        "n_folds_total": len(folds),
        "wall_clock_seconds": total_s,
        "by_tb": by_tb,
    }
    out_path = PROJECT_ROOT / "reports" / "phase_1" / "tb_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    LOG.info("wrote %s", out_path)
    return out


def print_tb_sweep_report(out: dict) -> None:
    print("\n========== Phase 1.0 — TB Sweep (DR v3.0.11, §16.4 step 1) ==========")
    print(f"Folds total: {out['n_folds_total']}")
    print(f"Wall clock:  {out['wall_clock_seconds']:.1f}s")
    print(f"TB values swept: {out['tb_values_swept']}")

    print("\n--- Side-by-side aggregates (default 0.60 confidence threshold) ---")
    header = (f"  {'TB':>5}  {'n_trd':>6}  {'tr/fld':>7}  {'0-tr':>5}  "
              f"{'mPnL':>9}  {'medPnL':>10}  {'win%':>6}  "
              f"{'L_win%':>7}  {'S_win%':>7}  {'Shp_all':>8}  {'Shp_!=0':>8}  {'annret':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tb in out["tb_values_swept"]:
        a = out["by_tb"][f"{tb:.2f}"]["aggregate"]
        print(f"  {tb:>5.2f}  "
              f"{a['n_trades_total']:>6}  "
              f"{a['trades_per_fold_mean']:>7.1f}  "
              f"{a['n_folds_zero_trades']:>5}  "
              f"{a['mean_pnl_bps_net']:>+9.2f}  "
              f"{a['median_pnl_bps_net']:>+10.2f}  "
              f"{a['win_pct_mean']:>6.1f}  "
              f"{a['long_win_pct_mean']:>7.1f}  "
              f"{a['short_win_pct_mean']:>7.1f}  "
              f"{a['sharpe_mean_across_folds']:>+8.3f}  "
              f"{a['sharpe_mean_nonzero_folds']:>+8.3f}  "
              f"{a['annual_return_mean']:>+7.3f}")

    print("\n--- Per-fold n_trades by TB ---")
    if out["tb_values_swept"]:
        first_tb_str = f"{out['tb_values_swept'][0]:.2f}"
        fold_ids = [r["fold"] for r in out["by_tb"][first_tb_str]["per_fold"]]
        print(f"  fold  " + "  ".join(f"TB={tb:.2f}" for tb in out["tb_values_swept"]))
        for i, fold_id in enumerate(fold_ids):
            row = []
            for tb in out["tb_values_swept"]:
                n = out["by_tb"][f"{tb:.2f}"]["per_fold"][i]["n_trades"]
                row.append(f"{n:>6}")
            print(f"  {fold_id:>4}  " + "  ".join(row))

    print("\n--- Interpretation note ---")
    print("  Confidence threshold fixed at 0.60 (default per §10.1).")
    print("  Per-fold Sharpe is high-variance for thin OOT (~90 daily returns)")
    print("  per 3-month window). Mean across folds is the meaningful aggregate.")


def run_joint_tb_threshold_sweep(
    first_n: Optional[int] = None,
    tb: float = 0.03,
    thresholds: tuple = (0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65),
    tier1: bool = False,
) -> dict:
    """DR v3.0.12 / v3.0.13: TB=0.03 × threshold sweep. In-memory relabel
    once; training shared per fold; backtest per threshold. tier1=True
    reads the extended 48-feature parquet."""
    from labels.triple_barrier import apply_triple_barrier

    cfg = _load_config()
    wf = cfg["walk_forward"]
    bt = cfg["backtest"]
    lgbm_params = cfg["model"]["L0_lightgbm"]
    cost_bps_round_trip = bt["costs_bps_round_trip"]
    vertical_bars = cfg["labeling"]["vertical_bars"]

    LOG.info("loading features (tier1=%s) + bars (with OHLC)...", tier1)
    feats = _load_features(tier1=tier1)
    bars_full = _load_bars_ohlc()
    feature_cols = [c for c in feats.columns if c not in KEY_COLS]

    LOG.info("relabeling at TB=tp=sl=%.2f...", tb)
    labels_df = apply_triple_barrier(
        bars_full[["bar_id", "bar_open_ts", "bar_close_ts", "close", "high", "low"]],
        tp_pct=tb, sl_pct=tb, vertical_bars=vertical_bars,
    )
    df_full = feats.merge(labels_df, on="bar_id", how="inner")
    df_full = df_full[df_full[LABEL_COL] != -1].copy()
    df_full["label"] = df_full["label"].astype("int64")
    df_full = df_full.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)
    LOG.info("relabel done: %d labelable bars", len(df_full))

    data_start = date(bars_full["bar_close_ts"].min().year,
                      bars_full["bar_close_ts"].min().month, 1)
    data_end_ts = bars_full["bar_close_ts"].max()
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
    if first_n is not None:
        folds = folds[:first_n]
    LOG.info("folds=%d  thresholds=%s", len(folds), list(thresholds))

    by_threshold: dict[str, list[dict]] = {f"{t:.2f}": [] for t in thresholds}
    n_evaluated = 0

    t0_total = time.perf_counter()
    for fold in folds:
        parts = split_fold(df_full, fold,
                           purge_bars=wf["purge_bars"],
                           embargo_bars=wf["embargo_bars"],
                           ts_col="bar_close_ts")
        train, val, oot = parts["train"], parts["val"], parts["oot"]
        if len(val) < 100 or len(oot) < 100:
            LOG.warning("Fold %d: SKIPPED (n_val=%d or n_oot=%d < 100)",
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

        for thr in thresholds:
            trades = simulate_trades(
                predictions=preds,
                bars_df=bars_full[["bar_id", "bar_close_ts", "close"]],
                labels_df=labels_oot,
                confidence_threshold=thr,
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
            by_threshold[f"{thr:.2f}"].append({
                "fold": fold.fold_id, "n_oot": len(oot),
                "n_trades": metrics["oot_n_trades"],
                "n_long": n_long, "n_short": n_short,
                "mean_pnl_bps_net": mean_pnl,
                "median_pnl_bps_net": median_pnl,
                "win_pct": metrics["oot_profitable_trade_pct"],
                "long_win_pct": (n_long_win / n_long * 100.0) if n_long > 0 else 0.0,
                "short_win_pct": (n_short_win / n_short * 100.0) if n_short > 0 else 0.0,
                "sharpe": metrics["oot_sharpe"],
                "sortino": metrics["oot_sortino"],
                "max_dd": metrics["oot_max_dd"],
                "annual_return": metrics["oot_annual_return"],
                "pct_time_in_market": metrics["oot_pct_time_in_market"],
            })
        LOG.info("Fold %d: trained; backtested %d thresholds",
                 fold.fold_id, len(thresholds))

    total_s = time.perf_counter() - t0_total

    results = {}
    for thr_str, fold_rows in by_threshold.items():
        if not fold_rows:
            results[thr_str] = {"aggregate": {}, "per_fold": []}
            continue
        sharpes = [r["sharpe"] for r in fold_rows]
        nonzero_sharpes = [s for s in sharpes if s != 0.0]
        any_trade_folds = [r for r in fold_rows if r["n_trades"] > 0]
        any_long_folds = [r for r in fold_rows if r["n_long"] > 0]
        any_short_folds = [r for r in fold_rows if r["n_short"] > 0]
        n_total_trades = sum(r["n_trades"] for r in fold_rows)
        results[thr_str] = {
            "aggregate": {
                "n_trades_total": n_total_trades,
                "trades_per_fold_mean": n_total_trades / len(fold_rows),
                "n_folds_with_trades": len(any_trade_folds),
                "n_folds_zero_trades": sum(1 for r in fold_rows if r["n_trades"] == 0),
                "mean_pnl_bps_net": float(np.mean([r["mean_pnl_bps_net"] for r in any_trade_folds])) if any_trade_folds else 0.0,
                "median_pnl_bps_net": float(np.median([r["median_pnl_bps_net"] for r in any_trade_folds])) if any_trade_folds else 0.0,
                "win_pct_mean": float(np.mean([r["win_pct"] for r in any_trade_folds])) if any_trade_folds else 0.0,
                "long_win_pct_mean": float(np.mean([r["long_win_pct"] for r in any_long_folds])) if any_long_folds else 0.0,
                "short_win_pct_mean": float(np.mean([r["short_win_pct"] for r in any_short_folds])) if any_short_folds else 0.0,
                "sharpe_mean_across_folds": float(np.mean(sharpes)),
                "sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
                "sharpe_mean_nonzero_folds": float(np.mean(nonzero_sharpes)) if nonzero_sharpes else 0.0,
                "annual_return_mean": float(np.mean([r["annual_return"] for r in fold_rows])),
            },
            "per_fold": fold_rows,
        }

    # Capture per-threshold top-10 feature importance from the LAST fold
    # we trained (representative) — re-runs training-free is too cumbersome
    # at this nesting; we already log per-fold during the loop.
    out = {
        "tb": tb,
        "thresholds_swept": list(thresholds),
        "n_folds_total": len(folds),
        "n_folds_evaluated": n_evaluated,
        "wall_clock_seconds": total_s,
        "by_threshold": results,
        "tier1_features": tier1,
        "feature_count": len(feature_cols),
    }
    fname = ("joint_tb03_threshold_sweep_tier1.json"
             if tier1 else "joint_tb03_threshold_sweep.json")
    out_path = PROJECT_ROOT / "reports" / "phase_1" / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    LOG.info("wrote %s", out_path)
    return out


def print_joint_sweep_report(out: dict) -> None:
    print(f"\n========== Phase 1.0 — Joint TB={out['tb']:.2f} × Threshold Sweep (DR v3.0.12) ==========")
    print(f"Folds total / evaluated: {out['n_folds_total']} / {out['n_folds_evaluated']}")
    print(f"Wall clock: {out['wall_clock_seconds']:.1f}s")
    print(f"TB held constant at: {out['tb']}")
    print(f"Thresholds swept: {out['thresholds_swept']}")

    print("\n--- Side-by-side aggregates ---")
    header = (f"  {'thr':>5}  {'n_trd':>6}  {'tr/fld':>7}  {'0-tr':>5}  "
              f"{'mPnL':>9}  {'medPnL':>10}  {'win%':>6}  "
              f"{'L_win%':>7}  {'S_win%':>7}  {'Shp_all':>8}  {'Shp_!=0':>8}  {'annret':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for thr in out["thresholds_swept"]:
        a = out["by_threshold"][f"{thr:.2f}"]["aggregate"]
        print(f"  {thr:>5.2f}  "
              f"{a.get('n_trades_total', 0):>6}  "
              f"{a.get('trades_per_fold_mean', 0):>7.1f}  "
              f"{a.get('n_folds_zero_trades', 0):>5}  "
              f"{a.get('mean_pnl_bps_net', 0):>+9.2f}  "
              f"{a.get('median_pnl_bps_net', 0):>+10.2f}  "
              f"{a.get('win_pct_mean', 0):>6.1f}  "
              f"{a.get('long_win_pct_mean', 0):>7.1f}  "
              f"{a.get('short_win_pct_mean', 0):>7.1f}  "
              f"{a.get('sharpe_mean_across_folds', 0):>+8.3f}  "
              f"{a.get('sharpe_mean_nonzero_folds', 0):>+8.3f}  "
              f"{a.get('annual_return_mean', 0):>+7.3f}")

    print("\n--- Per-fold n_trades by threshold ---")
    if out["thresholds_swept"]:
        first_thr = f"{out['thresholds_swept'][0]:.2f}"
        if out["by_threshold"][first_thr]["per_fold"]:
            fold_ids = [r["fold"] for r in out["by_threshold"][first_thr]["per_fold"]]
            print("  fold  " + "  ".join(f"thr={t:.2f}" for t in out["thresholds_swept"]))
            for i, fold_id in enumerate(fold_ids):
                row = []
                for thr in out["thresholds_swept"]:
                    n = out["by_threshold"][f"{thr:.2f}"]["per_fold"][i]["n_trades"]
                    row.append(f"{n:>7}")
                print(f"  {fold_id:>4}  " + "  ".join(row))

    print("\n--- Interpretation note ---")
    print(f"  TB held at {out['tb']} (DR v3.0.11 best). Confidence threshold varies.")
    print("  Per-fold Sharpe is high-variance for thin OOT (~90 daily returns")
    print("  per 3-month window). Mean across folds is the meaningful aggregate.")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="run_phase_1_lgbm")
    p.add_argument("--first-n", type=int, default=None,
                   help="Smoke: run only the first N folds.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build folds + report sample sizes, no training.")
    p.add_argument("--threshold-sweep", action="store_true",
                   help="DR v3.0.10: sensitivity analysis across "
                        "{0.50, 0.52, 0.55, 0.58, 0.60} confidence thresholds.")
    p.add_argument("--tb-sweep", action="store_true",
                   help="DR v3.0.11: §16.4 step (1) — sensitivity analysis "
                        "across TB ∈ {0.03, 0.04, 0.05, 0.06, 0.07}.")
    p.add_argument("--joint-sweep", action="store_true",
                   help="DR v3.0.12: TB=0.03 × threshold sweep "
                        "across {0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65}.")
    p.add_argument("--tier1-features", action="store_true",
                   help="DR v3.0.13: read features_btc_tier1.parquet "
                        "(48 features) instead of features_btc.parquet "
                        "(33 features). Used with --joint-sweep.")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.joint_sweep:
            out = run_joint_tb_threshold_sweep(
                first_n=args.first_n, tier1=args.tier1_features,
            )
            print_joint_sweep_report(out)
        elif args.tb_sweep:
            out = run_tb_sweep(first_n=args.first_n)
            print_tb_sweep_report(out)
        elif args.threshold_sweep:
            out = run_threshold_sweep(first_n=args.first_n)
            print_threshold_sweep_report(out)
        else:
            out = run(first_n=args.first_n, dry_run=args.dry_run)
            print_sanity_report(out)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
