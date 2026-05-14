"""Phase 1.1 L1 ResNet-LSTM walk-forward orchestrator (DR v3.0.17).

Reads features + labels parquets, generates 96-bar sequences per bar,
trains PyTorch ResNet-LSTM per fold with early stopping on val_logloss,
Platt-calibrates probabilities on val, runs OOT backtest, writes
per-config JSON results.

Architecture and hyperparams per `model/resnet_lstm.py` L1_CONFIGS.

CLI:
    # Smoke (1 config × 1 fold × 3 epochs):
    python -m scripts.run_phase_1_resnet_lstm --first-n 1 --smoke

    # Single config across all folds:
    python -m scripts.run_phase_1_resnet_lstm --config B_medium

    # Mini-Hyperband (5 configs × 18 folds):
    python -m scripts.run_phase_1_resnet_lstm --all-configs
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
from cv.pre_gate import class_prior_entropy, aggregate_pre_gate
from model.lgbm import fit_platt, apply_platt
from model.resnet_lstm import (
    L1_CONFIGS, L1Config, build_sequences,
    train_resnet_lstm, predict_proba,
)
from backtest.runner import (
    simulate_trades, build_equity_curve, compute_metrics,
)
from data.db import close_pool, bars_table, symbol_short, get_engine

LOG = logging.getLogger("scripts.run_phase_1_resnet_lstm")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

KEY_COLS = ["bar_id", "bar_close_ts"]
LABEL_COL = "label"
SEQ_LEN = 96


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_features(asset: str = "BTC") -> pd.DataFrame:
    """DR v3.0.17: 33-feature v1 baseline."""
    sym = symbol_short(asset)
    return pd.read_parquet(PROJECT_ROOT / f"data/storage/features/features_{sym}.parquet")


def _load_labels(asset: str = "BTC") -> pd.DataFrame:
    sym = symbol_short(asset)
    return pd.read_parquet(PROJECT_ROOT / f"data/storage/labels/labels_{sym}.parquet")


def _load_bars_ohlc(asset: str = "BTC") -> pd.DataFrame:
    sql = (f"SELECT bar_id, bar_open_ts, bar_close_ts, open, high, low, close "
           f"FROM {bars_table(asset)} WHERE threshold_pct=0.02 ORDER BY bar_close_ts, bar_id")
    return pd.read_sql_query(sql, get_engine())


def _standardize(train_X: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Z-score using train statistics. NaN-safe: leave NaN as NaN.
    Returns (standardized_train, *standardized_others)."""
    mu = np.nanmean(train_X, axis=0, keepdims=True)
    sd = np.nanstd(train_X, axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)  # avoid div-by-zero
    def apply(X):
        return ((X - mu) / sd).astype(np.float32)
    out = [apply(train_X)] + [apply(o) for o in others]
    return tuple(out)


def run_fold_l1(
    fold,
    df_full: pd.DataFrame,
    bars_full: pd.DataFrame,
    feature_cols: list[str],
    cfg_l1: L1Config,
    purge_bars: int,
    embargo_bars: int,
    confidence_threshold: float,
    cost_bps_round_trip: float,
    seed: int = 42,
    threshold_sweep: Optional[list[float]] = None,
) -> dict:
    """Train ResNet-LSTM + OOT backtest for one fold."""
    parts = split_fold(df_full, fold, purge_bars=purge_bars,
                       embargo_bars=embargo_bars, ts_col="bar_close_ts")
    train, val, oot = parts["train"], parts["val"], parts["oot"]
    n_train, n_val, n_oot = len(train), len(val), len(oot)

    res: dict = {
        "fold": fold.fold_id,
        "config": cfg_l1.name,
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
        LOG.warning("Fold %d (%s): SKIPPED (%s)", fold.fold_id, cfg_l1.name, res["skip_reason"])
        return res

    # ────── Build sequences (96-bar windows) ──────
    # Use the FULL df_full (all bars in time order) to assemble per-bar
    # history, then select training/val/oot bars by index position.
    df_sorted = df_full.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)
    bar_id_to_pos = {bid: i for i, bid in enumerate(df_sorted["bar_id"].values)}

    all_features_np = df_sorted[feature_cols].astype(np.float32).values
    all_labels_np = df_sorted[LABEL_COL].astype(np.int64).values

    train_pos = np.array([bar_id_to_pos[b] for b in train["bar_id"].values
                          if b in bar_id_to_pos], dtype=np.int64)
    val_pos = np.array([bar_id_to_pos[b] for b in val["bar_id"].values
                        if b in bar_id_to_pos], dtype=np.int64)
    oot_pos = np.array([bar_id_to_pos[b] for b in oot["bar_id"].values
                        if b in bar_id_to_pos], dtype=np.int64)

    # Standardize features using TRAIN ONLY mean/std (applied to all)
    train_X_flat = all_features_np[train_pos]
    feats_std, = _standardize(train_X_flat,)  # mu/sd on train only
    # Re-apply to full features array using train stats:
    mu = np.nanmean(train_X_flat, axis=0, keepdims=True)
    sd = np.nanstd(train_X_flat, axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    all_features_std = ((all_features_np - mu) / sd).astype(np.float32)

    X_tr, y_tr, kept_tr = build_sequences(all_features_std, all_labels_np, train_pos, SEQ_LEN)
    X_va, y_va, kept_va = build_sequences(all_features_std, all_labels_np, val_pos, SEQ_LEN)
    X_oo, _, kept_oo = build_sequences(all_features_std, all_labels_np, oot_pos, SEQ_LEN)

    if len(X_tr) < 100 or len(X_va) < 50:
        res["skipped"] = True
        res["skip_reason"] = f"too few sequences (tr={len(X_tr)}, va={len(X_va)})"
        LOG.warning("Fold %d (%s): SKIPPED (%s)", fold.fold_id, cfg_l1.name, res["skip_reason"])
        return res

    LOG.info("Fold %d (%s): seqs train=%d val=%d oot=%d  | training...",
             fold.fold_id, cfg_l1.name, len(X_tr), len(X_va), len(X_oo))

    t0 = time.perf_counter()
    model, history = train_resnet_lstm(
        X_tr, y_tr, X_va, y_va,
        config=cfg_l1, n_features=len(feature_cols),
        seed=seed,
    )
    train_s = time.perf_counter() - t0

    # ────── Pre-gate (val_logloss / H(p_train)) ──────
    val_proba = predict_proba(model, X_va, batch_size=cfg_l1.batch_size * 4)
    val_ll = float(history.get("best_val_loss", float("nan")))
    H_p = class_prior_entropy(y_tr, n_classes=3)
    ratio = val_ll / H_p if H_p > 0 else float("nan")

    # ────── Platt calibrate on val raw → predict OOT ──────
    cal = fit_platt(val_proba, y_va, n_classes=3)
    oot_proba_raw = predict_proba(model, X_oo, batch_size=cfg_l1.batch_size * 4)
    oot_cal = apply_platt(oot_proba_raw, cal)

    # ────── Backtest ──────
    oot_bar_ids = df_sorted["bar_id"].values[kept_oo].astype("int64")
    preds = pd.DataFrame({
        "bar_id": oot_bar_ids,
        "p_long":   oot_cal[:, 0],
        "p_short":  oot_cal[:, 1],
        "p_neutral": oot_cal[:, 2],
    })
    # labels subset for OOT bars (need exit_bar_id, exit_price, exit_reason, holding_bars)
    oot_label_rows = oot[oot["bar_id"].isin(oot_bar_ids)].copy()
    labels_oot = oot_label_rows[["bar_id", "exit_bar_id", "exit_price",
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
    metrics = compute_metrics(trades, eq, oot_n_bars=n_oot)

    # Optional: backtest at multiple thresholds (no retrain)
    threshold_metrics: dict = {}
    if threshold_sweep is not None:
        for thr in threshold_sweep:
            tr = simulate_trades(
                predictions=preds,
                bars_df=bars_full[["bar_id", "bar_close_ts", "close"]],
                labels_df=labels_oot,
                confidence_threshold=thr,
                cost_bps_round_trip=cost_bps_round_trip,
                max_concurrent=1,
            )
            eqt = build_equity_curve(tr, starting_equity=10_000.0)
            mt = compute_metrics(tr, eqt, oot_n_bars=n_oot)
            mean_pnl = float(np.mean([t.pnl_bps_net for t in tr])) if tr else 0.0
            threshold_metrics[f"{thr:.2f}"] = {
                "n_trades": mt["oot_n_trades"],
                "sharpe": mt["oot_sharpe"],
                "win_pct": mt["oot_profitable_trade_pct"],
                "mean_pnl_bps_net": mean_pnl,
                "annual_return": mt["oot_annual_return"],
            }

    res.update({
        "val_logloss": val_ll,
        "H_p_train": float(H_p),
        "ratio": float(ratio),
        "pre_gate_pass": bool(ratio < 0.99) if not np.isnan(ratio) else False,
        "epochs_run": int(history.get("epochs_run", 0)),
        "train_seconds": train_s,
        **metrics,
    })
    if threshold_metrics:
        res["threshold_sweep"] = threshold_metrics
    LOG.info(
        "Fold %d (%s): ratio=%.4f pass=%s sharpe=%.3f n_trades=%d epochs=%d trainS=%.0fs",
        fold.fold_id, cfg_l1.name, ratio, res["pre_gate_pass"],
        metrics["oot_sharpe"], metrics["oot_n_trades"],
        res["epochs_run"], train_s,
    )
    return res


def run_config(cfg_l1: L1Config, first_n: Optional[int] = None,
               smoke_epochs: Optional[int] = None,
               asset: str = "BTC",
               threshold_sweep: Optional[list[float]] = None,
               tb_relabel: Optional[float] = None) -> dict:
    """Run all 18 folds for one L1 config."""
    cfg = _load_config()
    wf = cfg["walk_forward"]
    pg = cfg["pre_gate"]
    bt = cfg["backtest"]
    confidence_threshold = cfg["model"]["signal_threshold"]
    cost_bps_round_trip = bt["costs_bps_round_trip"]

    if smoke_epochs is not None:
        cfg_l1 = L1Config(**{**cfg_l1.__dict__, "max_epochs": smoke_epochs, "patience": smoke_epochs + 1})
        LOG.info("[SMOKE] forcing max_epochs=%d for config %s", smoke_epochs, cfg_l1.name)

    LOG.info("loading features + labels + bars (asset=%s)...", asset)
    feats = _load_features(asset)
    bars_full = _load_bars_ohlc(asset)

    if tb_relabel is not None:
        from labels.triple_barrier import apply_triple_barrier
        wf_cfg = _load_config()
        vert = wf_cfg["labeling"]["vertical_bars"]
        LOG.info("  in-memory relabel at TB=%.3f (vertical=%d)...", tb_relabel, vert)
        labels = apply_triple_barrier(
            bars_full[["bar_id", "bar_open_ts", "bar_close_ts", "close", "high", "low"]],
            tp_pct=tb_relabel, sl_pct=tb_relabel, vertical_bars=vert,
        )
    else:
        labels = _load_labels(asset)

    df_full = feats.merge(labels, on="bar_id", how="inner")
    df_full = df_full[df_full[LABEL_COL] != -1].copy()
    df_full["label"] = df_full["label"].astype("int64")
    df_full = df_full.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)
    LOG.info("  merged: %d labelable bars × %d cols", len(df_full), df_full.shape[1])

    feature_cols = [c for c in feats.columns if c not in KEY_COLS]
    LOG.info("  features: %d", len(feature_cols))

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
        LOG.info("  first_n=%d → running %d folds", first_n, len(folds))

    LOG.info("config %s: %d folds × max_epochs=%d", cfg_l1.name, len(folds), cfg_l1.max_epochs)

    fold_results: list[dict] = []
    t0_total = time.perf_counter()
    for fold in folds:
        try:
            res = run_fold_l1(
                fold=fold, df_full=df_full, bars_full=bars_full,
                feature_cols=feature_cols, cfg_l1=cfg_l1,
                purge_bars=wf["purge_bars"], embargo_bars=wf["embargo_bars"],
                confidence_threshold=confidence_threshold,
                cost_bps_round_trip=cost_bps_round_trip,
                threshold_sweep=threshold_sweep,
            )
        except Exception:
            LOG.exception("fold %d (%s) raised", fold.fold_id, cfg_l1.name)
            res = {"fold": fold.fold_id, "config": cfg_l1.name,
                   "skipped": True, "skip_reason": "exception (see log)"}
        fold_results.append(res)
    total_s = time.perf_counter() - t0_total

    evaluated = [r for r in fold_results if not r.get("skipped")]
    agg = {}
    pg_summary = {}
    if evaluated:
        for k in ("oot_sharpe", "oot_sortino", "oot_max_dd",
                  "oot_pct_time_in_market", "oot_n_trades",
                  "oot_profitable_trade_pct", "oot_annual_return", "ratio"):
            vals = [r[k] for r in evaluated if k in r and r[k] is not None]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ratios = [r["ratio"] for r in evaluated if "ratio" in r and not np.isnan(r["ratio"])]
        pg_summary = aggregate_pre_gate(
            ratios, threshold=pg["threshold"],
            required_pass=pg["required_pass_folds"], n_first=6,
        )

    out = {
        "config": {
            "name": cfg_l1.name,
            "conv_kernel": cfg_l1.conv_kernel,
            "conv_channels": cfg_l1.conv_channels,
            "lstm_hidden": cfg_l1.lstm_hidden,
            "lstm_layers": cfg_l1.lstm_layers,
            "dropout": cfg_l1.dropout,
            "learning_rate": cfg_l1.learning_rate,
            "batch_size": cfg_l1.batch_size,
            "max_epochs": cfg_l1.max_epochs,
        },
        "asset": asset,
        "seq_len": SEQ_LEN,
        "n_folds_total": len(folds),
        "n_folds_evaluated": len(evaluated),
        "n_folds_skipped": len(folds) - len(evaluated),
        "wall_clock_seconds": total_s,
        "pre_gate": pg_summary,
        "aggregate": agg,
        "per_fold": fold_results,
    }
    # Aggregate threshold sweep metrics if present
    if any("threshold_sweep" in r for r in evaluated):
        thr_agg: dict = {}
        thr_keys = sorted({k for r in evaluated for k in r.get("threshold_sweep", {})}, key=float)
        for thr in thr_keys:
            sharpes = [r["threshold_sweep"][thr]["sharpe"] for r in evaluated
                       if thr in r.get("threshold_sweep", {})]
            n_trades = [r["threshold_sweep"][thr]["n_trades"] for r in evaluated
                        if thr in r.get("threshold_sweep", {})]
            wins = [r["threshold_sweep"][thr]["win_pct"] for r in evaluated
                    if thr in r.get("threshold_sweep", {}) and r["threshold_sweep"][thr]["n_trades"] > 0]
            pnls = [r["threshold_sweep"][thr]["mean_pnl_bps_net"] for r in evaluated
                    if thr in r.get("threshold_sweep", {}) and r["threshold_sweep"][thr]["n_trades"] > 0]
            nz = [s for s in sharpes if s != 0.0]
            thr_agg[thr] = {
                "n_trades_total": int(sum(n_trades)),
                "sharpe_mean_all": float(np.mean(sharpes)) if sharpes else 0.0,
                "sharpe_mean_nonzero": float(np.mean(nz)) if nz else 0.0,
                "n_active_folds": sum(1 for n in n_trades if n > 0),
                "win_pct_mean": float(np.mean(wins)) if wins else 0.0,
                "mean_pnl_bps": float(np.mean(pnls)) if pnls else 0.0,
            }
        out["threshold_sweep_aggregate"] = thr_agg

    tb_suffix = f"_tb{int(tb_relabel * 1000):03d}" if tb_relabel is not None else ""
    out_path = PROJECT_ROOT / "reports" / "phase_1" / f"resnet_lstm_config_{cfg_l1.name}{tb_suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out["tb_relabel"] = tb_relabel
    out_path.write_text(json.dumps(out, indent=2, default=str))
    LOG.info("wrote %s  (%.1f min)", out_path, total_s / 60.0)
    return out


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="run_phase_1_resnet_lstm")
    p.add_argument("--asset", default="BTC")
    p.add_argument("--config", default=None,
                   help="Run only this config (A_small | B_medium | C_large | D_deep | E_wide_batch)")
    p.add_argument("--all-configs", action="store_true",
                   help="Run all 5 configs sequentially (mini-Hyperband).")
    p.add_argument("--first-n", type=int, default=None,
                   help="Smoke: run only the first N folds.")
    p.add_argument("--smoke-epochs", type=int, default=None,
                   help="Override max_epochs (e.g., 3 for smoke).")
    p.add_argument("--threshold-sweep", default=None,
                   help="Comma-separated thresholds to backtest at no extra training cost "
                        "(e.g., '0.55,0.58,0.60,0.62,0.65').")
    p.add_argument("--tb", type=float, default=None,
                   help="In-memory triple-barrier relabel at tp=sl=this value "
                        "(e.g., 0.03 to match DR v3.0.12 best config).")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    thr_sweep = None
    if args.threshold_sweep:
        thr_sweep = [float(x) for x in args.threshold_sweep.split(",")]

    try:
        if args.all_configs:
            results = []
            for cfg_l1 in L1_CONFIGS:
                out = run_config(cfg_l1, first_n=args.first_n,
                                  smoke_epochs=args.smoke_epochs, asset=args.asset,
                                  threshold_sweep=thr_sweep)
                results.append(out)
            # Mini-summary
            print("\n========== L1 Mini-Hyperband Summary ==========")
            print(f"{'config':<14} {'n_eval':>6} {'Shp_all':>8} {'Shp_std':>8} {'mPnL_bps':>9} {'trades':>7}")
            for r in results:
                agg = r["aggregate"]
                print(f"  {r['config']['name']:<12} {r['n_folds_evaluated']:>6} "
                      f"{agg.get('oot_sharpe_mean', 0):>+8.3f} {agg.get('oot_sharpe_std', 0):>8.3f} "
                      f"{agg.get('oot_n_trades_mean', 0)*100/max(r['n_folds_evaluated'],1):>9.1f} "
                      f"{int(agg.get('oot_n_trades_mean', 0) * r['n_folds_evaluated']):>7d}")
        elif args.config:
            cfg_l1 = next((c for c in L1_CONFIGS if c.name == args.config), None)
            if cfg_l1 is None:
                raise SystemExit(f"unknown config: {args.config}  (valid: {[c.name for c in L1_CONFIGS]})")
            run_config(cfg_l1, first_n=args.first_n,
                        smoke_epochs=args.smoke_epochs, asset=args.asset,
                        threshold_sweep=thr_sweep, tb_relabel=args.tb)
        else:
            raise SystemExit("specify --config NAME or --all-configs")
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
