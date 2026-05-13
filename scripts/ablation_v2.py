"""v2 leave-one-feature-out ablation (DR v3.0.16).

For each of the 8 new features in features_btc_v2.parquet, runs the
L0 walk-forward EXCLUDING that one feature, and compares aggregate
Sharpe + n_trades to the full-v2 baseline.

Also computes:
  - baseline_33: just the 33 Lessmann features (drops all 8 new)
  - full_v2: all 41 features

Variants run: 1 (baseline) + 1 (full) + 8 (leave-one-out) = 10 walk-forwards.
Each walk-forward = ~2 min. Total = ~20 min.

Output: reports/phase_1/ablation_v2.json
"""
from __future__ import annotations

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
from cv.pre_gate import class_prior_entropy, pre_gate_ratio
from model.lgbm import train_lgbm, fit_platt, apply_platt
from backtest.runner import simulate_trades, build_equity_curve, compute_metrics
from data.db import close_pool, get_engine, bars_table, symbol_short
from features.v2_builder import V2_ORDERFLOW, V2_HTF, V2_NEW

LOG = logging.getLogger("scripts.ablation_v2")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
KEY_COLS = ["bar_id", "bar_close_ts"]
LABEL_COL = "label"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_features_v2(asset: str = "BTC") -> pd.DataFrame:
    sym = symbol_short(asset)
    path = PROJECT_ROOT / f"data/storage/features/features_{sym}_v2.parquet"
    return pd.read_parquet(path)


def _load_labels(asset: str = "BTC") -> pd.DataFrame:
    sym = symbol_short(asset)
    return pd.read_parquet(PROJECT_ROOT / f"data/storage/labels/labels_{sym}.parquet")


def _load_bars_ohlc(asset: str = "BTC") -> pd.DataFrame:
    sql = (f"SELECT bar_id, bar_close_ts, open, high, low, close "
           f"FROM {bars_table(asset)} WHERE threshold_pct=0.02 ORDER BY bar_close_ts, bar_id")
    return pd.read_sql_query(sql, get_engine())


def run_walkforward_variant(
    df_full: pd.DataFrame,
    feature_cols: list[str],
    bars_close: pd.DataFrame,
    cfg: dict,
    variant_name: str,
) -> dict:
    """Run the L0 walk-forward with a custom feature subset, return summary."""
    wf = cfg["walk_forward"]
    bt = cfg["backtest"]
    lgbm_params = cfg["model"]["L0_lightgbm"]
    confidence_threshold = cfg["model"]["signal_threshold"]
    cost_bps_round_trip = bt["costs_bps_round_trip"]

    bars_full = bars_close
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

    fold_rows = []
    t0 = time.perf_counter()
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
        val_log = float(np.mean(-np.log(np.clip(
            val_raw[np.arange(len(val)), val[LABEL_COL].astype(int).values], 1e-15, 1.0
        ))))
        H_p = class_prior_entropy(train[LABEL_COL].astype(int).values, n_classes=3)
        ratio = val_log / H_p if H_p > 0 else 1.0

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

        fold_rows.append({
            "fold": fold.fold_id,
            "n_oot": len(oot),
            "ratio": ratio,
            "n_trades": metrics["oot_n_trades"],
            "mean_pnl_bps_net": mean_pnl,
            "win_pct": metrics["oot_profitable_trade_pct"],
            "sharpe": metrics["oot_sharpe"],
            "annual_return": metrics["oot_annual_return"],
        })

    sharpes = [r["sharpe"] for r in fold_rows]
    n_trades = [r["n_trades"] for r in fold_rows]
    nonzero = [s for s in sharpes if s != 0.0]
    pnls = [r["mean_pnl_bps_net"] for r in fold_rows if r["n_trades"] > 0]

    summary = {
        "variant": variant_name,
        "n_features": len(feature_cols),
        "n_folds": len(fold_rows),
        "n_active_folds": sum(1 for n in n_trades if n > 0),
        "total_trades": int(sum(n_trades)),
        "sharpe_all": float(np.mean(sharpes)) if sharpes else 0.0,
        "sharpe_nonzero": float(np.mean(nonzero)) if nonzero else 0.0,
        "sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
        "mean_pnl_bps": float(np.mean(pnls)) if pnls else 0.0,
        "annret_mean": float(np.mean([r["annual_return"] for r in fold_rows])) if fold_rows else 0.0,
        "wall_clock": time.perf_counter() - t0,
        "per_fold": fold_rows,
    }
    LOG.info("  %s: n_feat=%d  Shp_all=%+.3f  Shp_!=0=%+.3f  trades=%d  active=%d/%d  (%.1fs)",
             variant_name, len(feature_cols),
             summary["sharpe_all"], summary["sharpe_nonzero"],
             summary["total_trades"], summary["n_active_folds"], summary["n_folds"],
             summary["wall_clock"])
    return summary


def run_ablation(asset: str = "BTC") -> dict:
    cfg = _load_config()

    LOG.info("loading v2 features + labels + bars (asset=%s)...", asset)
    feats = _load_features_v2(asset)
    labels = _load_labels(asset)
    bars_close = _load_bars_ohlc(asset)

    df_full = feats.merge(labels, on="bar_id", how="inner")
    df_full = df_full[df_full[LABEL_COL] != -1].copy()
    df_full["label"] = df_full["label"].astype("int64")
    df_full = df_full.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)

    # Feature column inventory
    all_cols = [c for c in feats.columns if c not in KEY_COLS]  # 41
    new_cols = V2_NEW                                            # 8
    base_cols = [c for c in all_cols if c not in new_cols]      # 33

    LOG.info("inventory: %d total feat cols (%d base + %d new)",
             len(all_cols), len(base_cols), len(new_cols))

    variants = []

    # 1) baseline 33-feature
    variants.append(("baseline_33", base_cols))
    # 2) full v2 41-feature
    variants.append(("full_v2", all_cols))
    # 3) leave-one-out × 8
    for c in new_cols:
        variants.append((f"minus_{c}", [col for col in all_cols if col != c]))

    LOG.info("running %d variants...", len(variants))
    results = []
    for vname, vcols in variants:
        s = run_walkforward_variant(df_full, vcols, bars_close, cfg, vname)
        results.append(s)

    out = {
        "asset": asset,
        "n_variants": len(variants),
        "variants": results,
    }
    out_path = PROJECT_ROOT / "reports" / "phase_1" / "ablation_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    LOG.info("wrote %s", out_path)
    return out


def print_ablation_report(out: dict) -> None:
    print(f"\n========== v2 Ablation Report (DR v3.0.16) ==========")
    print(f"Asset: {out['asset']}  Variants: {out['n_variants']}\n")

    baseline = next((v for v in out["variants"] if v["variant"] == "baseline_33"), None)
    full = next((v for v in out["variants"] if v["variant"] == "full_v2"), None)

    print(f"{'variant':<32}  {'n_feat':>6}  {'Shp_all':>8}  {'Shp_!=0':>8}  "
          f"{'trades':>7}  {'active':>7}  {'mPnL':>8}  {'annret':>8}")
    print("-" * 100)
    for v in out["variants"]:
        print(f"  {v['variant']:<30}  {v['n_features']:>6}  "
              f"{v['sharpe_all']:>+8.3f}  {v['sharpe_nonzero']:>+8.3f}  "
              f"{v['total_trades']:>7d}  {v['n_active_folds']:>3d}/{v['n_folds']:<3d}  "
              f"{v['mean_pnl_bps']:>+8.2f}  {v['annret_mean']:>+8.3f}")

    if baseline and full:
        delta_all = full["sharpe_all"] - baseline["sharpe_all"]
        delta_nz = full["sharpe_nonzero"] - baseline["sharpe_nonzero"]
        print(f"\n--- v2 lift vs baseline ---")
        print(f"  ΔSharpe(all):    {delta_all:+.3f}")
        print(f"  ΔSharpe(!=0):    {delta_nz:+.3f}")
        print(f"  Δtrades:         {full['total_trades'] - baseline['total_trades']:+d}")

        print(f"\n--- Per-feature contribution (full - leave_out) ---")
        print(f"  feature                       contribution_to_Shp_all  contribution_to_Shp_!=0")
        for v in out["variants"]:
            if v["variant"].startswith("minus_"):
                feat = v["variant"].replace("minus_", "")
                c_all = full["sharpe_all"] - v["sharpe_all"]
                c_nz = full["sharpe_nonzero"] - v["sharpe_nonzero"]
                marker = "*" if abs(c_all) > 0.05 else " "
                print(f"  {marker} {feat:<28}  {c_all:>+8.3f}                  {c_nz:>+8.3f}")
    print()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        out = run_ablation(asset="BTC")
        print_ablation_report(out)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
