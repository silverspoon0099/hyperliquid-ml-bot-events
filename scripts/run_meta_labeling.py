"""Meta-labeling on L0 baseline (DR v3.0.19).

Anchor: Lopez de Prado AFML §3.6.

Given L0's per-fold OOT primary signals (extracted via
`scripts.extract_l0_predictions`), trains a binary secondary model
to predict "given primary fires, will this trade win?" using
out-of-fold past predictions, then filters primary signals at multiple
meta-thresholds.

Walk-forward structure (no leakage):
  For each outer fold f ≥ MIN_TRAIN_FOLDS:
    - Train set: primary OOT signals from folds [1..f-1]
    - Test set:  primary OOT signals from fold f
    - Secondary inputs: [p_long, p_short, p_neutral, direction, 33 base features]
    - Secondary target: win (1 if pnl_bps_net > 0 else 0)
  Folds < MIN_TRAIN_FOLDS use primary alone (cold start).

Sweep:
  Primary thresholds: {0.62, 0.65}  (from extract step)
  Meta thresholds:    {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}

Output: `reports/phase_1/meta_labeling_results.json` + summary tables.

CLI:
    python -m scripts.run_meta_labeling
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.db import close_pool, symbol_short
from backtest.runner import Trade, build_equity_curve, compute_metrics

LOG = logging.getLogger("scripts.run_meta_labeling")

MIN_TRAIN_FOLDS = 4  # need ≥ 3 prior folds of data to train secondary
DEFAULT_META_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
# Total fold denominator for Sharpe(all 18 folds) — matches L0 joint sweep convention.
# 20 generated, 2 skipped (10, 18) → 18 evaluated.
N_EVALUATED_FOLDS = 18


def _load_predictions(thr: float) -> pd.DataFrame:
    thr_label = f"{int(thr * 100):03d}"
    path = PROJECT_ROOT / "reports" / "phase_1" / f"l0_predictions_thr{thr_label}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions parquet not found: {path}\n"
            f"Run: python -m scripts.extract_l0_predictions --tb 0.03 --thresholds {thr}"
        )
    return pd.read_parquet(path)


def _load_features(asset: str = "BTC") -> pd.DataFrame:
    sym = symbol_short(asset)
    return pd.read_parquet(PROJECT_ROOT / f"data/storage/features/features_{sym}.parquet")


def _train_secondary(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> lgb.Booster:
    """Train a binary LightGBM secondary (smaller than primary)."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return booster


def _rows_to_trades(rows: pd.DataFrame) -> list[Trade]:
    """Reconstruct list of Trade objects from a DataFrame of saved primary signals."""
    trades = []
    for _, r in rows.iterrows():
        trades.append(Trade(
            bar_id=int(r["bar_id"]),
            entry_ts=pd.Timestamp(r["entry_ts"]),
            exit_ts=pd.Timestamp(r["exit_ts"]),
            entry_price=float(r["entry_price"]),
            exit_price=float(r["exit_price"]),
            direction=int(r["direction"]),
            exit_reason=str(r["exit_reason"]),
            holding_bars=int(r["holding_bars"]),
            pnl_bps_gross=float(r["pnl_bps_gross"]),
            pnl_bps_net=float(r["pnl_bps_net"]),
        ))
    return trades


def _per_fold_backtest_metrics(rows: pd.DataFrame) -> dict:
    """Run backtest.runner.compute_metrics over filtered trade rows.

    Aggregates per-fold (daily-resampled Sharpe per fold) and overall (combined).
    Mirrors the L0 joint sweep aggregate convention so the Sharpe is
    directly comparable to baseline +0.721.
    """
    fold_metrics = []
    fold_sharpes = []
    if len(rows) == 0:
        return {
            "n_folds_with_data": 0, "n_active_folds": 0, "n_trades_total": 0,
            "win_pct_overall": 0.0, "mean_pnl_bps_net": 0.0,
            "sharpe_mean_across_folds": 0.0, "sharpe_mean_nonzero": 0.0,
            "total_pnl_bps": 0.0, "per_fold": [],
        }
    for f, group in rows.groupby("fold_id"):
        trades_f = _rows_to_trades(group)
        eq = build_equity_curve(trades_f, starting_equity=10_000.0)
        # oot_n_bars is needed by compute_metrics for pct_time_in_market;
        # use group length as a proxy (not strictly accurate but bounded)
        m = compute_metrics(trades_f, eq, oot_n_bars=max(len(group), 100))
        sharpe = float(m.get("oot_sharpe", 0.0))
        fold_metrics.append({
            "fold_id": int(f),
            "n_trades": int(len(group)),
            "win_pct": float(m.get("oot_profitable_trade_pct", 0.0)),
            "mean_pnl_bps_net": float(group["pnl_bps_net"].mean()),
            "sharpe": sharpe,
        })
        fold_sharpes.append(sharpe)
    nz = [s for s in fold_sharpes if abs(s) > 1e-9]
    n_active = sum(1 for m in fold_metrics if m["n_trades"] > 0)
    # Sharpe(all 18 folds) = sum of fold Sharpes (zero for inactive) / N_EVALUATED_FOLDS.
    # This matches the L0 joint sweep convention so we can compare directly to
    # the L0 baseline numbers (+0.657 at thr=0.62, +0.721 at thr=0.65).
    sharpe_all_18 = sum(fold_sharpes) / N_EVALUATED_FOLDS if fold_sharpes else 0.0
    return {
        "n_folds_with_data": len(fold_metrics),
        "n_active_folds": n_active,
        "n_trades_total": int(len(rows)),
        "win_pct_overall": 100.0 * rows["win"].sum() / max(len(rows), 1),
        "mean_pnl_bps_net": float(rows["pnl_bps_net"].mean()),
        "sharpe_mean_across_folds": float(sharpe_all_18),
        "sharpe_mean_active_folds": float(np.mean(fold_sharpes)) if fold_sharpes else 0.0,
        "sharpe_mean_nonzero": float(np.mean(nz)) if nz else 0.0,
        "total_pnl_bps": float(rows["pnl_bps_net"].sum()),
        "per_fold": fold_metrics,
    }


def run_meta_labeling(
    primary_thresholds: tuple = (0.62, 0.65),
    meta_thresholds: tuple = DEFAULT_META_THRESHOLDS,
    asset: str = "BTC",
) -> dict:
    """Run meta-labeling sweep across primary × meta thresholds."""
    LOG.info("loading base features (asset=%s)...", asset)
    feats = _load_features(asset)
    feat_cols = [c for c in feats.columns if c not in ("bar_id", "bar_close_ts")]
    LOG.info("base feature count: %d", len(feat_cols))

    all_results: dict = {}

    for primary_thr in primary_thresholds:
        LOG.info("=== primary threshold = %.2f ===", primary_thr)
        preds_df = _load_predictions(primary_thr)
        LOG.info("  loaded %d primary signals (folds: %s)",
                 len(preds_df), sorted(preds_df["fold_id"].unique().tolist()))

        # Merge base features so the secondary can see them
        merged = preds_df.merge(feats[["bar_id"] + feat_cols], on="bar_id", how="left")
        if merged[feat_cols].isna().any().any():
            n_with_nan = int(merged[feat_cols].isna().any(axis=1).sum())
            LOG.warning("  %d rows have NaN in base features (will be dropped)", n_with_nan)
            merged = merged.dropna(subset=feat_cols).reset_index(drop=True)

        # Secondary features: primary probas + direction + base features
        secondary_cols = ["p_long", "p_short", "p_neutral", "direction"] + feat_cols

        # Baseline metrics (primary alone, no filter)
        baseline_metrics = _per_fold_backtest_metrics(merged)

        # Meta-label scoring per fold
        # For each fold f ≥ MIN_TRAIN_FOLDS, train on prior folds, predict on fold f
        all_fold_ids = sorted(merged["fold_id"].unique().tolist())
        meta_scored_rows: list[dict] = []  # one row per primary signal, with meta_proba

        # Cold-start: folds < MIN_TRAIN_FOLDS use primary alone (meta_proba = 1.0 → always trade)
        cold = merged[merged["fold_id"] < MIN_TRAIN_FOLDS].copy()
        cold["meta_proba"] = 1.0  # always trade in cold-start period
        meta_scored_rows.extend(cold.to_dict("records"))

        for f in all_fold_ids:
            if f < MIN_TRAIN_FOLDS:
                continue
            train_mask = merged["fold_id"] < f
            test_mask = merged["fold_id"] == f
            train_df = merged.loc[train_mask]
            test_df = merged.loc[test_mask].copy()

            if len(train_df) < 30 or len(test_df) == 0:
                # Too little training data — fall back to primary alone
                test_df["meta_proba"] = 1.0
                meta_scored_rows.extend(test_df.to_dict("records"))
                LOG.info("  fold %d: cold (n_train=%d, n_test=%d)", f, len(train_df), len(test_df))
                continue

            # Split train into 80/20 for early stopping
            split = int(len(train_df) * 0.8)
            train_inner = train_df.iloc[:split]
            val_inner = train_df.iloc[split:]
            if len(val_inner) < 10:
                # too few for val — just train without early stop
                val_inner = train_inner.tail(max(10, int(len(train_inner) * 0.2)))
                train_inner = train_inner.iloc[:-len(val_inner)] if len(val_inner) < len(train_inner) else train_inner

            X_tr = train_inner[secondary_cols].values
            y_tr = train_inner["win"].astype(int).values
            X_va = val_inner[secondary_cols].values
            y_va = val_inner["win"].astype(int).values

            booster = _train_secondary(X_tr, y_tr, X_va, y_va)

            X_test = test_df[secondary_cols].values
            meta_proba = booster.predict(X_test)
            test_df["meta_proba"] = meta_proba

            meta_scored_rows.extend(test_df.to_dict("records"))
            LOG.info("  fold %2d: trained on %d rows (val_acc≈%s) → predicted %d test signals",
                     f, len(train_df), "n/a", len(test_df))

        scored = pd.DataFrame(meta_scored_rows)
        # Per-meta-threshold metrics
        results_by_meta = {}
        for meta_thr in meta_thresholds:
            filtered = scored[scored["meta_proba"] > meta_thr]
            m = _per_fold_backtest_metrics(filtered)
            results_by_meta[f"{meta_thr:.2f}"] = m

        all_results[f"{primary_thr:.2f}"] = {
            "baseline_primary_alone": baseline_metrics,
            "by_meta_threshold": results_by_meta,
        }

    out = {
        "asset": asset,
        "primary_thresholds": list(primary_thresholds),
        "meta_thresholds": list(meta_thresholds),
        "results": all_results,
    }
    out_path = PROJECT_ROOT / "reports" / "phase_1" / "meta_labeling_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    LOG.info("wrote %s", out_path)
    return out


def print_report(out: dict) -> None:
    print("\n========== Meta-Labeling Report (DR v3.0.19) ==========")
    print(f"Asset: {out['asset']}")
    print(f"Primary thresholds: {out['primary_thresholds']}")
    print(f"Meta thresholds:    {out['meta_thresholds']}\n")

    for primary_thr in out["primary_thresholds"]:
        pkey = f"{primary_thr:.2f}"
        block = out["results"][pkey]
        base = block["baseline_primary_alone"]
        print(f"--- Primary threshold = {primary_thr:.2f} ---")
        print(f"  Baseline (no meta): n_trades={base['n_trades_total']}  active={base['n_active_folds']}/{base['n_folds_with_data']}  "
              f"win%={base['win_pct_overall']:.1f}  mPnL={base['mean_pnl_bps_net']:+.1f}  "
              f"Sharpe(all_folds)={base['sharpe_mean_across_folds']:+.3f}  "
              f"Sharpe(!=0)={base['sharpe_mean_nonzero']:+.3f}")
        print()
        print(f"  {'meta_thr':>9}  {'n_trd':>6}  {'kept%':>6}  {'active':>6}  "
              f"{'win%':>6}  {'mPnL':>9}  {'Sharpe(all)':>11}  {'Sharpe(!=0)':>11}  {'lift_all':>9}")
        print("  " + "-" * 92)
        base_shp_all = base["sharpe_mean_across_folds"]
        for meta_thr in out["meta_thresholds"]:
            mkey = f"{meta_thr:.2f}"
            m = block["by_meta_threshold"][mkey]
            kept_pct = 100.0 * m["n_trades_total"] / max(base["n_trades_total"], 1)
            lift = m["sharpe_mean_across_folds"] - base_shp_all
            print(f"  {meta_thr:>9.2f}  {m['n_trades_total']:>6}  {kept_pct:>5.1f}%  "
                  f"{m['n_active_folds']:>2}/{m['n_folds_with_data']:<3}  "
                  f"{m['win_pct_overall']:>6.1f}  {m['mean_pnl_bps_net']:>+9.1f}  "
                  f"{m['sharpe_mean_across_folds']:>+11.3f}  "
                  f"{m['sharpe_mean_nonzero']:>+11.3f}  "
                  f"{lift:>+9.3f}")
        print()
    print("--- Interpretation ---")
    print("  Sharpe uses backtest.runner's daily-resample convention — DIRECTLY")
    print("  comparable to L0 baseline +0.721 (thr=0.65) / +0.657 (thr=0.62) at TB=0.03/11bps.")
    print("  Look for: positive 'lift_all' column AND maintained kept% (>40%).")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="run_meta_labeling")
    p.add_argument("--asset", default="BTC")
    p.add_argument("--primary-thr", default="0.62,0.65")
    p.add_argument("--meta-thr", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    primary_ts = tuple(float(x) for x in args.primary_thr.split(","))
    meta_ts = tuple(float(x) for x in args.meta_thr.split(","))

    try:
        out = run_meta_labeling(primary_thresholds=primary_ts,
                                meta_thresholds=meta_ts, asset=args.asset)
        print_report(out)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
