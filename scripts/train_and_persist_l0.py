"""Train L0 LightGBM on full historical data, persist as artifact (DR v3.0.23).

For paper/live deployment we need a fixed model snapshot. Unlike the walk-
forward CV which trains 18 models, this script trains ONE model on the
full labeled history at the champion bar threshold (1.5%) and TB=0.03.

The model uses the same LightGBM hyperparams as L0 walk-forward (per
config.yaml model.L0_lightgbm). Platt calibration is fit on a held-out
tail of the data (last 20% by time).

Artifact format (joblib pickle):
    {
        "booster_bytes": bytes (LightGBM model serialized),
        "platt": dict (3 sigmoid calibrators),
        "feature_columns": list[str] (column names + order, locked),
        "label_classes": [0, 1, 2],  # LONG, SHORT, NEUTRAL
        "training_metadata": {
            "trained_at": iso str,
            "bar_threshold": 0.015,
            "tb_pct": 0.03,
            "vertical_bars": 24,
            "n_train": int, "n_val": int,
            "data_min_ts": iso, "data_max_ts": iso,
            "val_logloss": float,
            "H_p_train": float,
            "ratio": float,
            "n_trees": int,
            "lgbm_params": dict,
        }
    }

Saved to: data/storage/models/l0_btc_thr015_v{N}.pkl
where N is auto-incremented.

CLI:
    python -m scripts.train_and_persist_l0 --asset BTC --bar-threshold 0.015 --tb 0.03
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cv.pre_gate import class_prior_entropy
from model.lgbm import train_lgbm, fit_platt, apply_platt
from data.db import close_pool, bars_table, symbol_short, get_engine

LOG = logging.getLogger("scripts.train_and_persist_l0")
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


def _next_artifact_version(model_dir: Path, base_name: str) -> int:
    """Find the next available version number for base_name."""
    existing = list(model_dir.glob(f"{base_name}_v*.pkl"))
    if not existing:
        return 1
    versions = []
    for p in existing:
        try:
            v = int(p.stem.split("_v")[-1])
            versions.append(v)
        except (ValueError, IndexError):
            continue
    return max(versions, default=0) + 1


def train_and_persist(
    asset: str = "BTC",
    bar_threshold: float = 0.015,
    tb_pct: float = 0.03,
    val_frac: float = 0.20,
) -> dict:
    """Train L0 on full history, fit Platt on tail, save artifact."""
    from labels.triple_barrier import apply_triple_barrier

    cfg = _load_config()
    lgbm_params = cfg["model"]["L0_lightgbm"]
    vertical_bars = cfg["labeling"]["vertical_bars"]

    sym = symbol_short(asset)
    feats_path = PROJECT_ROOT / f"data/storage/features/features_{sym}_thr{int(bar_threshold*1000):03d}.parquet"
    LOG.info("loading features from %s", feats_path)
    feats = _load_features(str(feats_path))

    LOG.info("loading bars (asset=%s, bar_thr=%s)...", asset, bar_threshold)
    bars_full = _load_bars_ohlc(asset, bar_threshold)
    feature_cols = [c for c in feats.columns if c not in KEY_COLS]
    LOG.info("loaded: %d features (%d cols), %d bars", len(feats), len(feature_cols), len(bars_full))

    LOG.info("relabeling at TB=tp=sl=%.3f, vertical=%d bars...", tb_pct, vertical_bars)
    labels_df = apply_triple_barrier(
        bars_full[["bar_id", "bar_open_ts", "bar_close_ts", "close", "high", "low"]],
        tp_pct=tb_pct, sl_pct=tb_pct, vertical_bars=vertical_bars,
    )
    df_full = feats.merge(labels_df, on="bar_id", how="inner")
    df_full = df_full[df_full[LABEL_COL] != -1].copy()
    df_full["label"] = df_full["label"].astype("int64")
    df_full = df_full.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)
    LOG.info("merged: %d labelable bars", len(df_full))

    # Split: last val_frac by time → val, rest → train
    split_idx = int(len(df_full) * (1 - val_frac))
    train_df = df_full.iloc[:split_idx]
    val_df = df_full.iloc[split_idx:]
    LOG.info("split: train=%d (%s..%s), val=%d (%s..%s)",
             len(train_df), train_df["bar_close_ts"].iloc[0], train_df["bar_close_ts"].iloc[-1],
             len(val_df), val_df["bar_close_ts"].iloc[0], val_df["bar_close_ts"].iloc[-1])

    # Train
    LOG.info("training LightGBM (params: num_leaves=%d, lr=%.3f, num_boost_round=%d)...",
             lgbm_params["num_leaves"], lgbm_params["learning_rate"], lgbm_params["num_boost_round"])
    t0 = time.perf_counter()
    booster = train_lgbm(
        train_df[feature_cols], train_df[LABEL_COL].astype(int).values,
        val_df[feature_cols], val_df[LABEL_COL].astype(int).values,
        dict(lgbm_params),
    )
    train_s = time.perf_counter() - t0
    LOG.info("trained in %.1fs (best_iteration=%d)", train_s, booster.best_iteration or booster.num_trees())

    # Val metrics
    val_raw = booster.predict(val_df[feature_cols])
    val_ll = float(np.mean(-np.log(np.clip(
        val_raw[np.arange(len(val_df)), val_df[LABEL_COL].astype(int).values], 1e-15, 1.0
    ))))
    H_p = class_prior_entropy(train_df[LABEL_COL].astype(int).values, n_classes=3)
    ratio = val_ll / H_p
    LOG.info("val_logloss=%.4f, H(p_train)=%.4f, ratio=%.4f (pre-gate %s)",
             val_ll, H_p, ratio, "PASS" if ratio < 0.99 else "FAIL")

    # Platt calibration on val
    platt = fit_platt(val_raw, val_df[LABEL_COL].astype(int).values, n_classes=3)
    LOG.info("Platt calibration fit on %d val examples", len(val_df))

    # Serialize artifact
    booster_bytes = booster.model_to_string().encode("utf-8")
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "asset": asset,
        "bar_threshold": float(bar_threshold),
        "tb_pct": float(tb_pct),
        "vertical_bars": int(vertical_bars),
        "val_frac": float(val_frac),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "data_min_ts": str(df_full["bar_close_ts"].iloc[0]),
        "data_max_ts": str(df_full["bar_close_ts"].iloc[-1]),
        "train_min_ts": str(train_df["bar_close_ts"].iloc[0]),
        "train_max_ts": str(train_df["bar_close_ts"].iloc[-1]),
        "val_min_ts": str(val_df["bar_close_ts"].iloc[0]),
        "val_max_ts": str(val_df["bar_close_ts"].iloc[-1]),
        "val_logloss": float(val_ll),
        "H_p_train": float(H_p),
        "ratio": float(ratio),
        "pre_gate_pass": bool(ratio < 0.99),
        "n_trees": int(booster.best_iteration or booster.num_trees()),
        "lgbm_params": dict(lgbm_params),
        "feature_columns": list(feature_cols),
        "label_classes": [0, 1, 2],
        "train_seconds": float(train_s),
    }

    artifact = {
        "booster_bytes": booster_bytes,
        "platt": platt,
        "feature_columns": list(feature_cols),
        "label_classes": [0, 1, 2],
        "training_metadata": metadata,
    }

    model_dir = PROJECT_ROOT / "data" / "storage" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"l0_{sym.lower()}_thr{int(bar_threshold*1000):03d}"
    version = _next_artifact_version(model_dir, base_name)
    artifact_path = model_dir / f"{base_name}_v{version}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOG.info("wrote artifact: %s (%.2f MB)", artifact_path, artifact_path.stat().st_size / 1e6)

    # Also write metadata as separate JSON for easy inspection
    meta_path = model_dir / f"{base_name}_v{version}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    LOG.info("wrote metadata:  %s", meta_path)

    return {
        "artifact_path": str(artifact_path),
        "metadata_path": str(meta_path),
        "metadata": metadata,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="train_and_persist_l0")
    p.add_argument("--asset", default="BTC")
    p.add_argument("--bar-threshold", type=float, default=0.015,
                   help="Champion baseline: 1.5% bars")
    p.add_argument("--tb", type=float, default=0.03,
                   help="Triple-barrier TP/SL (default 0.03 from joint sweep best)")
    p.add_argument("--val-frac", type=float, default=0.20,
                   help="Tail fraction for Platt calibration (default 0.20)")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        result = train_and_persist(
            asset=args.asset,
            bar_threshold=args.bar_threshold,
            tb_pct=args.tb,
            val_frac=args.val_frac,
        )
        m = result["metadata"]
        print()
        print("=" * 70)
        print(f"L0 model artifact persisted: {result['artifact_path']}")
        print(f"Metadata: {result['metadata_path']}")
        print()
        print(f"Data range:    {m['data_min_ts']} → {m['data_max_ts']}")
        print(f"Train:         {m['n_train']} bars ({m['train_min_ts']} → {m['train_max_ts']})")
        print(f"Val:           {m['n_val']} bars ({m['val_min_ts']} → {m['val_max_ts']})")
        print(f"Val logloss:   {m['val_logloss']:.4f}")
        print(f"H(p_train):    {m['H_p_train']:.4f}")
        print(f"Pre-gate ratio: {m['ratio']:.4f}  ({'PASS' if m['pre_gate_pass'] else 'FAIL'})")
        print(f"Trees used:    {m['n_trees']}")
        print(f"Bar threshold: {m['bar_threshold']}")
        print(f"TB pct:        {m['tb_pct']}")
        print("=" * 70)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
