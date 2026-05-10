"""Regime-segmented Sharpe diagnostic — DR v3.0.14 §3.

Segments the 20 walk-forward folds into 3 market eras and reports
per-era Sharpe + activity, so we can apply the §5 decision tree:
  - Aggregate >=1.0 AND Era 3 >=1.0 -> GO L1
  - Aggregate >=1.0 BUT Era 3 <=0.5 -> ship 3b ETH historical
  - Aggregate <=0.5 -> ship 3b BTC tier1 baseline

Eras (per user research, oot_end basis):
  Era 1 (folds 1-6):   2021-07 .. 2022-10  Lessmann era / late bull + 2022 bear
  Era 2 (folds 7-14):  2023-01 .. 2024-10  FTX recovery, halving, ETF approval
  Era 3 (folds 15-20): 2025-01 .. 2026-04  post-halving, ETH ETF era, recent regime

Usage:
  python -m scripts.regime_segmented_sharpe --asset ETH
  python -m scripts.regime_segmented_sharpe --asset BTC --source l0
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase_1"

ERA_BOUNDS = [
    ("Era 1 (Lessmann)",     list(range(1, 7))),    # folds 1-6
    ("Era 2 (recovery)",     list(range(7, 15))),   # folds 7-14
    ("Era 3 (recent)",       list(range(15, 21))),  # folds 15-20
]


def _agg(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _segment_l0(per_fold: list[dict]) -> dict:
    """Segment default L0 (thr=0.60) by era."""
    out = {}
    for era_name, fold_ids in ERA_BOUNDS:
        sub = [f for f in per_fold if f["fold"] in fold_ids]
        sharpes = [f["oot_sharpe"] for f in sub]
        n_trades = [f["oot_n_trades"] for f in sub]
        active = sum(1 for n in n_trades if n > 0)
        out[era_name] = {
            "fold_ids": fold_ids,
            "n_folds": len(sub),
            "active_folds": active,
            "sharpe_mean": statistics.mean(sharpes) if sharpes else 0.0,
            "sharpe_std": statistics.stdev(sharpes) if len(sharpes) > 1 else 0.0,
            "total_trades": sum(n_trades),
            "annret_mean": statistics.mean([f["oot_annual_return"] for f in sub]) if sub else 0.0,
            "fold_sharpes": sharpes,
            "fold_n_trades": n_trades,
        }
    return out


def _segment_joint(by_threshold: dict) -> dict:
    """Segment joint TB×threshold sweep by era, per threshold."""
    out = {}
    for thr_key, payload in by_threshold.items():
        per_fold = payload["per_fold"]
        era_breakdown = {}
        for era_name, fold_ids in ERA_BOUNDS:
            sub = [f for f in per_fold if f["fold"] in fold_ids]
            sharpes = [f["sharpe"] for f in sub]
            n_trades = [f["n_trades"] for f in sub]
            active = sum(1 for n in n_trades if n > 0)
            era_breakdown[era_name] = {
                "n_folds": len(sub),
                "active_folds": active,
                "sharpe_mean": statistics.mean(sharpes) if sharpes else 0.0,
                "sharpe_std": statistics.stdev(sharpes) if len(sharpes) > 1 else 0.0,
                "total_trades": sum(n_trades),
                "annret_mean": statistics.mean([f["annual_return"] for f in sub]) if sub else 0.0,
            }
        out[thr_key] = era_breakdown
    return out


def _print_l0_table(eras: dict, asset: str) -> None:
    print(f"\n========== L0 default (thr=0.60) — {asset} regime breakdown ==========\n")
    print(f"  {'Era':<20}  {'folds':>6}  {'active':>6}  {'Sharpe (mean ± std)':>22}  {'trades':>7}  {'annret':>8}")
    print(f"  {'-' * 20}  {'-' * 6}  {'-' * 6}  {'-' * 22}  {'-' * 7}  {'-' * 8}")
    for era_name, m in eras.items():
        sm = m["sharpe_mean"]
        ss = m["sharpe_std"]
        print(f"  {era_name:<20}  {m['n_folds']:>6}  {m['active_folds']:>6}  "
              f"{sm:>+8.3f} ± {ss:>8.3f}    {m['total_trades']:>7}  {m['annret_mean']:>+8.3f}")
    print()
    for era_name, m in eras.items():
        if m["fold_sharpes"]:
            sharpes_str = "  ".join(f"{s:>+6.2f}" for s in m["fold_sharpes"])
            trades_str = "  ".join(f"{t:>6d}" for t in m["fold_n_trades"])
            print(f"  {era_name:<20}  Sharpes:  {sharpes_str}")
            print(f"  {' ':<20}  Trades :  {trades_str}")


def _print_joint_table(joint_eras: dict, asset: str) -> None:
    print(f"\n========== Joint TB=0.03 × threshold sweep — {asset} regime breakdown ==========\n")
    thresholds = sorted(joint_eras.keys(), key=float)
    print(f"  {'thr':>5}  ", end="")
    for era_name, _ in ERA_BOUNDS:
        print(f"{era_name:<22} ", end="")
    print()
    print(f"  {'-' * 5}  ", end="")
    for _ in ERA_BOUNDS:
        print(f"{'-' * 22} ", end="")
    print()
    for thr in thresholds:
        print(f"  {thr:>5}  ", end="")
        eras = joint_eras[thr]
        for era_name, _ in ERA_BOUNDS:
            m = eras[era_name]
            print(f"Shp={m['sharpe_mean']:>+5.2f} act={m['active_folds']}/{m['n_folds']} tr={m['total_trades']:<4d} ", end="")
        print()


def _decision_summary(asset: str, agg_l0: dict, l0_eras: dict,
                     joint_aggregates: dict, joint_eras: dict) -> None:
    print(f"\n========== DR v3.0.14 §5 Decision Tree — {asset} ==========\n")

    best_thr = None
    best_thr_agg_sharpe = -999.0
    for thr_key, payload in joint_aggregates.items():
        s = payload["aggregate"]["sharpe_mean_across_folds"]
        if s > best_thr_agg_sharpe:
            best_thr_agg_sharpe = s
            best_thr = thr_key

    l0_agg = agg_l0["oot_sharpe_mean"]
    era3_l0 = l0_eras["Era 3 (recent)"]["sharpe_mean"]
    era3_joint_best = joint_eras[best_thr]["Era 3 (recent)"]["sharpe_mean"]

    print(f"  L0 default (thr=0.60):           aggregate Sharpe = {l0_agg:+.3f}")
    print(f"                                     Era 3 Sharpe  = {era3_l0:+.3f}")
    print(f"  Joint sweep best (thr={best_thr}):   aggregate Sharpe = {best_thr_agg_sharpe:+.3f}")
    print(f"                                     Era 3 Sharpe  = {era3_joint_best:+.3f}")
    print()

    best_agg = max(l0_agg, best_thr_agg_sharpe)
    best_era3 = era3_joint_best if best_thr_agg_sharpe > l0_agg else era3_l0

    if best_agg >= 1.0 and best_era3 >= 1.0:
        verdict = "→ GO L1 (ResNet-LSTM tuning)"
    elif best_agg >= 1.0 and best_era3 <= 0.5:
        verdict = "→ ship 3b ETH historical (Lessmann replication, not deployable)"
    elif best_agg <= 0.5:
        verdict = "→ ship 3b BTC tier1 baseline (ETH does not unlock §16.1 gate)"
    else:
        verdict = "→ ambiguous — manual review needed"

    print(f"  Best aggregate Sharpe across configs: {best_agg:+.3f}")
    print(f"  Best Era 3 Sharpe at that config:     {best_era3:+.3f}")
    print(f"  VERDICT: {verdict}")
    print()


def main(asset: str = "ETH") -> None:
    sym = asset.lower()
    l0_path = REPORTS_DIR / f"lgbm_results_{sym}.json"
    joint_path = REPORTS_DIR / f"joint_tb03_threshold_sweep_{sym}.json"

    print(f"Loading {l0_path.name}, {joint_path.name}...")
    l0 = json.load(open(l0_path))
    joint = json.load(open(joint_path))

    l0_eras = _segment_l0(l0["per_fold"])
    joint_eras = _segment_joint(joint["by_threshold"])

    _print_l0_table(l0_eras, asset)
    _print_joint_table(joint_eras, asset)
    _decision_summary(asset, l0["aggregate"], l0_eras, joint["by_threshold"], joint_eras)

    out_path = REPORTS_DIR / f"regime_segmented_sharpe_{sym}.json"
    out = {
        "asset": asset,
        "era_bounds": [{"name": n, "fold_ids": ids} for n, ids in ERA_BOUNDS],
        "l0_default_thr_0.60": {
            "aggregate": l0["aggregate"],
            "by_era": l0_eras,
        },
        "joint_tb03_sweep": {
            "by_threshold": {
                thr: {
                    "aggregate": joint["by_threshold"][thr]["aggregate"],
                    "by_era": joint_eras[thr],
                }
                for thr in joint["by_threshold"]
            }
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="ETH")
    args = ap.parse_args()
    main(args.asset)
