"""Triple-barrier labeler for Phase 0.4 (spec §8.1 + DR v3.0.8).

Reads bars from `events.bars_btc_cusum`, applies the triple-barrier
algorithm with the §10.1-frozen parameters (TP=0.05, SL=0.05, vertical=24
for BTC), and writes a labels parquet keyed by `bar_id` for downstream
JOIN with features.

Class encoding per spec §8.1:
  LONG=0, SHORT=1, NEUTRAL=2, UNLABELABLE=-1

CLI:
    python -m labels.triple_barrier                  # full build
    python -m labels.triple_barrier --dry-run        # in-memory + sanity
    python -m labels.triple_barrier --month YYYY-MM  # smoke (single month)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from data.db import close_pool, get_engine

LOG = logging.getLogger("labels.triple_barrier")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

LABEL_LONG = 0
LABEL_SHORT = 1
LABEL_NEUTRAL = 2
LABEL_UNLABELABLE = -1


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _check_frozen_params(cfg: dict) -> None:
    """DR v3.0.8 §8: assert config matches §10.1 frozen Phase A values."""
    tp = cfg["labeling"]["tp_pct"]["BTC"]
    sl = cfg["labeling"]["sl_pct"]["BTC"]
    vert = cfg["labeling"]["vertical_bars"]
    if not (tp == 0.05 and sl == 0.05 and vert == 24):
        raise AssertionError(
            f"Frozen Phase A parameter drift detected: tp={tp}, sl={sl}, "
            f"vertical={vert}. Spec §10.1 freezes BTC TP=0.05, SL=0.05, "
            f"vertical=24. Any change requires a DR in PROJECT_LOG.md."
        )


# ─────────────────────────────────────────────────────────────────────
# Triple-barrier algorithm (DR v3.0.8 §2-§4, §9)
# ─────────────────────────────────────────────────────────────────────
def apply_triple_barrier(
    bars_df: pd.DataFrame,
    tp_pct: float = 0.05,
    sl_pct: float = 0.05,
    vertical_bars: int = 24,
) -> pd.DataFrame:
    """Apply triple-barrier labeling to a bars DataFrame.

    `bars_df` must have columns `bar_id`, `close`, `high`, `low` and be
    sorted ascending by (bar_close_ts, bar_id). Returns a DataFrame
    matching the DR v3.0.8 §1 schema, in the same row order.
    """
    n = len(bars_df)
    bar_ids = bars_df["bar_id"].astype("int64").to_numpy()
    closes = bars_df["close"].astype("float64").to_numpy()
    highs = bars_df["high"].astype("float64").to_numpy()
    lows = bars_df["low"].astype("float64").to_numpy()

    labels = np.full(n, LABEL_UNLABELABLE, dtype=np.int8)
    exit_bar_ids: list = [None] * n          # → Int64 nullable
    exit_reasons: list = [pd.NA] * n          # → string nullable
    holding_bars_arr: list = [None] * n       # → Int8 nullable
    exit_prices = np.full(n, np.nan, dtype=np.float64)

    for t in range(n):
        if t + vertical_bars >= n:
            continue  # UNLABELABLE — sentinel already set

        P_t = closes[t]
        TP = P_t * (1.0 + tp_pct)
        SL = P_t * (1.0 - sl_pct)

        end = t + vertical_bars
        determined = False
        for k in range(t + 1, end + 1):
            tp_hit = highs[k] >= TP
            sl_hit = lows[k] <= SL

            if tp_hit and sl_hit:
                # Both barriers in one bar → ambiguous whipsaw
                labels[t] = LABEL_NEUTRAL
                exit_reasons[t] = "ambiguous"
                exit_bar_ids[t] = int(bar_ids[k])
                holding_bars_arr[t] = k - t
                exit_prices[t] = closes[k]
                determined = True
                break
            elif tp_hit:
                labels[t] = LABEL_LONG
                exit_reasons[t] = "tp"
                exit_bar_ids[t] = int(bar_ids[k])
                holding_bars_arr[t] = k - t
                exit_prices[t] = closes[k]
                determined = True
                break
            elif sl_hit:
                labels[t] = LABEL_SHORT
                exit_reasons[t] = "sl"
                exit_bar_ids[t] = int(bar_ids[k])
                holding_bars_arr[t] = k - t
                exit_prices[t] = closes[k]
                determined = True
                break

        if not determined:
            labels[t] = LABEL_NEUTRAL
            exit_reasons[t] = "timeout"
            exit_bar_ids[t] = int(bar_ids[end])
            holding_bars_arr[t] = vertical_bars
            exit_prices[t] = closes[end]

    return pd.DataFrame({
        "bar_id": bar_ids,
        "label": labels,
        "exit_bar_id": pd.array(exit_bar_ids, dtype="Int64"),
        "exit_reason": pd.array(exit_reasons, dtype="string"),
        "holding_bars": pd.array(holding_bars_arr, dtype="Int8"),
        "exit_price": exit_prices,
    })


# ─────────────────────────────────────────────────────────────────────
# Path-dependence diagnostic (DR v3.0.8 §12 fold)
# ─────────────────────────────────────────────────────────────────────
def _path_dependence_check(
    labels_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    tp_pct: float = 0.05,
    sl_pct: float = 0.05,
    vertical_bars: int = 24,
) -> dict:
    """For each LONG (clean tp), check if SL was touched after exit.
    Mirror for SHORT. Returns counts."""
    bar_ids = bars_df["bar_id"].astype("int64").to_numpy()
    closes = bars_df["close"].astype("float64").to_numpy()
    highs = bars_df["high"].astype("float64").to_numpy()
    lows = bars_df["low"].astype("float64").to_numpy()
    bid_to_idx = {int(b): i for i, b in enumerate(bar_ids)}

    n_long = n_long_then_sl = 0
    n_short = n_short_then_tp = 0

    for _, row in labels_df.iterrows():
        reason = row["exit_reason"]
        if pd.isna(reason):
            continue
        if reason == "tp":
            n_long += 1
            t_idx = bid_to_idx[int(row["bar_id"])]
            exit_idx = bid_to_idx[int(row["exit_bar_id"])]
            P_t = closes[t_idx]
            SL = P_t * (1.0 - sl_pct)
            end = min(t_idx + vertical_bars, len(closes) - 1)
            for k in range(exit_idx + 1, end + 1):
                if lows[k] <= SL:
                    n_long_then_sl += 1
                    break
        elif reason == "sl":
            n_short += 1
            t_idx = bid_to_idx[int(row["bar_id"])]
            exit_idx = bid_to_idx[int(row["exit_bar_id"])]
            P_t = closes[t_idx]
            TP = P_t * (1.0 + tp_pct)
            end = min(t_idx + vertical_bars, len(closes) - 1)
            for k in range(exit_idx + 1, end + 1):
                if highs[k] >= TP:
                    n_short_then_tp += 1
                    break

    return {
        "n_long": n_long,
        "n_long_then_sl": n_long_then_sl,
        "n_short": n_short,
        "n_short_then_tp": n_short_then_tp,
    }


# ─────────────────────────────────────────────────────────────────────
# DB load + parquet write
# ─────────────────────────────────────────────────────────────────────
def _load_bars(threshold: float, month_filter: Optional[date] = None) -> pd.DataFrame:
    where = "WHERE threshold_pct = %(t)s"
    params: dict = {"t": threshold}
    if month_filter is not None:
        next_m = (
            date(month_filter.year + 1, 1, 1)
            if month_filter.month == 12
            else date(month_filter.year, month_filter.month + 1, 1)
        )
        where += " AND bar_close_ts >= %(s)s AND bar_close_ts < %(e)s"
        params["s"] = month_filter
        params["e"] = next_m
    sql = f"""
        SELECT bar_id, bar_close_ts, close, high, low
        FROM events.bars_btc_cusum
        {where}
        ORDER BY bar_close_ts, bar_id
    """
    return pd.read_sql_query(sql, get_engine(), params=params)


def _md5_dataframe(df: pd.DataFrame) -> str:
    """Canonical md5: sort by bar_id, text-serialize all cols, hash."""
    sorted_df = df.sort_values("bar_id").reset_index(drop=True)
    h = hashlib.md5()
    for col in sorted_df.columns:
        h.update(col.encode("utf-8"))
        h.update(b"\x00")
        h.update(sorted_df[col].astype(str).str.cat(sep="\n").encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────
# Sanity report (DR v3.0.8 §12)
# ─────────────────────────────────────────────────────────────────────
def sanity_report(
    labels_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    parquet_path: Optional[Path],
    smoke: bool,
) -> dict:
    print("\n========== Phase 0.4 Sanity Report ==========")
    print(f"Rows:         {len(labels_df):,}")
    print(f"Columns:      {len(labels_df.columns)}  (expect 6)")
    expected_cols = ["bar_id", "label", "exit_bar_id", "exit_reason",
                     "holding_bars", "exit_price"]
    assert list(labels_df.columns) == expected_cols, (
        f"column order mismatch: got {list(labels_df.columns)}"
    )

    # Class distribution
    print("\n--- Class distribution ---")
    lab = labels_df["label"]
    n_total = len(lab)
    n_unlab = int((lab == LABEL_UNLABELABLE).sum())
    n_labelable = n_total - n_unlab
    n_long = int((lab == LABEL_LONG).sum())
    n_short = int((lab == LABEL_SHORT).sum())
    n_neutral = int((lab == LABEL_NEUTRAL).sum())

    print(f"  UNLABELABLE: {n_unlab:>6,}  ({n_unlab*100/n_total:>5.2f}% of all bars)")
    print(f"  --- below: % of {n_labelable} labelable bars ---")
    if n_labelable > 0:
        long_pct = n_long * 100 / n_labelable
        short_pct = n_short * 100 / n_labelable
        neutral_pct = n_neutral * 100 / n_labelable
        print(f"  LONG    (0): {n_long:>6,}  ({long_pct:>5.2f}%)")
        print(f"  SHORT   (1): {n_short:>6,}  ({short_pct:>5.2f}%)")
        print(f"  NEUTRAL (2): {n_neutral:>6,}  ({neutral_pct:>5.2f}%)")
    else:
        long_pct = short_pct = neutral_pct = 0.0
        print("  (no labelable bars in this slice)")

    # exit_reason distribution
    print("\n--- exit_reason distribution ---")
    reasons = labels_df["exit_reason"].fillna("(null)").value_counts()
    for reason, cnt in reasons.items():
        print(f"  {str(reason):<14} {cnt:>6,}")

    # holding_bars histogram
    print("\n--- holding_bars distribution ---")
    hb = labels_df["holding_bars"].dropna().astype(int)
    if len(hb) > 0:
        bins = hb.value_counts().sort_index()
        for h, cnt in bins.items():
            print(f"  {h:>3}: {cnt:>6,}")
        print(f"  median={int(hb.median())}  mean={hb.mean():.2f}  max={int(hb.max())}")

    # §8.3 expected-range check
    print("\n--- §8.3 class-balance check ---")
    if n_labelable == 0:
        print("  (no labelable bars — skipping)")
    else:
        in_spec = (
            35.0 <= long_pct <= 40.0
            and 35.0 <= short_pct <= 40.0
            and 20.0 <= neutral_pct <= 30.0
        )
        print(f"  LONG    in [35,40]: {35.0 <= long_pct <= 40.0}  ({long_pct:.2f}%)")
        print(f"  SHORT   in [35,40]: {35.0 <= short_pct <= 40.0}  ({short_pct:.2f}%)")
        print(f"  NEUTRAL in [20,30]: {20.0 <= neutral_pct <= 30.0}  ({neutral_pct:.2f}%)")
        print(f"  ALL within §8.3 expected range: {in_spec}")

        # Hard-fail bounds (DR §6) — full sweep only
        if not smoke:
            worst = max(long_pct, short_pct, neutral_pct)
            best = min(long_pct, short_pct, neutral_pct)
            classes = {"LONG": long_pct, "SHORT": short_pct, "NEUTRAL": neutral_pct}
            if worst > 50.0 or best < 10.0:
                worst_class = max(classes, key=classes.get)
                least_class = min(classes, key=classes.get)
                raise AssertionError(
                    f"Class balance violation: {worst_class}={worst:.2f}% (>50% bound) "
                    f"or {least_class}={best:.2f}% (<10% bound). "
                    f"See spec §8.3 for expected class balance ranges "
                    f"(35-40% LONG/SHORT, 20-30% NEUTRAL). "
                    f"Investigate label-config (TP/SL vs CUSUM threshold) "
                    f"before proceeding to Phase 1."
                )
        else:
            print("  (smoke mode — hard-fail bounds skipped)")

    # Path-dependence diagnostic (DR §12 fold)
    print("\n--- Path-dependence diagnostic ---")
    pd_stats = _path_dependence_check(labels_df, bars_df)
    if pd_stats["n_long"] > 0:
        long_then_sl_pct = pd_stats["n_long_then_sl"] * 100 / pd_stats["n_long"]
        print(f"  LONG labels where SL touched after TP exit: "
              f"{pd_stats['n_long_then_sl']:,} / {pd_stats['n_long']:,} "
              f"({long_then_sl_pct:.2f}%)")
    else:
        long_then_sl_pct = 0.0
        print("  (no LONG labels in this slice)")
    if pd_stats["n_short"] > 0:
        short_then_tp_pct = pd_stats["n_short_then_tp"] * 100 / pd_stats["n_short"]
        print(f"  SHORT labels where TP touched after SL exit: "
              f"{pd_stats['n_short_then_tp']:,} / {pd_stats['n_short']:,} "
              f"({short_then_tp_pct:.2f}%)")
    else:
        short_then_tp_pct = 0.0
        print("  (no SHORT labels in this slice)")
    print(f"  guidance: <10% clean (sustained moves), "
          f">40% noisy (frequent post-exit reversal — informational)")

    # Determinism fingerprint
    fp = _md5_dataframe(labels_df)
    print(f"\n--- Determinism fingerprint ---")
    print(f"  md5(labels) = {fp}")

    if parquet_path is not None and parquet_path.exists():
        size = parquet_path.stat().st_size
        print(f"\nParquet:      {parquet_path}  ({size/1e6:.2f} MB)")

    return {
        "n_total": n_total,
        "n_labelable": n_labelable,
        "n_long": n_long,
        "n_short": n_short,
        "n_neutral": n_neutral,
        "n_unlab": n_unlab,
        "fingerprint": fp,
        "path_dep": pd_stats,
    }


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────
def run_label(
    month_filter: Optional[date] = None,
    dry_run: bool = False,
) -> dict:
    cfg = load_config()
    _check_frozen_params(cfg)

    threshold = cfg["bars"]["threshold"]["BTC"]
    tp_pct = cfg["labeling"]["tp_pct"]["BTC"]
    sl_pct = cfg["labeling"]["sl_pct"]["BTC"]
    vertical_bars = cfg["labeling"]["vertical_bars"]

    output_dir = PROJECT_ROOT / "data" / "storage" / "labels"
    output_path = output_dir / "labels_btc.parquet"

    LOG.info("loading bars: threshold=%s month=%s",
             threshold, f"{month_filter:%Y-%m}" if month_filter else "<all>")
    t0 = time.perf_counter()
    bars = _load_bars(threshold, month_filter)
    load_s = time.perf_counter() - t0
    LOG.info("loaded %d bars in %.1fs", len(bars), load_s)

    if len(bars) == 0:
        raise RuntimeError("No bars matched the filter — refusing to write empty labels parquet")

    LOG.info("applying triple-barrier...")
    t1 = time.perf_counter()
    labels_df = apply_triple_barrier(bars, tp_pct=tp_pct, sl_pct=sl_pct,
                                     vertical_bars=vertical_bars)
    label_s = time.perf_counter() - t1
    LOG.info("labeled %d bars in %.1fs", len(labels_df), label_s)

    parquet_path = None
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_df.to_parquet(output_path, index=False)
        parquet_path = output_path
        LOG.info("wrote parquet: %s (%.2f MB)",
                 output_path, output_path.stat().st_size / 1e6)
    else:
        LOG.info("dry-run: skipping parquet write")

    smoke = month_filter is not None
    rep = sanity_report(labels_df, bars, parquet_path, smoke=smoke)
    rep.update({"load_s": load_s, "label_s": label_s})
    return rep


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="labels.triple_barrier")
    p.add_argument("--month", metavar="YYYY-MM",
                   help="Limit to one calendar month (smoke / debug). "
                   "Disables hard-fail class-balance bounds.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build in memory + report; do not write parquet.")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    month_filter: Optional[date] = None
    if args.month:
        month_filter = date.fromisoformat(args.month + "-01")

    try:
        run_label(month_filter=month_filter, dry_run=args.dry_run)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
