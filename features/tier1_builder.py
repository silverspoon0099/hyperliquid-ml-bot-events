"""Tier 1 feature additions (spec §7.2 + DR v3.0.13).

Adds 15 features across 4 categories to the existing 33-feature
`features_btc.parquet`:

  1. bars_since_* event-memory (6) — 30m project's strongest signal
  2. HTF context (3) — log-returns at 4h / 1d / 5d horizons
  3. ATR + percentile (2) — vol-regime classifier
  4. Fibonacci pivot distance (4) — chart-reading anchor

Reads `data/storage/features/features_btc.parquet` (33 features) and
`events.bars_btc_cusum` (close/high/low). Writes
`data/storage/features/features_btc_tier1.parquet` (48 columns:
2 keys + 33 original + 15 new).

CLI:
    python -m features.tier1_builder
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.db import close_pool, get_engine

LOG = logging.getLogger("features.tier1_builder")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────
# Feature inventory (DR v3.0.13)
# ─────────────────────────────────────────────────────────────────────
TIER1_BARS_SINCE = [
    "bars_since_rsi_ob_14", "bars_since_rsi_os_14",
    "bars_since_macd_cross",
    "bars_since_volume_spike",
    "bars_since_close_gt_ema50", "bars_since_close_lt_ema50",
]
TIER1_HTF = ["htf_ret_4h", "htf_ret_1d", "htf_ret_5d"]
TIER1_ATR = ["atr_14", "atr_pct_rank_100"]
TIER1_PIVOT = [
    "pivot_distance", "r1_distance", "s1_distance", "fib_618_distance",
]
TIER1_ALL = TIER1_BARS_SINCE + TIER1_HTF + TIER1_ATR + TIER1_PIVOT


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _bars_since_event(events: pd.Series) -> pd.Series:
    """Number of bars since the last True in `events`. NaN before the
    first event ever fires.

    `events` is a boolean (or 0/1) Series indexed 0..N-1.
    """
    n = len(events)
    pos = pd.Series(np.arange(n, dtype=float))
    last_event_pos = pos.where(events.astype(bool), np.nan).ffill()
    return pos - last_event_pos  # NaN when no prior event


def compute_bars_since(
    feats: pd.DataFrame, bars: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the 6 bars_since_* features per DR §1.

    Inputs (assumed sorted by bar_close_ts asc, bar_id asc):
        feats: features parquet (has rsi_14, macd_line, volume, ema_50)
        bars:  bar_id, close
    """
    df = feats.merge(bars[["bar_id", "close"]], on="bar_id", how="inner")

    out = pd.DataFrame(index=df.index)
    out["bars_since_rsi_ob_14"] = _bars_since_event(df["rsi_14"] > 70)
    out["bars_since_rsi_os_14"] = _bars_since_event(df["rsi_14"] < 30)

    macd_sign = np.sign(df["macd_line"])
    macd_cross_event = (macd_sign != macd_sign.shift(1)) & macd_sign.notna() & macd_sign.shift(1).notna()
    out["bars_since_macd_cross"] = _bars_since_event(macd_cross_event)

    vol_med_50 = df["volume"].rolling(50, min_periods=1).median()
    out["bars_since_volume_spike"] = _bars_since_event(df["volume"] > 3.0 * vol_med_50)

    out["bars_since_close_gt_ema50"] = _bars_since_event(df["close"] > df["ema_50"])
    out["bars_since_close_lt_ema50"] = _bars_since_event(df["close"] < df["ema_50"])

    return out.astype("float64")


def compute_htf(bars: pd.DataFrame) -> pd.DataFrame:
    """HTF log-returns at 4h / 1d / 5d via merge_asof.

    Returns a DataFrame indexed by bar_id with htf_ret_{4h, 1d, 5d}.
    """
    b = bars[["bar_id", "bar_close_ts", "close"]].copy()
    b["bar_close_ts"] = pd.to_datetime(b["bar_close_ts"], utc=True)
    b = b.sort_values("bar_close_ts").reset_index(drop=True)

    out = pd.DataFrame({"bar_id": b["bar_id"].values})

    for label, td in [
        ("htf_ret_4h", pd.Timedelta(hours=4)),
        ("htf_ret_1d", pd.Timedelta(days=1)),
        ("htf_ret_5d", pd.Timedelta(days=5)),
    ]:
        # For each bar at time T, find the latest bar with ts ≤ T - td
        target = b.copy()
        target["lookup_ts"] = target["bar_close_ts"] - td
        # merge_asof: left keys must be sorted (already are)
        target_sorted = target.sort_values("lookup_ts").reset_index(drop=True)
        merged = pd.merge_asof(
            target_sorted[["bar_id", "lookup_ts"]],
            b[["bar_close_ts", "close"]].rename(columns={"close": "close_past"}),
            left_on="lookup_ts", right_on="bar_close_ts",
            direction="backward", allow_exact_matches=True,
        )
        # Re-align to original bar_id order
        merged = merged.set_index("bar_id").reindex(b["bar_id"].values)
        ret = np.log(b["close"].values / merged["close_past"].values)
        out[label] = ret

    return out.astype({c: "float64" for c in TIER1_HTF})


def compute_atr(bars: pd.DataFrame) -> pd.DataFrame:
    """ATR(14) + percentile rank over rolling 100 bars.

    Simple ATR proxy: mean(high - low) over 14 bars (no true-range
    accounting — sufficient for relative-rank purposes here).
    """
    b = bars[["bar_id", "high", "low"]].copy()
    out = pd.DataFrame({"bar_id": b["bar_id"].values})
    bar_range = (b["high"] - b["low"]).astype(float)
    out["atr_14"] = bar_range.rolling(14, min_periods=14).mean().values
    out["atr_pct_rank_100"] = (
        pd.Series(out["atr_14"])
        .rolling(100, min_periods=100)
        .rank(pct=True)
        .values
    )
    return out.astype({"atr_14": "float64", "atr_pct_rank_100": "float64"})


def compute_pivot(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily-aggregated Fibonacci pivot proximity (4 features).

    For each bar at time T, look up the previous calendar day's
    aggregated H/L/C (deterministic from past data — no leakage)
    and compute distance from current close to pivot, R1, S1, fib_618.
    """
    b = bars[["bar_id", "bar_close_ts", "close", "high", "low"]].copy()
    b["bar_close_ts"] = pd.to_datetime(b["bar_close_ts"], utc=True)
    b["date"] = b["bar_close_ts"].dt.tz_convert("UTC").dt.date

    # Daily H/L/C aggregated from CUSUM bars
    daily = b.groupby("date").agg(
        H=("high", "max"),
        L=("low", "min"),
        C=("close", "last"),
    )
    daily.index = pd.to_datetime(daily.index)
    daily["P"] = (daily["H"] + daily["L"] + daily["C"]) / 3.0
    daily["R1"] = 2.0 * daily["P"] - daily["L"]
    daily["S1"] = 2.0 * daily["P"] - daily["H"]
    daily["fib_618"] = daily["P"] + 0.618 * (daily["H"] - daily["L"])
    daily_shifted = daily.shift(1)  # use PRIOR day's pivot, not current

    # Map each bar to its prior-day pivot via date lookup
    bar_dates = pd.to_datetime(b["date"])
    prior = daily_shifted.reindex(bar_dates).reset_index(drop=True)

    out = pd.DataFrame({"bar_id": b["bar_id"].values})
    close_arr = b["close"].astype(float).values
    with np.errstate(divide="ignore", invalid="ignore"):
        out["pivot_distance"]   = np.log(close_arr / prior["P"].values)
        out["r1_distance"]      = np.log(close_arr / prior["R1"].values)
        out["s1_distance"]      = np.log(close_arr / prior["S1"].values)
        out["fib_618_distance"] = np.log(close_arr / prior["fib_618"].values)

    return out.astype({c: "float64" for c in TIER1_PIVOT})


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────
def build_tier1_features() -> pd.DataFrame:
    """Build the extended 48-column features DataFrame."""
    feats_path = PROJECT_ROOT / "data/storage/features/features_btc.parquet"
    LOG.info("loading existing features from %s", feats_path)
    feats = pd.read_parquet(feats_path)

    LOG.info("loading bars OHLC from DB...")
    sql = """
        SELECT bar_id, bar_close_ts, close, high, low
        FROM events.bars_btc_cusum
        ORDER BY bar_close_ts, bar_id
    """
    bars = pd.read_sql_query(sql, get_engine())

    # Sort consistently
    feats = feats.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)
    bars = bars.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)

    LOG.info("computing tier-1 features (15)...")
    t0 = time.perf_counter()

    bs = compute_bars_since(feats, bars)
    bs.insert(0, "bar_id", feats["bar_id"].values)

    htf = compute_htf(bars)
    atr = compute_atr(bars)
    piv = compute_pivot(bars)

    out = (
        feats.merge(bs, on="bar_id", how="left")
             .merge(htf, on="bar_id", how="left")
             .merge(atr, on="bar_id", how="left")
             .merge(piv, on="bar_id", how="left")
    )
    LOG.info("built %d × %d in %.1fs", len(out), len(out.columns),
             time.perf_counter() - t0)
    return out


def write_tier1_parquet(df: pd.DataFrame) -> Path:
    out_path = PROJECT_ROOT / "data/storage/features/features_btc_tier1.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    LOG.info("wrote %s (%.2f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        df = build_tier1_features()
        path = write_tier1_parquet(df)

        # Brief sanity report
        print("\n========== Tier 1 features sanity ==========")
        print(f"rows × cols: {len(df)} × {len(df.columns)}  (expect 18,629 × 50)")
        print(f"new feature NaN counts (warmup edges):")
        for col in TIER1_ALL:
            n_na = int(df[col].isna().sum())
            print(f"  {col:<28} {n_na:>5} NaN")
        print(f"\nparquet: {path}  ({path.stat().st_size/1e6:.2f} MB)")
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
