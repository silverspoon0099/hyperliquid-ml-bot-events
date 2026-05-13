"""v2 feature additions (DR v3.0.16).

Adds 8 features across 2 categories to the existing 33-feature
`features_{sym}.parquet`:

  A. Order flow imbalance (5) — derived from raw aggTrades
     - taker_buy_ratio: taker buy vol / total vol in bar
     - taker_buy_ratio_ema5, ema20: smoothed momentum
     - max_trade_share: largest single trade / total bar volume
     - trade_intensity: (n_trades / duration_sec) / trailing-50 median

  B. Higher-timeframe context (3) — pure pandas on bars
     - daily_ret_pct: return since 00:00 UTC at bar close
     - weekly_range_pos: position in trailing 7-day high/low range
     - regime_vol_ratio: trailing-50 std(ret_1) / trailing-200 median std

Hypothesis: information ceiling — the 33 Lessmann features miss
order flow + HTF context. If aggregate Sharpe lifts meaningfully,
the bottleneck was information, not model architecture.

CLI:
    python -m features.v2_builder --symbol BTC
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.db import close_pool, get_connection, get_engine, symbol_short, ticks_table, bars_table

LOG = logging.getLogger("features.v2_builder")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

V2_ORDERFLOW = [
    "taker_buy_ratio",
    "taker_buy_ratio_ema5",
    "taker_buy_ratio_ema20",
    "max_trade_share",
    "trade_intensity",
]
V2_HTF = [
    "daily_ret_pct",
    "weekly_range_pos",
    "regime_vol_ratio",
]
V2_NEW = V2_ORDERFLOW + V2_HTF

assert len(V2_NEW) == 8, f"v2 added count {len(V2_NEW)} != 8"


# ─────────────────────────────────────────────────────────────────────
# Order flow aggregation (per-month batched)
# ─────────────────────────────────────────────────────────────────────
def _month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return list of (month_start, month_end_exclusive) tuples."""
    out = []
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end.tz is None else pd.Timestamp(end)
    while cur < end_ts:
        if cur.month == 12:
            nxt = pd.Timestamp(cur.year + 1, 1, 1, tz="UTC")
        else:
            nxt = pd.Timestamp(cur.year, cur.month + 1, 1, tz="UTC")
        out.append((cur, nxt))
        cur = nxt
    return out


def compute_orderflow_features(
    symbol: str, threshold: float,
    start_month: Optional[pd.Timestamp] = None,
    end_month: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Aggregate per-bar order flow stats from raw aggTrades.

    Returns DataFrame [bar_id, taker_buy_vol, taker_sell_vol,
    total_vol_ticks, max_trade_qty, n_ticks_in_bar].

    Strategy: psycopg COPY streaming + numpy searchsorted in Python.
    The SQL-side inequality range-join (t.ts > b.bar_open_ts AND t.ts <=
    b.bar_close_ts) forces NestedLoop on Postgres which is 30+ min/month
    on bull-run high-vol months. Instead:

      1. Load all bars (small, ~18k rows) into a numpy array of close_ts
      2. Stream ticks per month via COPY (10-20x faster than read_sql)
      3. numpy.searchsorted assigns each tick to its bar in O(log N)
      4. groupby + aggregate per bar_id, accumulate across months

    Per-month time: dominated by tick volume × disk read speed.
    Expected: 10-40 sec/month for normal months, 1-3 min for bull months.
    """
    bars_tbl = bars_table(symbol)
    ticks_tbl = ticks_table(symbol)

    LOG.info("loading bar boundaries [%s] thr=%.2f...", bars_tbl, threshold)
    bars_sql = f"""
        SELECT bar_id, bar_open_ts, bar_close_ts
        FROM {bars_tbl}
        WHERE threshold_pct = {threshold}
        ORDER BY bar_close_ts, bar_id
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(bars_sql)
            bars_rows = cur.fetchall()
    bars = pd.DataFrame(bars_rows)
    bars["bar_close_ts"] = pd.to_datetime(bars["bar_close_ts"], utc=True)
    bars["bar_open_ts"] = pd.to_datetime(bars["bar_open_ts"], utc=True)
    LOG.info("  loaded %d bars (%s .. %s)",
             len(bars), bars["bar_open_ts"].iloc[0], bars["bar_close_ts"].iloc[-1])

    # Apply date range filter if specified (DR v3.0.16: recent regime scope)
    if start_month is not None or end_month is not None:
        sm = start_month if start_month is not None else bars["bar_open_ts"].iloc[0]
        em = end_month if end_month is not None else bars["bar_close_ts"].iloc[-1] + pd.Timedelta(days=1)
        if sm.tzinfo is None:
            sm = sm.tz_localize("UTC")
        if em.tzinfo is None:
            em = em.tz_localize("UTC")
        bars = bars[(bars["bar_close_ts"] >= sm) & (bars["bar_close_ts"] < em)].reset_index(drop=True)
        LOG.info("  filtered to date range [%s, %s): %d bars", sm, em, len(bars))

    if len(bars) == 0:
        return pd.DataFrame(columns=["bar_id", "taker_buy_vol", "taker_sell_vol",
                                      "total_vol_ticks", "max_trade_qty", "n_ticks_in_bar"])

    # Strip tz for numpy comparisons. Force datetime64[ns] resolution
    # explicitly so view('int64') yields nanoseconds consistently.
    bar_close_arr = bars["bar_close_ts"].dt.tz_localize(None).values.astype("datetime64[ns]")
    bar_id_arr = bars["bar_id"].values.astype(np.int64)
    first_open_ts = bars["bar_open_ts"].iloc[0]
    last_close_ts = bars["bar_close_ts"].iloc[-1]
    # Critical: np.datetime64(pd.Timestamp) defaults to datetime64[us], not [ns].
    # We need [ns] for comparison with ts_ns (which is nanoseconds).
    first_open_naive = np.datetime64(first_open_ts.tz_localize(None)).astype("datetime64[ns]")

    months = _month_ranges(first_open_ts, last_close_ts + pd.Timedelta(days=1))
    LOG.info("processing %d months via COPY-streaming + numpy aggregation...", len(months))

    # Per bar_id accumulator
    n_bars = len(bars)
    acc_tbv = np.zeros(n_bars, dtype=np.float64)
    acc_tsv = np.zeros(n_bars, dtype=np.float64)
    acc_tv = np.zeros(n_bars, dtype=np.float64)
    acc_maxq = np.zeros(n_bars, dtype=np.float64)
    acc_n = np.zeros(n_bars, dtype=np.int64)

    t0_total = time.perf_counter()
    for i, (m_start, m_end) in enumerate(months, 1):
        t0 = time.perf_counter()
        # Non-overlapping month windows (half-open [m_start, m_end)). Bars that
        # span boundaries naturally collect ticks from each month's window via
        # searchsorted — no buffer needed (and buffering would double-count).
        copy_sql = (
            f"COPY (SELECT EXTRACT(EPOCH FROM ts) * 1e9, qty, "
            f"is_buyer_maker FROM {ticks_tbl} "
            f"WHERE ts >= '{m_start.isoformat()}' AND ts < '{m_end.isoformat()}') "
            f"TO STDOUT WITH (FORMAT CSV)"
        )
        with get_connection() as conn:
            with conn.cursor() as cur:
                with cur.copy(copy_sql) as copy:
                    chunks = []
                    for chunk in copy:
                        chunks.append(bytes(chunk))
                    raw = b"".join(chunks)
        if not raw:
            dt = time.perf_counter() - t0
            LOG.info("  [%3d/%d] %s: 0 ticks  (%.1fs)", i, len(months),
                     m_start.strftime("%Y-%m"), dt)
            continue

        # Parse CSV bytes → numpy arrays
        df = pd.read_csv(
            io.BytesIO(raw),
            header=None,
            names=["ts_ns", "qty", "is_buyer_maker"],
            dtype={"ts_ns": np.float64, "qty": np.float64, "is_buyer_maker": str},
        )
        n_ticks = len(df)
        if n_ticks == 0:
            dt = time.perf_counter() - t0
            LOG.info("  [%3d/%d] %s: 0 ticks  (%.1fs)", i, len(months),
                     m_start.strftime("%Y-%m"), dt)
            continue

        ts_ns = df["ts_ns"].to_numpy(dtype=np.int64)
        qty = df["qty"].to_numpy(dtype=np.float64)
        ibm = (df["is_buyer_maker"].to_numpy() == "t")

        # bar_close_ts as int64 nanoseconds for searchsorted
        bar_close_ns = bar_close_arr.view("int64")
        idx = np.searchsorted(bar_close_ns, ts_ns, side="left")

        # Mask ticks outside [first_open_ns, last_close_ns]
        first_open_ns = first_open_naive.view("int64")
        valid = (idx < n_bars) & (ts_ns > first_open_ns)
        idx = idx[valid]
        qty_v = qty[valid]
        ibm_v = ibm[valid]

        # Accumulate per bar_id via numpy bincount-like additions
        buy_qty = np.where(~ibm_v, qty_v, 0.0)
        sell_qty = np.where(ibm_v, qty_v, 0.0)
        np.add.at(acc_tbv, idx, buy_qty)
        np.add.at(acc_tsv, idx, sell_qty)
        np.add.at(acc_tv, idx, qty_v)
        # max via reduceat is non-trivial; loop replacement: scan order is OK
        np.maximum.at(acc_maxq, idx, qty_v)
        np.add.at(acc_n, idx, 1)

        dt = time.perf_counter() - t0
        rate = n_ticks / dt / 1000.0
        LOG.info("  [%3d/%d] %s: %d ticks  (%.1fs, %.0fk/s)",
                 i, len(months), m_start.strftime("%Y-%m"), n_ticks, dt, rate)

    total_dt = time.perf_counter() - t0_total
    LOG.info("orderflow aggregation done in %.1fs (%.1f min)", total_dt, total_dt / 60.0)

    # Keep only bars that received ticks
    has_data = acc_n > 0
    out = pd.DataFrame({
        "bar_id": bar_id_arr[has_data],
        "taker_buy_vol": acc_tbv[has_data],
        "taker_sell_vol": acc_tsv[has_data],
        "total_vol_ticks": acc_tv[has_data],
        "max_trade_qty": acc_maxq[has_data],
        "n_ticks_in_bar": acc_n[has_data],
    })
    out["bar_id"] = out["bar_id"].astype("int64")
    out = out.sort_values("bar_id").reset_index(drop=True)
    return out


def compute_orderflow_derived(of_raw: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Convert raw aggregates into the 5 V2_ORDERFLOW features.

    bars must have [bar_id, bar_close_ts, bar_open_ts, n_trades] sorted by bar_close_ts.
    """
    b = bars[["bar_id", "bar_close_ts", "bar_open_ts", "n_trades"]].copy()
    b["bar_close_ts"] = pd.to_datetime(b["bar_close_ts"], utc=True)
    b["bar_open_ts"] = pd.to_datetime(b["bar_open_ts"], utc=True)
    b["duration_sec"] = (b["bar_close_ts"] - b["bar_open_ts"]).dt.total_seconds().astype(float)
    b = b.sort_values("bar_close_ts").reset_index(drop=True)

    merged = b.merge(of_raw, on="bar_id", how="left")

    # taker_buy_ratio = tbv / total_vol; NaN where order flow data missing
    total = merged["total_vol_ticks"].astype(float)
    has_flow = total > 0  # False for both missing and zero-vol
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = merged["taker_buy_vol"].astype(float) / total
    # NaN where missing or zero-vol — let LightGBM handle NaN natively
    ratio = np.where(has_flow.fillna(False), ratio, np.nan)
    merged["taker_buy_ratio"] = ratio

    # EMAs on ratio (NaN propagates appropriately via pandas EWM)
    s = pd.Series(merged["taker_buy_ratio"].values)
    merged["taker_buy_ratio_ema5"] = s.ewm(span=5, adjust=False, min_periods=5).mean().values
    merged["taker_buy_ratio_ema20"] = s.ewm(span=20, adjust=False, min_periods=20).mean().values

    # max_trade_share = max_qty / total_vol_ticks; NaN where missing
    with np.errstate(divide="ignore", invalid="ignore"):
        mts = merged["max_trade_qty"].astype(float) / total
    merged["max_trade_share"] = np.where(has_flow.fillna(False), mts, np.nan)

    # trade_intensity = (n_trades / duration_sec) / trailing-50 median
    # This uses bars table (always present), not order flow — full range
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(
            merged["duration_sec"] > 0,
            merged["n_trades"].astype(float) / merged["duration_sec"].astype(float),
            0.0,
        )
    rate_s = pd.Series(rate)
    median_50 = rate_s.rolling(50, min_periods=50).median()
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["trade_intensity"] = np.where(
            (median_50 > 0).to_numpy(), rate / median_50.to_numpy(), 1.0
        )

    out = merged[["bar_id"] + V2_ORDERFLOW].copy()
    for c in V2_ORDERFLOW:
        out[c] = out[c].astype("float64")
    return out


# ─────────────────────────────────────────────────────────────────────
# HTF context (Group B) — pure pandas on bars
# ─────────────────────────────────────────────────────────────────────
def compute_htf_v2(bars: pd.DataFrame) -> pd.DataFrame:
    """3 HTF features: daily_ret_pct, weekly_range_pos, regime_vol_ratio.

    Uses positional indexing throughout (not timestamp-based) to handle
    rare duplicate bar_close_ts values (5 observed in 2019 high-vol day).
    """
    b = bars[["bar_id", "bar_close_ts", "close", "high", "low"]].copy()
    b["bar_close_ts"] = pd.to_datetime(b["bar_close_ts"], utc=True)
    b = b.sort_values(["bar_close_ts", "bar_id"]).reset_index(drop=True)
    n = len(b)
    out = pd.DataFrame({"bar_id": b["bar_id"].values})

    ts_ns = b["bar_close_ts"].dt.tz_localize(None).values.astype(
        "datetime64[ns]"
    ).view("int64")
    close_arr = b["close"].astype(float).values
    high_arr = b["high"].astype(float).values
    low_arr = b["low"].astype(float).values

    # daily_ret_pct: return since 00:00 UTC of current bar's day.
    # For each bar i, find the latest bar j where bar_close_ts[j] >= day_floor(ts[i])
    # i.e., j_start = first index with ts >= day_floor. Then daily_anchor_close = close[j_start - 1]
    # (or close[j_start] if want first-of-day close; we use prior-day close convention).
    day_floor_ns = (ts_ns // (24 * 3600 * int(1e9))) * (24 * 3600 * int(1e9))
    j_first_today = np.searchsorted(ts_ns, day_floor_ns, side="left")
    # daily_anchor = close at the bar that opens the current day
    # = close[j_first_today] if that bar has ts == day_floor (rare), else close[j_first_today - 1]
    # Simplest correct: use close[max(0, j_first_today - 1)] as the prior-day close
    j_anchor = np.clip(j_first_today - 1, 0, n - 1)
    daily_anchor_close = close_arr[j_anchor]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["daily_ret_pct"] = (close_arr / daily_anchor_close) - 1.0
    # First bar of dataset has no prior day -> NaN
    out.loc[j_first_today == 0, "daily_ret_pct"] = np.nan

    # weekly_range_pos: position in trailing 7-day [high, low] range
    # For each bar i, find j_start = first index where ts >= ts[i] - 7d
    week_ns = 7 * 24 * 3600 * int(1e9)
    j_starts = np.searchsorted(ts_ns, ts_ns - week_ns, side="left")
    high_7d = np.empty(n)
    low_7d = np.empty(n)
    for i in range(n):
        js = j_starts[i]
        high_7d[i] = high_arr[js:i + 1].max() if i + 1 > js else high_arr[i]
        low_7d[i] = low_arr[js:i + 1].min() if i + 1 > js else low_arr[i]
    rng = high_7d - low_7d
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.where(rng > 0, (close_arr - low_7d) / rng, 0.5)
    out["weekly_range_pos"] = np.clip(pos, 0.0, 1.0)

    # regime_vol_ratio: std(ret_1, 50) / median(std(ret_1), 200)
    ret_1 = np.log(b["close"].astype(float) / b["close"].astype(float).shift(1))
    std_50 = ret_1.rolling(50, min_periods=50).std()
    std_50_med_200 = std_50.rolling(200, min_periods=200).median()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["regime_vol_ratio"] = (std_50 / std_50_med_200).values

    for c in V2_HTF:
        out[c] = out[c].astype("float64")
    return out


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────
def build_v2_features(
    symbol: str = "BTC", threshold: float = 0.02,
    start_month: Optional[pd.Timestamp] = None,
    end_month: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Build v2 features (33 base + 5 orderflow + 3 HTF = 41).

    DR v3.0.16 recent-regime scope: pass start_month=2024-01-01 to
    restrict order flow computation to recent regime only. Older bars
    get NaN for the 5 order-flow columns (LightGBM handles natively).
    HTF features are always computed for the full range (cheap).
    """
    sym = symbol_short(symbol)
    feats_path = PROJECT_ROOT / f"data/storage/features/features_{sym}.parquet"
    LOG.info("loading existing features from %s", feats_path)
    feats = pd.read_parquet(feats_path)
    LOG.info("  base features: %d rows × %d cols", len(feats), feats.shape[1])

    LOG.info("loading bars (with OHLC + duration meta)...")
    bars_sql = f"""
        SELECT bar_id, bar_open_ts, bar_close_ts, open, high, low, close, n_trades
        FROM {bars_table(symbol)}
        WHERE threshold_pct = {threshold}
        ORDER BY bar_close_ts, bar_id
    """
    bars = pd.read_sql_query(bars_sql, get_engine())
    LOG.info("  bars: %d rows", len(bars))

    # Group A: order flow (date-range scoped if start_month set)
    LOG.info("=== computing order flow features (Group A) ===")
    of_raw = compute_orderflow_features(symbol, threshold,
                                         start_month=start_month, end_month=end_month)
    LOG.info("  raw orderflow agg: %d bars (date-range scoped: %s)",
             len(of_raw), "yes" if start_month else "no")
    of_derived = compute_orderflow_derived(of_raw, bars)
    LOG.info("  derived orderflow: %d rows × %d cols", len(of_derived), of_derived.shape[1])

    # Group B: HTF context (always full range — cheap)
    LOG.info("=== computing HTF features (Group B) ===")
    htf = compute_htf_v2(bars)
    LOG.info("  HTF: %d rows × %d cols", len(htf), htf.shape[1])

    # Merge: feats (33+keys) ⨝ of_derived (5+key) ⨝ htf (3+key) on bar_id
    LOG.info("merging into v2 features...")
    out = feats.merge(of_derived, on="bar_id", how="left")
    out = out.merge(htf, on="bar_id", how="left")
    LOG.info("  final: %d rows × %d cols", len(out), out.shape[1])

    expected_cols = 2 + 33 + 5 + 3  # keys + base + orderflow + htf
    assert out.shape[1] == expected_cols, f"expected {expected_cols} cols, got {out.shape[1]}"

    # Report orderflow coverage
    n_of_valid = out["taker_buy_ratio"].notna().sum()
    LOG.info("  orderflow coverage: %d / %d bars (%.1f%%)",
             n_of_valid, len(out), 100.0 * n_of_valid / len(out))

    return out


def write_v2_parquet(df: pd.DataFrame, symbol: str = "BTC") -> Path:
    sym = symbol_short(symbol)
    out_path = PROJECT_ROOT / f"data/storage/features/features_{sym}_v2.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    LOG.info("wrote %s  (%.2f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def sanity_report(df: pd.DataFrame, symbol: str = "BTC") -> None:
    sym = symbol_short(symbol)
    print(f"\n========== v2 Features Sanity Report ({sym.upper()}) ==========")
    print(f"Total rows:    {len(df)}")
    print(f"Total cols:    {df.shape[1]}")
    print(f"New features:  {V2_NEW}")
    print(f"\n--- New feature distributions ---")
    for c in V2_NEW:
        s = df[c].dropna()
        if len(s) == 0:
            print(f"  {c:<24} NO VALID ROWS")
            continue
        print(f"  {c:<24} min={s.min():>12.6f}  p50={s.median():>12.6f}  "
              f"max={s.max():>12.6f}  nan={(df[c].isna().sum()):>5d}")

    print(f"\n--- NaN counts ---")
    for c in V2_NEW:
        n = int(df[c].isna().sum())
        pct = 100.0 * n / max(len(df), 1)
        if n > 0:
            print(f"  {c:<24} {n:>5d} NaN ({pct:.2f}%)")
    print()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="features.v2_builder")
    p.add_argument("--symbol", default="BTC")
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--start-month", default=None,
                   help="DR v3.0.16 recent-regime scope: YYYY-MM-DD lower bound "
                        "for order flow computation. Older bars get NaN for "
                        "order flow features. e.g. 2024-01-01")
    p.add_argument("--end-month", default=None,
                   help="Upper bound (exclusive) for order flow. YYYY-MM-DD.")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start_m = pd.Timestamp(args.start_month, tz="UTC") if args.start_month else None
    end_m = pd.Timestamp(args.end_month, tz="UTC") if args.end_month else None

    try:
        df = build_v2_features(symbol=args.symbol, threshold=args.threshold,
                                start_month=start_m, end_month=end_m)
        path = write_v2_parquet(df, symbol=args.symbol)
        sanity_report(df, symbol=args.symbol)
        print(f"Parquet: {path}")
        return 0
    finally:
        close_pool()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
