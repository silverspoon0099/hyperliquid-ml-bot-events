"""Feature builder for Phase 0.3 (spec §7.1 + DR v3.0.7).

Reads bars from `events.bars_btc_cusum`, computes Lessmann's 33-feature
set, and writes a parquet to `data/storage/features/features_btc.parquet`.

Indicators are implemented directly in pandas + numpy per DR v3.0.7 §10
(pandas-ta unavailable for Python 3.10). Canonical formulas:
  - EMA: α = 2/(N+1), pandas ewm(adjust=False)
  - std: rolling(N).std() of close
  - MACD: EMA(fast) − EMA(slow), signal = EMA(macd, 9), hist = line − signal
  - RSI: Wilder's RMA (α = 1/N), SMA(N) initialization at index N
  - Stoch %K, %D: 100·(c−LL)/(HH−LL) over 14, smoothed by 3, then SMA(3)
  - Williams %R: −100·(HH−c)/(HH−LL) over 14
  - Bollinger: SMA(5) ± 2.0·std(5)
  - CMF: sum(MFV, 21) / sum(volume, 21)
  - MFI: 100 − 100/(1 + pos_sum/neg_sum)
  - sin/cos seasonality on bar_close_ts hour ∈ [0,23] and weekday ∈ [0,6]

CLI:
    python -m features.builder                  # full build
    python -m features.builder --dry-run        # in-memory, no file write
    python -m features.builder --month YYYY-MM  # smoke (single month)
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

LOG = logging.getLogger("features.builder")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────
# Indicator functions (DR v3.0.7 §10 — hand-rolled)
# ─────────────────────────────────────────────────────────────────────
def ema(close: pd.Series, length: int) -> pd.Series:
    """EMA with α = 2/(N+1). y[0] = x[0]; y[i] = α·x[i] + (1−α)·y[i−1]."""
    return close.ewm(alpha=2.0 / (length + 1), adjust=False).mean()


def stddev(close: pd.Series, length: int) -> pd.Series:
    """Rolling sample std of close. NaN for first length−1 indices."""
    return close.rolling(length).std()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal, histogram."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI. NaN for first `length` indices.

    Initialization: avg_gain[length] = SMA of gain[1..length]; same for
    avg_loss. Then Wilder RMA: avg[i] = (avg[i-1]·(N-1) + value[i]) / N.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    n = len(close)
    avg_gain = pd.Series(np.nan, index=close.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=close.index, dtype=float)

    if n > length:
        avg_gain.iloc[length] = gain.iloc[1:length + 1].mean()
        avg_loss.iloc[length] = loss.iloc[1:length + 1].mean()
        prev_g = avg_gain.iloc[length]
        prev_l = avg_loss.iloc[length]
        for i in range(length + 1, n):
            prev_g = (prev_g * (length - 1) + gain.iloc[i]) / length
            prev_l = (prev_l * (length - 1) + loss.iloc[i]) / length
            avg_gain.iloc[i] = prev_g
            avg_loss.iloc[i] = prev_l

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, k_smooth: int = 3, d_smooth: int = 3,
):
    """Slow stochastic %K (smoothed) and %D (SMA of %K)."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    raw_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_k = raw_k.rolling(k_smooth).mean()
    stoch_d = stoch_k.rolling(d_smooth).mean()
    return stoch_k, stoch_d


def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14,
) -> pd.Series:
    """Williams %R = −100·(HH − close) / (HH − LL) over N."""
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100.0 * (hh - close) / (hh - ll)


def bollinger(close: pd.Series, length: int = 5, std_mult: float = 2.0):
    """Bollinger upper/lower bands around SMA(N). Middle band dropped."""
    sma = close.rolling(length).mean()
    s = close.rolling(length).std()
    return sma + std_mult * s, sma - std_mult * s


def log_return_1(close: pd.Series) -> pd.Series:
    """1-bar log return. NaN at index 0."""
    return np.log(close / close.shift(1))


def cmf(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    length: int = 21,
) -> pd.Series:
    """Chaikin Money Flow over N. MFM = 0 when high == low (no range)."""
    hl_range = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    mfm = mfm.fillna(0.0)
    mfv = mfm * volume
    return mfv.rolling(length).sum() / volume.rolling(length).sum()


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Money Flow Index over N.

    Edge cases (let inf and NaN propagate through the formula):
      - all-up window (neg_sum=0, pos_sum>0): mr=inf → MFI=100
      - all-down window (pos_sum=0, neg_sum>0): mr=0   → MFI=0
      - no-moves window (both 0):               mr=NaN → MFI=NaN
    """
    typical = (high + low + close) / 3.0
    money_flow = typical * volume
    direction = typical.diff()
    pos_flow = money_flow.where(direction > 0, 0.0)
    neg_flow = money_flow.where(direction < 0, 0.0)
    pos_sum = pos_flow.rolling(length).sum()
    neg_sum = neg_flow.rolling(length).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        mr = pos_sum / neg_sum
    return 100.0 - (100.0 / (1.0 + mr))


# ─────────────────────────────────────────────────────────────────────
# Build features (DR v3.0.7 §5 — exact 33-feature set + 2 keys)
# ─────────────────────────────────────────────────────────────────────
EMA_PERIODS = [5, 10, 15, 20, 50]
STD_PERIODS = [5, 10, 15, 20, 50]
RSI_PERIODS = [6, 10, 14]

FEATURE_COLS = (
    [f"ema_{n}" for n in EMA_PERIODS]
    + [f"std_{n}" for n in STD_PERIODS]
    + ["macd_line", "macd_signal", "macd_hist"]
    + [f"rsi_{n}" for n in RSI_PERIODS]
    + ["stoch_k", "stoch_d"]
    + ["williams_r"]
    + ["bb_upper", "bb_lower"]
    + ["ret_1"]
    + ["cmf_21", "mfi_14"]
    + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    + ["bar_duration_sec"]
    + ["n_trades", "volume", "cusum_pos", "cusum_neg"]
)
KEY_COLS = ["bar_id", "bar_close_ts"]
ALL_COLS = KEY_COLS + FEATURE_COLS

assert len(FEATURE_COLS) == 33, f"feature count {len(FEATURE_COLS)} != 33"


def build_features(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Build 35-column feature matrix from bars (sorted by bar_close_ts, bar_id)."""
    h = bars_df["high"].astype(float)
    l = bars_df["low"].astype(float)
    c = bars_df["close"].astype(float)
    v = bars_df["volume"].astype(float)

    out = pd.DataFrame(index=bars_df.index)
    out["bar_id"] = bars_df["bar_id"].astype("int64")
    out["bar_close_ts"] = pd.to_datetime(bars_df["bar_close_ts"], utc=True)

    for n in EMA_PERIODS:
        out[f"ema_{n}"] = ema(c, n)
    for n in STD_PERIODS:
        out[f"std_{n}"] = stddev(c, n)

    macd_line, macd_signal, macd_hist = macd(c, 12, 26, 9)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    for n in RSI_PERIODS:
        out[f"rsi_{n}"] = rsi(c, n)

    stoch_k_s, stoch_d_s = stochastic(h, l, c, 14, 3, 3)
    out["stoch_k"] = stoch_k_s
    out["stoch_d"] = stoch_d_s

    out["williams_r"] = williams_r(h, l, c, 14)

    bb_u, bb_l = bollinger(c, 5, 2.0)
    out["bb_upper"] = bb_u
    out["bb_lower"] = bb_l

    out["ret_1"] = log_return_1(c)
    out["cmf_21"] = cmf(h, l, c, v, 21)
    out["mfi_14"] = mfi(h, l, c, v, 14)

    ts = pd.to_datetime(bars_df["bar_close_ts"], utc=True)
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.weekday.astype(float)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    open_ts = pd.to_datetime(bars_df["bar_open_ts"], utc=True)
    out["bar_duration_sec"] = (ts - open_ts).dt.total_seconds().astype(float)

    out["n_trades"] = bars_df["n_trades"].astype(float)
    out["volume"] = v
    out["cusum_pos"] = bars_df["cusum_pos"].astype(float)
    out["cusum_neg"] = bars_df["cusum_neg"].astype(float)

    out[FEATURE_COLS] = out[FEATURE_COLS].astype("float64")
    out = out[ALL_COLS]  # enforce column order
    return out


# ─────────────────────────────────────────────────────────────────────
# DB load + parquet write
# ─────────────────────────────────────────────────────────────────────
def _load_bars(
    threshold: float, month_filter: Optional[date] = None,
    symbol: str = "BTC",
) -> pd.DataFrame:
    """Load bars from events.bars_{sym}_cusum (DR v3.0.14 multi-asset)."""
    from data.db import bars_table
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
        SELECT bar_id, bar_open_ts, bar_close_ts,
               open, high, low, close, volume, n_trades,
               cusum_pos, cusum_neg
        FROM {bars_table(symbol)}
        {where}
        ORDER BY bar_close_ts, bar_id
    """
    return pd.read_sql_query(sql, get_engine(), params=params)


def _md5_dataframe(df: pd.DataFrame) -> str:
    """Canonical md5 of a DataFrame: sort by bar_id, text-serialize, hash.
    Mirrors DR v3.0.6's bar fingerprint pattern."""
    sorted_df = df.sort_values("bar_id").reset_index(drop=True)
    h = hashlib.md5()
    for col in sorted_df.columns:
        h.update(col.encode("utf-8"))
        h.update(b"\x00")
        h.update(sorted_df[col].astype(str).str.cat(sep="\n").encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────
# Sanity report (DR v3.0.7 §12)
# ─────────────────────────────────────────────────────────────────────
def sanity_report(df: pd.DataFrame, parquet_path: Optional[Path] = None) -> dict:
    print("\n========== Phase 0.3 Sanity Report ==========")
    print(f"Rows:           {len(df):,}")
    print(f"Columns:        {len(df.columns)}  (expect 35)")
    print(f"First bar_id:   {df['bar_id'].iloc[0] if len(df) else '(empty)'}")
    print(f"Last  bar_id:   {df['bar_id'].iloc[-1] if len(df) else '(empty)'}")
    print(f"Time range:     {df['bar_close_ts'].min()}  →  {df['bar_close_ts'].max()}")

    # Column order check
    assert list(df.columns) == ALL_COLS, (
        f"column order mismatch:\n  got: {list(df.columns)}\n  expected: {ALL_COLS}"
    )

    # NaN density
    print("\n--- NaN counts per feature column ---")
    nan_per_col = df[FEATURE_COLS].isna().sum()
    nonzero = nan_per_col[nan_per_col > 0].sort_values(ascending=False)
    if len(nonzero) == 0:
        print("  (none — all features dense from row 0)")
    else:
        for col, n in nonzero.items():
            print(f"  {col:<20} {n:>6}")
    zero_count = int((nan_per_col == 0).sum())
    print(f"  ({zero_count} of {len(FEATURE_COLS)} columns have 0 NaN)")
    print(f"  total NaN cells: {int(nan_per_col.sum())}")

    # Explicit warmup assertion (DR v3.0.7 §12 fold)
    if len(df) > 50:
        post_warmup_nan = df[FEATURE_COLS].iloc[50:].isna().sum().sum()
        print(f"\nNaN total in df[FEATURE_COLS].iloc[50:]: {post_warmup_nan}")
        assert post_warmup_nan == 0, (
            f"DR v3.0.7 §12 violation: {post_warmup_nan} NaN values past row 50"
        )
        print("  ✓ NaN density drops to 0 by row 50 (DR v3.0.7 §12)")

    # Feature range plausibility (DR v3.0.7 §12)
    # Tolerance handles float-boundary cases (e.g. close exactly at the
    # rolling HH or LL gives williams_r = 0 or -100 ± float64 epsilon).
    print("\n--- Feature range checks ---")
    eps = 1e-9
    for col in ["rsi_6", "rsi_10", "rsi_14"]:
        v = df[col].dropna()
        ok = (v >= -eps).all() and (v <= 100.0 + eps).all()
        print(f"  {col:<14} ∈ [0, 100]:  {ok}  (min={v.min():.6f}, max={v.max():.6f})")
        assert ok
    v = df["williams_r"].dropna()
    ok = (v >= -100.0 - eps).all() and (v <= eps).all()
    print(f"  williams_r     ∈ [-100, 0]: {ok}  (min={v.min():.6f}, max={v.max():.6f})")
    assert ok
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        v = df[col]
        ok = (v >= -1.0 - 1e-12).all() and (v <= 1.0 + 1e-12).all()
        print(f"  {col:<14} ∈ [-1, 1]:    {ok}  (min={v.min():.4f}, max={v.max():.4f})")
        assert ok

    # Per-feature min / max (full matrix) — degenerate-range catch
    print("\n--- Per-feature min / max (full matrix) ---")
    for col in FEATURE_COLS:
        s = df[col].dropna()
        if len(s) == 0:
            print(f"  {col:<20} (all NaN)")
            continue
        mn, mx = float(s.min()), float(s.max())
        flag = "  <-- DEGENERATE (min == max)" if mn == mx else ""
        print(f"  {col:<20} min={mn:>16.6g}  max={mx:>16.6g}{flag}")

    # Determinism fingerprint
    fp = _md5_dataframe(df)
    print(f"\n--- Determinism fingerprint ---")
    print(f"  md5(features) = {fp}")
    print(f"  (compare to a future re-run's value to verify identity)")

    if parquet_path is not None:
        size_mb = parquet_path.stat().st_size / 1e6
        print(f"\nParquet:        {parquet_path}  ({size_mb:.2f} MB)")

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "fingerprint": fp,
    }


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────
def run_build(
    threshold: float,
    month_filter: Optional[date] = None,
    dry_run: bool = False,
    symbol: str = "BTC",
) -> dict:
    """Build features for one symbol; writes features_{sym}.parquet (DR v3.0.14)."""
    from data.db import symbol_short
    sym = symbol_short(symbol)
    cfg = load_config()
    output_dir = PROJECT_ROOT / cfg["features"]["output_dir"]
    output_path = output_dir / f"features_{sym}.parquet"

    LOG.info("loading bars: symbol=%s threshold=%s month=%s",
             symbol, threshold,
             f"{month_filter:%Y-%m}" if month_filter else "<all>")
    t0 = time.perf_counter()
    bars = _load_bars(threshold, month_filter, symbol=symbol)
    load_s = time.perf_counter() - t0
    LOG.info("loaded %d bars in %.1fs", len(bars), load_s)

    if len(bars) == 0:
        raise RuntimeError("No bars matched the filter — refusing to write empty parquet")

    LOG.info("building features...")
    t1 = time.perf_counter()
    df = build_features(bars)
    build_s = time.perf_counter() - t1
    LOG.info("built %d × %d features in %.1fs", len(df), len(df.columns), build_s)

    parquet_path = None
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        parquet_path = output_path
        LOG.info("wrote parquet: %s (%.2f MB)",
                 output_path, output_path.stat().st_size / 1e6)
    else:
        LOG.info("dry-run: skipping parquet write")

    rep = sanity_report(df, parquet_path)
    rep["bars_loaded"] = len(bars)
    rep["load_s"] = load_s
    rep["build_s"] = build_s
    return rep


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="features.builder")
    p.add_argument("--month", metavar="YYYY-MM",
                   help="Limit to one calendar month (smoke / debug).")
    p.add_argument("--dry-run", action="store_true",
                   help="Build in memory + report; do not write parquet.")
    p.add_argument("--symbol", default="BTC",
                   help="DR v3.0.14: asset symbol (BTC|ETH). Default BTC.")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from data.db import symbol_short
    sym = symbol_short(args.symbol).upper()

    cfg = load_config()
    threshold = cfg["bars"]["threshold"].get(sym, cfg["bars"]["threshold"]["BTC"])

    month_filter: Optional[date] = None
    if args.month:
        month_filter = date.fromisoformat(args.month + "-01")

    try:
        run_build(threshold=threshold, month_filter=month_filter,
                  dry_run=args.dry_run, symbol=sym)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
