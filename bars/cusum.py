"""CUSUM event-bar construction (spec §6.4 + DR v3.0.5).

Streams `events.ticks_btc` via COPY TO STDOUT (binary), runs the
Lessmann CUSUM algorithm in pure Python, force-closes any bar that
exceeds 168 h without a CUSUM trigger (§6.5 fail-safe), and bulk-INSERTs
into `events.bars_btc_cusum`. Single transaction: DELETE → COPY-stream
ticks → INSERT bars → commit. Per DR v3.0.5 §1, full rebuild semantics.

CLI:
    python -m bars.cusum                              # full rebuild
    python -m bars.cusum --dry-run                    # full read+compute, no DB write
    python -m bars.cusum --dry-run --month YYYY-MM    # smoke test on one month
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import yaml

from data.db import close_pool, get_connection, init_schema

LOG = logging.getLogger("bars.cusum")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────
# Bar dataclass — frozen for hashable equality (used by determinism test)
# ─────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class Bar:
    bar_open_ts: datetime
    bar_close_ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    n_trades: int
    cusum_pos: float
    cusum_neg: float


# ─────────────────────────────────────────────────────────────────────
# CUSUM algorithm (DR v3.0.5 §1, §5–§7)
# ─────────────────────────────────────────────────────────────────────
class CusumBuilder:
    """Stateful single-asset CUSUM bar builder.

    Per spec §6.4 algorithm with §6.5 168-h force-close fail-safe. Tests
    inspect `s_pos`/`s_neg`/`bar_open_ts` attributes to verify reset
    semantics after a CUSUM-triggered close.
    """

    __slots__ = (
        "threshold", "max_duration_s",
        "s_pos", "s_neg",
        "bar_open_ts", "bar_open_price",
        "bar_high", "bar_low",
        "bar_volume", "bar_n_trades",
        "last_price", "tick_index",
    )

    def __init__(self, threshold: float, max_duration_h: float = 168.0):
        self.threshold = threshold
        self.max_duration_s = max_duration_h * 3600.0
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.bar_open_ts: Optional[datetime] = None
        self.bar_open_price: Optional[float] = None
        self.bar_high: Optional[float] = None
        self.bar_low: Optional[float] = None
        self.bar_volume: float = 0.0
        self.bar_n_trades: int = 0
        self.last_price: Optional[float] = None
        self.tick_index: int = 0

    def step(self, ts: datetime, price: float, qty: float) -> Optional[Bar]:
        """Process one tick. Returns Bar if one closes, else None."""
        if self.bar_open_ts is None:
            self.bar_open_ts = ts
            self.bar_open_price = price
            self.bar_high = price
            self.bar_low = price
            self.bar_volume = qty
            self.bar_n_trades = 1
        else:
            if price > self.bar_high:
                self.bar_high = price
            if price < self.bar_low:
                self.bar_low = price
            self.bar_volume += qty
            self.bar_n_trades += 1

        if self.tick_index > 0:
            r = math.log(price / self.last_price)
            self.s_pos = max(0.0, self.s_pos + r)
            self.s_neg = min(0.0, self.s_neg + r)

        self.last_price = price
        self.tick_index += 1

        triggered = max(self.s_pos, -self.s_neg) >= self.threshold
        timeout = (ts - self.bar_open_ts).total_seconds() >= self.max_duration_s

        if triggered or timeout:
            bar = Bar(
                bar_open_ts=self.bar_open_ts,
                bar_close_ts=ts,
                open=self.bar_open_price,
                high=self.bar_high,
                low=self.bar_low,
                close=price,
                volume=self.bar_volume,
                n_trades=self.bar_n_trades,
                cusum_pos=self.s_pos,
                cusum_neg=self.s_neg,
            )
            self.s_pos = 0.0
            self.s_neg = 0.0
            self.bar_open_ts = None
            return bar
        return None


def cusum_bars(
    ticks: Iterable[tuple],
    threshold: float,
    max_duration_h: float = 168.0,
) -> Iterator[Bar]:
    """Generator producing bars from a tick iterable. Each tick is a
    (ts: datetime, price: float, qty: float) tuple."""
    builder = CusumBuilder(threshold, max_duration_h)
    for ts, price, qty in ticks:
        bar = builder.step(ts, price, qty)
        if bar is not None:
            yield bar


# ─────────────────────────────────────────────────────────────────────
# DB driver (DR v3.0.5 §1, §4)
# ─────────────────────────────────────────────────────────────────────
_BARS_INSERT_TYPES = [
    "timestamptz", "timestamptz",
    "float8", "float8", "float8", "float8",
    "float8", "int4",
    "float8", "float8",
    "float8",
]


def _bars_insert_copy_sql(symbol: str = "BTC") -> str:
    """COPY-FROM-STDIN target for bars table, parameterized by symbol (DR v3.0.14)."""
    from data.db import bars_table
    return (
        f"COPY {bars_table(symbol)} "
        "(bar_open_ts, bar_close_ts, open, high, low, close, volume, n_trades, "
        " cusum_pos, cusum_neg, threshold_pct) "
        "FROM STDIN WITH (FORMAT BINARY)"
    )


def _ticks_query(month_filter: Optional[date], symbol: str = "BTC") -> str:
    """Build the COPY-TO-STDOUT query body, optionally filtered to one month."""
    from data.db import ticks_table
    where = ""
    if month_filter is not None:
        ms = month_filter.isoformat()
        next_m = (
            date(month_filter.year + 1, 1, 1)
            if month_filter.month == 12
            else date(month_filter.year, month_filter.month + 1, 1)
        ).isoformat()
        where = f"WHERE ts >= '{ms}'::timestamptz AND ts < '{next_m}'::timestamptz "
    return (
        f"COPY (SELECT ts, price, qty FROM {ticks_table(symbol)} "
        f"{where}ORDER BY ts, agg_id) TO STDOUT (FORMAT BINARY)"
    )


def build_bars(
    threshold: float,
    max_duration_h: float = 168.0,
    month_filter: Optional[date] = None,
    dry_run: bool = False,
    symbol: str = "BTC",
) -> dict:
    """Build all bars for one threshold; return a stats dict.

    If `month_filter` is set, restrict the source ticks to that calendar
    month. If `dry_run` is True, do not touch the bars table — stream,
    compute, and report only. DR v3.0.14: `symbol` selects ticks/bars
    table.
    """
    LOG.info(
        "build_bars: symbol=%s threshold=%s max_duration_h=%s month=%s dry_run=%s",
        symbol, threshold, max_duration_h,
        f"{month_filter:%Y-%m}" if month_filter else "<all>",
        dry_run,
    )

    bars: list[Bar] = []
    builder = CusumBuilder(threshold, max_duration_h)
    n_ticks = 0
    prev_month: Optional[tuple[int, int]] = None
    month_ticks = 0
    month_bars = 0
    month_t0 = time.perf_counter()
    t0 = time.perf_counter()

    with get_connection() as conn:
        with conn.cursor() as cur:
            with cur.copy(_ticks_query(month_filter, symbol=symbol)) as copy:
                copy.set_types(["timestamptz", "float8", "float8"])
                for ts, price, qty in copy.rows():
                    cur_month = (ts.year, ts.month)
                    if prev_month is None:
                        prev_month = cur_month
                    elif cur_month != prev_month:
                        elapsed = time.perf_counter() - month_t0
                        LOG.info(
                            "[scan] %04d-%02d: ticks=%d processed, "
                            "bars=%d emitted, elapsed=%.1fs",
                            prev_month[0], prev_month[1],
                            month_ticks, month_bars, elapsed,
                        )
                        prev_month = cur_month
                        month_ticks = 0
                        month_bars = 0
                        month_t0 = time.perf_counter()
                    n_ticks += 1
                    month_ticks += 1
                    bar = builder.step(ts, price, qty)
                    if bar is not None:
                        bars.append(bar)
                        month_bars += 1
                # Emit final-month progress on EOF
                if prev_month is not None and month_ticks > 0:
                    elapsed = time.perf_counter() - month_t0
                    LOG.info(
                        "[scan] %04d-%02d: ticks=%d processed, "
                        "bars=%d emitted, elapsed=%.1fs",
                        prev_month[0], prev_month[1],
                        month_ticks, month_bars, elapsed,
                    )
        scan_s = time.perf_counter() - t0
        n_force_closed = sum(
            1 for b in bars
            if max(b.cusum_pos, -b.cusum_neg) < threshold
        )
        rate = int(n_ticks / scan_s) if scan_s > 0 else 0
        LOG.info(
            "[done] total ticks=%.2fB, total bars=%d, elapsed=%.1fs, "
            "force-closed=%d, throughput=%d ticks/sec",
            n_ticks / 1e9, len(bars), scan_s, n_force_closed, rate,
        )

        if not dry_run:
            from data.db import bars_table
            t1 = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {bars_table(symbol)} WHERE threshold_pct = %s",
                    (threshold,),
                )
                deleted = cur.rowcount
                with cur.copy(_bars_insert_copy_sql(symbol)) as cp:
                    cp.set_types(_BARS_INSERT_TYPES)
                    for b in bars:
                        cp.write_row((
                            b.bar_open_ts, b.bar_close_ts,
                            b.open, b.high, b.low, b.close,
                            b.volume, b.n_trades,
                            b.cusum_pos, b.cusum_neg,
                            threshold,
                        ))
            conn.commit()
            write_s = time.perf_counter() - t1
            LOG.info("DB write complete: deleted %d, inserted %d in %.1f s",
                     deleted, len(bars), write_s)

    return {
        "threshold": threshold,
        "n_ticks": n_ticks,
        "n_bars": len(bars),
        "scan_s": time.perf_counter() - t0,
        "bars_sample_first": bars[:3],
        "bars_sample_last": bars[-3:] if len(bars) > 3 else [],
        "all_bars": bars,
    }


def print_summary(stats: dict) -> None:
    print("\n========== bars/cusum.py summary ==========")
    print(f"threshold:      {stats['threshold']}")
    print(f"ticks scanned:  {stats['n_ticks']:,}")
    print(f"bars produced:  {stats['n_bars']:,}")
    print(f"scan time:      {stats['scan_s']:.1f}s")
    if stats['n_ticks'] > 0 and stats['scan_s'] > 0:
        rate = stats['n_ticks'] / stats['scan_s']
        print(f"throughput:     {rate:,.0f} ticks/sec")

    bars = stats["all_bars"]
    if bars:
        durations = [(b.bar_close_ts - b.bar_open_ts).total_seconds() for b in bars]
        n_timeout = sum(
            1 for b in bars
            if max(b.cusum_pos, -b.cusum_neg) < stats["threshold"]
        )
        print(f"\nbar duration (sec): "
              f"min={min(durations):.0f}  "
              f"median={sorted(durations)[len(durations)//2]:.0f}  "
              f"max={max(durations):.0f}")
        print(f"force-closed (timeout) bars: {n_timeout} of {len(bars)}")

    if stats["bars_sample_first"]:
        print("\nfirst 3 bars:")
        for b in stats["bars_sample_first"]:
            print(_format_bar(b, stats["threshold"]))
    if stats["bars_sample_last"]:
        print("\nlast 3 bars:")
        for b in stats["bars_sample_last"]:
            print(_format_bar(b, stats["threshold"]))


def _format_bar(b: Bar, threshold: float) -> str:
    duration = (b.bar_close_ts - b.bar_open_ts).total_seconds()
    trigger = "cusum" if max(b.cusum_pos, -b.cusum_neg) >= threshold else "timeout"
    return (
        f"  {b.bar_open_ts} → {b.bar_close_ts}  dur={duration:.0f}s  trigger={trigger}\n"
        f"     OHLC=({b.open:.2f}, {b.high:.2f}, {b.low:.2f}, {b.close:.2f})  "
        f"vol={b.volume:.4f} n={b.n_trades}  "
        f"S+={b.cusum_pos:+.4f} S-={b.cusum_neg:+.4f}"
    )


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────
def sanity_report(threshold: float, symbol: str = "BTC") -> None:
    """Phase 0.2 post-build sanity report (DR v3.0.5 + DR v3.0.14 symbol-aware)."""
    from data.db import bars_table
    t = bars_table(symbol)
    print(f"\n========== Phase 0.2 Sanity Report — symbol={symbol} ==========")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) AS n,
                       MIN(bar_open_ts)  AS first_open,
                       MAX(bar_close_ts) AS last_close
                FROM {t} WHERE threshold_pct = %s
            """, (threshold,))
            r = cur.fetchone()
            n_total = r["n"]
            print(f"Threshold:    {threshold}")
            print(f"Total bars:   {n_total:,}")
            print(f"First open:   {r['first_open']}")
            print(f"Last close:   {r['last_close']}")
            if n_total == 0:
                print("WARNING: zero bars in DB — investigate")
                return

            # Bars per month
            cur.execute(f"""
                SELECT date_trunc('month', bar_close_ts) AS m, COUNT(*) AS n
                FROM {t} WHERE threshold_pct = %s
                GROUP BY 1 ORDER BY 1
            """, (threshold,))
            month_rows = list(cur.fetchall())
            print(f"\n--- Bars per month ({len(month_rows)} months) ---")
            for r in month_rows:
                m = r["m"]
                next_m = (date(m.year + 1, 1, 1) if m.month == 12
                          else date(m.year, m.month + 1, 1))
                days = (next_m - m.date()).days
                print(f"  {m:%Y-%m}  bars={r['n']:>5}  /day={r['n']/days:>5.2f}")
            counts = [r["n"] for r in month_rows]
            print(f"  min={min(counts)}  max={max(counts)}  "
                  f"median={sorted(counts)[len(counts)//2]}")

            # Close-reason distribution
            cur.execute(f"""
                SELECT
                    CASE WHEN GREATEST(cusum_pos, -cusum_neg) >= threshold_pct
                         THEN 'cusum' ELSE 'timeout' END AS reason,
                    COUNT(*) AS n
                FROM {t} WHERE threshold_pct = %s
                GROUP BY 1
            """, (threshold,))
            print(f"\n--- Close-reason distribution ---")
            timeout_pct = 0.0
            for r in cur.fetchall():
                pct = r["n"] * 100.0 / n_total
                print(f"  {r['reason']:>10}: {r['n']:>6,}  ({pct:>5.2f}%)")
                if r["reason"] == "timeout":
                    timeout_pct = pct
            if timeout_pct > 5.0:
                print(f"  WARNING: timeout fraction {timeout_pct:.2f}% > 5% — "
                      f"threshold or fail-safe may be misconfigured")

            # Duration percentiles
            cur.execute(f"""
                SELECT EXTRACT(EPOCH FROM (bar_close_ts - bar_open_ts)) AS dur_s
                FROM {t} WHERE threshold_pct = %s
                ORDER BY 1
            """, (threshold,))
            durations = [r["dur_s"] for r in cur.fetchall()]
            mn = durations[0]
            med = durations[len(durations) // 2]
            p95 = durations[int(len(durations) * 0.95)]
            mx = durations[-1]
            print(f"\n--- Bar duration ---")
            print(f"  min:    {mn:>10,.0f}s  ({mn/60:>7.2f} min)")
            print(f"  median: {med:>10,.0f}s  ({med/3600:>7.2f} h)")
            print(f"  p95:    {p95:>10,.0f}s  ({p95/3600:>7.2f} h)")
            print(f"  max:    {mx:>10,.0f}s  ({mx/3600:>7.2f} h)")
            if mx > 168 * 3600 + 1:
                print(f"  ERROR: max duration > 168h fail-safe — algorithm bug")

            # Bar invariants
            cur.execute(f"""
                SELECT
                    SUM((high < GREATEST(open, close))::int) AS bad_high,
                    SUM((low  > LEAST(open, close))::int)    AS bad_low,
                    SUM((n_trades < 1)::int)                 AS bad_n,
                    SUM((volume <= 0)::int)                  AS bad_vol
                FROM {t} WHERE threshold_pct = %s
            """, (threshold,))
            inv = cur.fetchone()
            print(f"\n--- Bar invariants ---")
            print(f"  high < max(open,close): {inv['bad_high']}")
            print(f"  low  > min(open,close): {inv['bad_low']}")
            print(f"  n_trades < 1:           {inv['bad_n']}")
            print(f"  volume <= 0:            {inv['bad_vol']}")
            ok = (inv["bad_high"] == 0 and inv["bad_low"] == 0
                  and inv["bad_n"] == 0 and inv["bad_vol"] == 0)
            print(f"  ALL INVARIANTS OK: {ok}")

            # Determinism fingerprint — md5 over the canonical bar fields
            cur.execute(f"""
                SELECT md5(string_agg(
                    bar_open_ts::text  || '|' || bar_close_ts::text || '|' ||
                    open::text  || '|' || high::text  || '|' ||
                    low::text   || '|' || close::text || '|' ||
                    volume::text || '|' || n_trades::text || '|' ||
                    cusum_pos::text || '|' || cusum_neg::text,
                    chr(10) ORDER BY bar_id
                )) AS h
                FROM {t} WHERE threshold_pct = %s
            """, (threshold,))
            print(f"\n--- Determinism fingerprint ---")
            print(f"  md5(bars) = {cur.fetchone()['h']}")
            print(f"  (compare to a future re-run's value to verify identity)")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="bars.cusum")
    p.add_argument("--month", metavar="YYYY-MM",
                   help="Limit ticks to one calendar month (smoke / debug).")
    p.add_argument("--dry-run", action="store_true",
                   help="Stream + compute + print; no DB writes.")
    p.add_argument("--symbol", default="BTC",
                   help="DR v3.0.14: asset symbol (BTC|ETH). Default BTC.")
    p.add_argument("--threshold", type=float, default=None,
                   help="DR v3.0.20: override CUSUM threshold (e.g., 0.01, 0.015). "
                        "Default reads from config.yaml bars.threshold.{SYM}.")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from data.db import symbol_short
    sym = symbol_short(args.symbol).upper()

    cfg = load_config()
    # Threshold: CLI override > per-symbol > BTC default
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = cfg["bars"]["threshold"].get(sym, cfg["bars"]["threshold"]["BTC"])
    max_duration_h = cfg["bars"]["max_bar_duration_hours"]

    init_schema(
        chunk_interval_ticks=cfg["database"]["chunk_interval_ticks"],
        compress_after_ticks=cfg["database"]["compress_after_ticks"],
        chunk_interval_bars=cfg["database"]["chunk_interval_bars"],
        symbols=[sym],
    )

    month_filter: Optional[date] = None
    if args.month:
        month_filter = date.fromisoformat(args.month + "-01")

    try:
        stats = build_bars(
            threshold=threshold,
            max_duration_h=max_duration_h,
            month_filter=month_filter,
            dry_run=args.dry_run,
            symbol=sym,
        )
        print_summary(stats)
        if not args.dry_run:
            sanity_report(threshold, symbol=sym)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
