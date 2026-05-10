"""Binance Vision aggTrades → Postgres+TimescaleDB.

Implements the Phase 0.1 ingestion contract per DR v3.0.2:
  §1   target table  events.ticks_btc
  §1b  audit table   events.ingest_log
  §3   per-month idempotency: SHA256 from .CHECKSUM + agg_id PK + atomic
       (DELETE range + COPY + INSERT log) within one transaction
  §3   loop bound:   complete months only (strictly < current UTC month)
  §4   sanity:       minute-aggregated zero-volume gaps > 1 hr

Source CSV columns (headerless or header-row in 2025+ archives):
    a, p, q, f, l, T, m, M
    agg_id, price, qty, first_trade_id, last_trade_id,
    timestamp_(ms|us), is_buyer_maker, was_best_match
The 8th column (was_best_match) is dropped. quote_qty is computed as
price * qty per DR §1 (Binance does not publish it for aggTrades).

Run:
    python -m data.ingest_ticks                       # full Phase 0.1 ingest
    python -m data.ingest_ticks --month YYYY-MM       # single-month smoke test
    python -m data.ingest_ticks --sanity              # sanity checks only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import requests
import yaml

from .db import close_pool, get_connection, init_schema, symbol_short

LOG = logging.getLogger("ingest_ticks")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

BINANCE_BASE = "https://data.binance.vision/data/spot/monthly/aggTrades"


# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────
# Month iteration (DR §3 — complete months only)
# ─────────────────────────────────────────────────────────────────────────
def _next_month(d: date) -> date:
    return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)


def months_to_ingest(start: date, today_utc: date) -> list[date]:
    """First-of-month dates from `start` up to (but not including) the
    current UTC calendar month. The in-flight month is never ingested."""
    out: list[date] = []
    cur = date(start.year, start.month, 1)
    cutoff = date(today_utc.year, today_utc.month, 1)
    while cur < cutoff:
        out.append(cur)
        cur = _next_month(cur)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Binance Vision fetch + checksum
# ─────────────────────────────────────────────────────────────────────────
def _zip_url(symbol: str, month: date) -> str:
    return f"{BINANCE_BASE}/{symbol}/{symbol}-aggTrades-{month:%Y-%m}.zip"


def _checksum_url(symbol: str, month: date) -> str:
    return f"{BINANCE_BASE}/{symbol}/{symbol}-aggTrades-{month:%Y-%m}.zip.CHECKSUM"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def _published_sha256(checksum_path: Path) -> str:
    """Binance .CHECKSUM is a single line: '<sha256>  <filename>'."""
    line = checksum_path.read_text().strip()
    sha = line.split()[0]
    if len(sha) != 64:
        raise ValueError(f"Bad CHECKSUM in {checksum_path}: {line!r}")
    return sha


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────
# CSV streaming
# ─────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Tick:
    agg_id: int
    ts: datetime
    price: float
    qty: float
    quote_qty: float
    is_buyer_maker: bool
    first_trade_id: int
    last_trade_id: int


def _parse_ts(raw: str) -> datetime:
    """Auto-detect ms vs µs (Binance Vision switched to µs in 2025)."""
    n = int(raw)
    secs = n / 1_000_000.0 if n > 10**14 else n / 1_000.0
    return datetime.fromtimestamp(secs, tz=timezone.utc)


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in ("true", "1")


def _open_csv(zip_path: Path):
    """Open the canonical CSV inside the Binance Vision aggTrades zip.

    DR v3.0.3: some Binance archives ship the CSV twice — once at the
    root and once at a nested `fsx-data/...` packaging-artifact path
    (the two copies are byte-identical, verified via CRC32). Select the
    root-level match; fall back to nested only if there is no root
    match. Hard-fail if the expected name is absent entirely.
    """
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    expected = zip_path.stem + ".csv"
    candidates = [n for n in names
                  if n == expected or n.endswith("/" + expected)]
    root = [n for n in candidates if "/" not in n]
    chosen = root[0] if root else (candidates[0] if candidates else None)
    if chosen is None:
        zf.close()
        raise ValueError(
            f"No CSV named {expected} in {zip_path}; archive contains {names}"
        )
    if len(candidates) > 1:
        # Surface the multi-CSV case inline (DR v3.0.3 §Decision):
        # one log line per affected month, no separate sanity pass.
        parts = zip_path.stem.split("-aggTrades-")
        sym = parts[0] if len(parts) == 2 else "?"
        mon = parts[1] if len(parts) == 2 else "?"
        others = [n for n in candidates if n != chosen]
        LOG.info(
            "[%s %s] zip contains %d CSVs; using %s, others: %s",
            sym, mon, len(candidates), chosen, others,
        )
    return zf, zf.open(chosen)


def _iter_ticks(zip_path: Path) -> Iterator[Tick]:
    zf, raw = _open_csv(zip_path)
    try:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        reader = csv.reader(text)
        for row in reader:
            try:
                agg_id = int(row[0])
            except ValueError:
                continue  # header line
            price = float(row[1])
            qty = float(row[2])
            yield Tick(
                agg_id=agg_id,
                ts=_parse_ts(row[5]),
                price=price,
                qty=qty,
                quote_qty=price * qty,
                is_buyer_maker=_parse_bool(row[6]),
                first_trade_id=int(row[3]),
                last_trade_id=int(row[4]),
            )
    finally:
        raw.close()
        zf.close()


def _count_data_rows(zip_path: Path) -> int:
    """`wc -l` of the unzipped CSV minus header (if present). DR §3 step 3."""
    n = 0
    zf, raw = _open_csv(zip_path)
    try:
        for i, line in enumerate(raw):
            if i == 0:
                first = line.lstrip()[:1]
                if first and not (b"0" <= first <= b"9"):
                    continue  # header line
            if line.strip():
                n += 1
    finally:
        raw.close()
        zf.close()
    return n


# ─────────────────────────────────────────────────────────────────────────
# DB loading (DR §3 — single transaction per month)
# ─────────────────────────────────────────────────────────────────────────
# DR v3.0.4: stage in TEMP without PK, then INSERT…ON CONFLICT into target.
# `LIKE events.ticks_btc` copies columns + NOT NULL but NOT the PK constraint,
# so staging accepts duplicate (agg_id, ts) rows. The target's PK then drops
# the second occurrence silently via ON CONFLICT DO NOTHING.
_CREATE_STAGING = (
    "CREATE TEMP TABLE _staging_ticks (LIKE events.ticks_btc) ON COMMIT DROP"
)

_COPY_STAGING_SQL = (
    "COPY _staging_ticks "
    "(agg_id, ts, price, qty, quote_qty, is_buyer_maker, first_trade_id, last_trade_id) "
    "FROM STDIN WITH (FORMAT BINARY)"
)

# DR v3.0.14: symbol-parameterized INSERT (target table per symbol).
def _insert_from_staging_sql(symbol: str) -> str:
    from data.db import ticks_table
    return f"""
INSERT INTO {ticks_table(symbol)}
    (agg_id, ts, price, qty, quote_qty, is_buyer_maker,
     first_trade_id, last_trade_id)
SELECT agg_id, ts, price, qty, quote_qty, is_buyer_maker,
       first_trade_id, last_trade_id
FROM _staging_ticks
ON CONFLICT (agg_id, ts) DO NOTHING;
"""


_COPY_TYPES = ["bigint", "timestamptz", "float8", "float8", "float8",
               "bool", "bigint", "bigint"]

_LOG_UPSERT = """
INSERT INTO events.ingest_log
    (symbol, month, sha256, expected_rows, actual_rows)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (symbol, month) DO UPDATE SET
    sha256        = EXCLUDED.sha256,
    expected_rows = EXCLUDED.expected_rows,
    actual_rows   = EXCLUDED.actual_rows,
    ingested_at   = now();
"""


def _existing_count(cur, symbol: str, month: date) -> int:
    from data.db import ticks_table
    cur.execute(
        f"SELECT COUNT(*) AS n FROM {ticks_table(symbol)} WHERE ts >= %s AND ts < %s",
        (month, _next_month(month)),
    )
    return cur.fetchone()["n"]


def _logged_sha(cur, symbol: str, month: date) -> str | None:
    cur.execute(
        "SELECT sha256 FROM events.ingest_log WHERE symbol=%s AND month=%s",
        (symbol, month),
    )
    row = cur.fetchone()
    return row["sha256"] if row else None


def _ingest_month_atomic(
    symbol: str,
    month: date,
    zip_path: Path,
    sha256: str,
    expected_rows: int,
    ticks: Iterable[Tick],
) -> int:
    """DR v3.0.2 §3 step 8 + DR v3.0.4: DELETE range → COPY into TEMP
    staging → INSERT…ON CONFLICT into target → audit log. Single tx.
    Returns post-dedup row count (may be < expected_rows if the source
    CSV contained Binance publishing duplicates).
    """
    from data.db import ticks_table
    target_table = ticks_table(symbol)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {target_table} WHERE ts >= %s AND ts < %s",
                (month, _next_month(month)),
            )
            cur.execute(_CREATE_STAGING)
            with cur.copy(_COPY_STAGING_SQL) as cp:
                cp.set_types(_COPY_TYPES)
                for t in ticks:
                    cp.write_row((
                        t.agg_id, t.ts, t.price, t.qty, t.quote_qty,
                        t.is_buyer_maker, t.first_trade_id, t.last_trade_id,
                    ))
            cur.execute(_insert_from_staging_sql(symbol))
            actual = cur.rowcount
            cur.execute(_LOG_UPSERT, (symbol, month, sha256, expected_rows, actual))
        conn.commit()
    return actual


# ─────────────────────────────────────────────────────────────────────────
# Per-month orchestration
# ─────────────────────────────────────────────────────────────────────────
def ingest_one_month(symbol: str, month: date, storage_dir: Path) -> dict:
    """Idempotent ingest of one (symbol, month). Returns a status dict
    with phase timings: download_s, sha_s, count_s, copy_s."""
    yyyymm = f"{month:%Y-%m}"
    zip_name = f"{symbol}-aggTrades-{yyyymm}.zip"
    zip_path = storage_dir / symbol / zip_name
    sha_path = storage_dir / symbol / (zip_name + ".CHECKSUM")

    # 1. download (skip if cached)
    t0 = time.perf_counter()
    if not zip_path.exists():
        _download(_zip_url(symbol, month), zip_path)
    if not sha_path.exists():
        _download(_checksum_url(symbol, month), sha_path)
    download_s = time.perf_counter() - t0

    # 2. verify SHA256
    t0 = time.perf_counter()
    published = _published_sha256(sha_path)
    actual_sha = _file_sha256(zip_path)
    if published != actual_sha:
        raise RuntimeError(
            f"[{symbol} {yyyymm}] SHA mismatch: published={published} "
            f"actual={actual_sha} — refusing to ingest"
        )
    sha_s = time.perf_counter() - t0

    # 3. expected row count
    t0 = time.perf_counter()
    expected = _count_data_rows(zip_path)
    count_s = time.perf_counter() - t0

    # 4-7. skip / re-ingest decision
    with get_connection() as conn:
        with conn.cursor() as cur:
            existing = _existing_count(cur, symbol, month)
            logged = _logged_sha(cur, symbol, month)

    base = {
        "month": yyyymm, "expected": expected, "sha256": published,
        "download_s": download_s, "sha_s": sha_s, "count_s": count_s,
    }

    # DR v3.0.4: SHA-only skip rule. Atomicity guarantees no partial COPY
    # ever lands in ingest_log, so a logged sha match implies a complete
    # prior ingest. With dedup, post-dedup count < raw expected for some
    # months, so the old "existing >= expected" check would force forever
    # re-ingestion.
    if logged is not None and logged == published:
        return {**base, "status": "skipped", "actual": existing, "copy_s": 0.0}

    if logged is not None and logged != published:
        LOG.warning("[%s %s] sha changed (logged=%s published=%s) — re-ingest",
                    symbol, yyyymm, logged, published)

    # 8. atomic: DELETE + COPY into staging + INSERT into target + log
    t0 = time.perf_counter()
    actual = _ingest_month_atomic(
        symbol, month, zip_path, published, expected, _iter_ticks(zip_path)
    )
    copy_s = time.perf_counter() - t0
    if actual > expected:
        raise RuntimeError(
            f"[{symbol} {yyyymm}] row count overshoot: expected={expected} "
            f"loaded={actual} — impossible by construction, investigate"
        )
    if actual < expected:
        LOG.info("[%s %s] dedup: %d duplicate rows in source CSV "
                 "(kept %d / %d)",
                 symbol, yyyymm, expected - actual, actual, expected)
    return {**base, "status": "ingested", "actual": actual, "copy_s": copy_s}


# ─────────────────────────────────────────────────────────────────────────
# Sanity checks (DR §4 + Appendix A)
# ─────────────────────────────────────────────────────────────────────────
def sanity_checks(symbol: str = "BTC") -> dict:
    """Returns a report dict; raises on hard failures. DR v3.0.14: symbol-aware."""
    from data.db import ticks_table
    t = ticks_table(symbol)
    out: dict = {"symbol": symbol}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts
                FROM {t}
            """)
            r = cur.fetchone()
            out["total_ticks"] = r["n"]
            out["first_ts"] = r["first_ts"]
            out["last_ts"] = r["last_ts"]

            cur.execute(f"""
                SELECT date_trunc('month', ts) AS m, COUNT(*) AS n
                FROM {t}
                GROUP BY 1 ORDER BY 1
            """)
            out["ticks_per_month"] = [(r["m"], r["n"]) for r in cur.fetchall()]

            cur.execute(f"""
                SELECT date_trunc('day', ts) AS d, SUM(quote_qty) AS quote_vol
                FROM {t}
                GROUP BY 1 ORDER BY 1
            """)
            out["daily_quote_volume"] = [(r["d"], r["quote_vol"]) for r in cur.fetchall()]

            # DR §4: minute-aggregated zero-activity gaps > 1 hr
            cur.execute(f"""
                WITH minute_bins AS (
                    SELECT date_trunc('minute', ts) AS m
                    FROM {t}
                    GROUP BY 1
                ),
                gaps AS (
                    SELECT m, m - LAG(m) OVER (ORDER BY m) AS gap
                    FROM minute_bins
                )
                SELECT m, gap FROM gaps WHERE gap > INTERVAL '1 hour'
                ORDER BY m
            """)
            out["activity_gaps_gt_1h"] = [(r["m"], r["gap"]) for r in cur.fetchall()]

            # agg_id monotonicity per month
            cur.execute(f"""
                SELECT date_trunc('month', ts) AS m,
                       MIN(agg_id) AS min_id, MAX(agg_id) AS max_id, COUNT(*) AS n
                FROM {t}
                GROUP BY 1 ORDER BY 1
            """)
            out["agg_id_per_month"] = [
                (r["m"], r["min_id"], r["max_id"], r["n"]) for r in cur.fetchall()
            ]

            # (d) agg_id global span vs count
            cur.execute(f"""
                SELECT MIN(agg_id) AS mn, MAX(agg_id) AS mx, COUNT(*) AS n,
                       MAX(agg_id) - MIN(agg_id) + 1 AS span
                FROM {t}
            """)
            out["agg_id_global"] = dict(cur.fetchone())

            # ingest_log audit
            cur.execute("""
                SELECT COUNT(*) AS n_rows,
                       COUNT(DISTINCT sha256) AS n_unique_sha,
                       SUM((sha256 IS NULL)::int) AS n_null
                FROM events.ingest_log
            """)
            out["ingest_log_audit"] = dict(cur.fetchone())

            # DR v3.0.4: source-dupe diagnostic — Binance publishing artifacts
            cur.execute("""
                SELECT month, expected_rows, actual_rows,
                       expected_rows - actual_rows AS source_dupes
                FROM events.ingest_log
                WHERE actual_rows < expected_rows
                ORDER BY month
            """)
            out["source_dupes"] = [dict(r) for r in cur.fetchall()]
    return out


def print_sanity_report(rep: dict) -> None:
    print("\n========== Phase 0.1 Sanity Report ==========")
    print(f"Total ticks:    {rep['total_ticks']:,}")
    print(f"First ts:       {rep['first_ts']}")
    print(f"Last ts:        {rep['last_ts']}")
    print(f"Months covered: {len(rep['ticks_per_month'])}")

    # (a) Per-month tick counts
    print("\n--- (a) Per-month tick counts ---")
    for m, n in rep["ticks_per_month"]:
        print(f"  {m:%Y-%m}  {n:>14,}")
    counts = [n for _, n in rep["ticks_per_month"]]
    if counts:
        median = sorted(counts)[len(counts) // 2]
        print(f"  min={min(counts):,}  max={max(counts):,}  median={median:,}")

    # (b) Daily volume — top-20 highest by quote volume
    print("\n--- (b) Top-20 daily quote-volume days ---")
    daily = rep["daily_quote_volume"]
    top = sorted(daily, key=lambda x: -float(x[1] or 0))[:20]
    for d, q in top:
        print(f"  {d:%Y-%m-%d}  ${float(q or 0):>18,.0f}")
    print(f"  total days with ticks: {len(daily):,}")

    # (c) Minute-aggregated activity gaps > 1 hr
    gaps = rep["activity_gaps_gt_1h"]
    print(f"\n--- (c) Activity gaps > 1 hr: {len(gaps)} ---")
    for m, g in gaps:
        print(f"   {m}  +{g}")

    # Per-month agg_id monotonicity (across-month)
    last_max = -1
    breaks: list = []
    for m, mn, mx, _ in rep["agg_id_per_month"]:
        if mn <= last_max:
            breaks.append((m, last_max, mn))
        last_max = mx
    if breaks:
        print(f"\n--- agg_id ACROSS-MONTH BREAKS: {len(breaks)} ---")
        for m, prev_max, cur_min in breaks[:10]:
            print(f"   month={m} prev_max={prev_max:,} cur_min={cur_min:,}")
    else:
        print("\n--- agg_id monotonic across months: OK ---")

    # (d) Global agg_id span vs count
    g = rep["agg_id_global"]
    n = g["n"] or 0
    span = g["span"] or 0
    print("\n--- (d) agg_id global ---")
    print(f"  MIN(agg_id) = {g['mn']:,}")
    print(f"  MAX(agg_id) = {g['mx']:,}")
    print(f"  COUNT(*)    = {n:,}")
    print(f"  span        = {span:,}")
    if span:
        gap = span - n
        print(f"  span - count = {gap:,}  ({gap / span * 100:.4f}% gap fraction)")
        if n > span:
            print("  WARNING: count > span — duplicates present?")
        elif n == span:
            print("  count == span: zero gaps (every agg_id present)")
        else:
            print("  count < span: OK (small Binance-side gaps allowed)")

    # ingest_log audit
    a = rep["ingest_log_audit"]
    print("\n--- ingest_log audit ---")
    print(f"  rows:       {a['n_rows']}")
    print(f"  unique sha: {a['n_unique_sha']}")
    print(f"  null sha:   {a['n_null']}")

    # DR v3.0.4: source-dupe diagnostic
    dupes = rep["source_dupes"]
    print(f"\n--- source-dupe diagnostic: {len(dupes)} affected month(s) ---")
    if dupes:
        for d in dupes:
            print(f"   {d['month']:%Y-%m}  expected={d['expected_rows']:>11,}  "
                  f"actual={d['actual_rows']:>11,}  "
                  f"dupes={d['source_dupes']:,}")
        total_dupes = sum(d["source_dupes"] for d in dupes)
        print(f"   total source dupes across all months: {total_dupes:,}")


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────
def _print_checkpoint(symbol: str, after_month: str, cum_rows: int,
                       cum_t: float) -> None:
    from data.db import ticks_table, symbol_short
    t = ticks_table(symbol)
    sym = symbol_short(symbol)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n FROM timescaledb_information.chunks
                WHERE hypertable_schema='events' AND hypertable_name=%s
            """, (f"ticks_{sym}",))
            chunks = cur.fetchone()["n"]
            cur.execute(f"SELECT pg_size_pretty(hypertable_size('{t}')) AS sz")
            size = cur.fetchone()["sz"]
    print(
        f"=== [{symbol}] checkpoint after {after_month}: rows={cum_rows:,}  "
        f"time={cum_t:.0f}s  chunks={chunks}  size={size} ===",
        flush=True,
    )


def run_ingest(symbol_filter: Optional[str] = None) -> None:
    """Run ingest for all configured symbols, or just one if symbol_filter given.

    DR v3.0.14: when symbol_filter is set, init_schema only the target symbol
    and ingest only its months from the config's symbols list.
    """
    cfg = load_config()
    config_symbols = cfg["data"]["binance"]["symbols"]
    if symbol_filter is not None:
        if symbol_filter not in config_symbols:
            raise ValueError(
                f"symbol {symbol_filter!r} not in config symbols list "
                f"{config_symbols}; add it to config.yaml first"
            )
        symbols_to_run = [symbol_filter]
    else:
        symbols_to_run = config_symbols

    # Map BTCUSDT → BTC for schema bootstrap
    from data.db import symbol_short
    schema_symbols = list({symbol_short(s).upper() for s in symbols_to_run})
    init_schema(
        chunk_interval_ticks=cfg["database"]["chunk_interval_ticks"],
        compress_after_ticks=cfg["database"]["compress_after_ticks"],
        chunk_interval_bars=cfg["database"]["chunk_interval_bars"],
        symbols=schema_symbols,
    )

    binance_cfg = cfg["data"]["binance"]
    storage_dir = PROJECT_ROOT / binance_cfg["storage_dir"]
    storage_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(binance_cfg["start_date"])
    today = datetime.now(timezone.utc).date()
    all_months = months_to_ingest(start, today)

    # Pre-filter months already in ingest_log so [NN/total] reflects pending
    # work only. ingest_one_month still detects sha-republish on filtered-in
    # months; pre-filtered months are trusted by audit log presence.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol, month FROM events.ingest_log")
            done = {(r["symbol"], r["month"]) for r in cur.fetchall()}

    work = [
        (symbol, month)
        for symbol in symbols_to_run
        for month in all_months
        if (symbol, month) not in done
    ]
    N = len(work)
    total = len(all_months) * len(symbols_to_run)
    print(
        f"Phase 0.1 ingest: {N} pending of {total} total "
        f"({len(done)} already in ingest_log; symbols={symbols_to_run})",
        flush=True,
    )

    cum_rows = 0
    cum_t = 0.0
    for i, (symbol, month) in enumerate(work, start=1):
        t0 = time.perf_counter()
        res = ingest_one_month(symbol, month, storage_dir)
        total_s = time.perf_counter() - t0
        cum_rows += res["actual"]
        cum_t += total_s
        print(
            f"[{i:>2}/{N}] {symbol} {res['month']}  "
            f"download={res['download_s']:5.1f}s  sha=ok  "
            f"rows={res['actual']:>10,}  "
            f"copy={res['copy_s']:5.1f}s  "
            f"total={total_s:5.1f}s",
            flush=True,
        )
        if i % 12 == 0 or i == N:
            _print_checkpoint(symbol, res["month"], cum_rows, cum_t)
    # No try/except: STEP 3 contract — fail fast, do not skip and continue.

    # Per-symbol sanity report
    for sym_ticker in symbols_to_run:
        sym = symbol_short(sym_ticker).upper()
        print_sanity_report(sanity_checks(symbol=sym))


def _run_single_month(month_str: str) -> int:
    """Smoke-test entry: ingest exactly one month, hard-fail on any error."""
    cfg = load_config()
    init_schema(
        chunk_interval_ticks=cfg["database"]["chunk_interval_ticks"],
        compress_after_ticks=cfg["database"]["compress_after_ticks"],
        chunk_interval_bars=cfg["database"]["chunk_interval_bars"],
    )

    binance_cfg = cfg["data"]["binance"]
    storage_dir = PROJECT_ROOT / binance_cfg["storage_dir"]
    storage_dir.mkdir(parents=True, exist_ok=True)

    month = date.fromisoformat(month_str + "-01")
    today = datetime.now(timezone.utc).date()
    cutoff = date(today.year, today.month, 1)
    if month >= cutoff:
        LOG.error("Cannot ingest in-flight month %s (cutoff=%s)", month, cutoff)
        return 2

    for symbol in binance_cfg["symbols"]:
        t0 = time.perf_counter()
        res = ingest_one_month(symbol, month, storage_dir)
        elapsed = time.perf_counter() - t0
        LOG.info("M %s: %s, %d rows in %.1f s",
                 res["month"], res["status"], res["actual"], elapsed)
        print(json.dumps(res, default=str, indent=2))
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="ingest_ticks")
    p.add_argument("--month", metavar="YYYY-MM",
                   help="Smoke-test: ingest a single month and exit.")
    p.add_argument("--sanity", action="store_true",
                   help="Run sanity checks only (no ingest).")
    p.add_argument("--symbol", metavar="SYMBOL",
                   help="DR v3.0.14: restrict to one symbol (e.g. BTCUSDT, "
                        "ETHUSDT). Must be in config.yaml symbols list. "
                        "If omitted, processes all configured symbols.")
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.sanity:
            sym_arg = symbol_short(args.symbol).upper() if args.symbol else "BTC"
            print_sanity_report(sanity_checks(symbol=sym_arg))
            return 0
        if args.month:
            return _run_single_month(args.month)
        run_ingest(symbol_filter=args.symbol)
        return 0
    finally:
        close_pool()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
