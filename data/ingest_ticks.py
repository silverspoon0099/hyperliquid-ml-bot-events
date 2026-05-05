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

from .db import close_pool, get_connection, init_schema

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
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    if len(names) != 1:
        zf.close()
        raise ValueError(f"Expected 1 file in {zip_path}, found {names}")
    return zf, zf.open(names[0])


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
_COPY_SQL = (
    "COPY events.ticks_btc "
    "(agg_id, ts, price, qty, quote_qty, is_buyer_maker, first_trade_id, last_trade_id) "
    "FROM STDIN WITH (FORMAT BINARY)"
)
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


def _existing_count(cur, month: date) -> int:
    cur.execute(
        "SELECT COUNT(*) AS n FROM events.ticks_btc WHERE ts >= %s AND ts < %s",
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
    """DR §3 step 8: DELETE range → COPY → audit log, all in one tx."""
    actual = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM events.ticks_btc WHERE ts >= %s AND ts < %s",
                (month, _next_month(month)),
            )
            with cur.copy(_COPY_SQL) as cp:
                cp.set_types(_COPY_TYPES)
                for t in ticks:
                    cp.write_row((
                        t.agg_id, t.ts, t.price, t.qty, t.quote_qty,
                        t.is_buyer_maker, t.first_trade_id, t.last_trade_id,
                    ))
                    actual += 1
            cur.execute(_LOG_UPSERT, (symbol, month, sha256, expected_rows, actual))
        conn.commit()
    return actual


# ─────────────────────────────────────────────────────────────────────────
# Per-month orchestration
# ─────────────────────────────────────────────────────────────────────────
def ingest_one_month(symbol: str, month: date, storage_dir: Path) -> dict:
    """Idempotent ingest of one (symbol, month). Returns a status dict."""
    yyyymm = f"{month:%Y-%m}"
    zip_name = f"{symbol}-aggTrades-{yyyymm}.zip"
    zip_path = storage_dir / symbol / zip_name
    sha_path = storage_dir / symbol / (zip_name + ".CHECKSUM")

    # 1. download (skip if cached)
    if not zip_path.exists():
        LOG.info("[%s %s] downloading zip", symbol, yyyymm)
        _download(_zip_url(symbol, month), zip_path)
    if not sha_path.exists():
        _download(_checksum_url(symbol, month), sha_path)

    # 2. verify SHA256
    published = _published_sha256(sha_path)
    actual_sha = _file_sha256(zip_path)
    if published != actual_sha:
        raise RuntimeError(
            f"[{symbol} {yyyymm}] SHA mismatch: published={published} "
            f"actual={actual_sha} — refusing to ingest"
        )

    # 3. expected row count
    expected = _count_data_rows(zip_path)

    # 4 + 5 + 6 + 7. skip / re-ingest decision
    with get_connection() as conn:
        with conn.cursor() as cur:
            existing = _existing_count(cur, month)
            logged = _logged_sha(cur, symbol, month)

    if existing >= expected and logged == published:
        LOG.info("[%s %s] already complete (%d rows, sha match) — skip",
                 symbol, yyyymm, existing)
        return {"month": yyyymm, "status": "skipped",
                "expected": expected, "actual": existing, "sha256": published}

    if existing >= expected and logged != published:
        LOG.warning("[%s %s] sha changed (logged=%s published=%s) — re-ingest",
                    symbol, yyyymm, logged, published)

    # 8. atomic: DELETE + COPY + log
    LOG.info("[%s %s] ingesting %d rows", symbol, yyyymm, expected)
    actual = _ingest_month_atomic(
        symbol, month, zip_path, published, expected, _iter_ticks(zip_path)
    )
    if actual != expected:
        raise RuntimeError(
            f"[{symbol} {yyyymm}] row count mismatch: expected={expected} "
            f"loaded={actual} — transaction rolled back? investigate"
        )
    LOG.info("[%s %s] ingested %d rows", symbol, yyyymm, actual)
    return {"month": yyyymm, "status": "ingested",
            "expected": expected, "actual": actual, "sha256": published}


# ─────────────────────────────────────────────────────────────────────────
# Sanity checks (DR §4 + Appendix A)
# ─────────────────────────────────────────────────────────────────────────
def sanity_checks() -> dict:
    """Returns a report dict; raises on hard failures."""
    out: dict = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts
                FROM events.ticks_btc
            """)
            r = cur.fetchone()
            out["total_ticks"] = r["n"]
            out["first_ts"] = r["first_ts"]
            out["last_ts"] = r["last_ts"]

            cur.execute("""
                SELECT date_trunc('month', ts) AS m, COUNT(*) AS n
                FROM events.ticks_btc
                GROUP BY 1 ORDER BY 1
            """)
            out["ticks_per_month"] = [(r["m"], r["n"]) for r in cur.fetchall()]

            cur.execute("""
                SELECT date_trunc('day', ts) AS d, SUM(quote_qty) AS quote_vol
                FROM events.ticks_btc
                GROUP BY 1 ORDER BY 1
            """)
            out["daily_quote_volume"] = [(r["d"], r["quote_vol"]) for r in cur.fetchall()]

            # DR §4: minute-aggregated zero-activity gaps > 1 hr
            cur.execute("""
                WITH minute_bins AS (
                    SELECT date_trunc('minute', ts) AS m
                    FROM events.ticks_btc
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
            cur.execute("""
                SELECT date_trunc('month', ts) AS m,
                       MIN(agg_id) AS min_id, MAX(agg_id) AS max_id, COUNT(*) AS n
                FROM events.ticks_btc
                GROUP BY 1 ORDER BY 1
            """)
            out["agg_id_per_month"] = [
                (r["m"], r["min_id"], r["max_id"], r["n"]) for r in cur.fetchall()
            ]
    return out


def print_sanity_report(rep: dict) -> None:
    print(f"Total ticks:    {rep['total_ticks']:,}")
    print(f"First ts:       {rep['first_ts']}")
    print(f"Last ts:        {rep['last_ts']}")
    print(f"Months covered: {len(rep['ticks_per_month'])}")
    gaps = rep["activity_gaps_gt_1h"]
    print(f"Activity gaps > 1h: {len(gaps)}")
    for m, g in gaps[:20]:
        print(f"   {m}  +{g}")
    if len(gaps) > 20:
        print(f"   ... +{len(gaps) - 20} more")
    # monotonicity check
    last_max = -1
    breaks = []
    for m, mn, mx, _ in rep["agg_id_per_month"]:
        if mn <= last_max:
            breaks.append((m, last_max, mn))
        last_max = mx
    if breaks:
        print(f"agg_id MONOTONICITY BREAKS: {len(breaks)}")
        for m, prev_max, cur_min in breaks[:5]:
            print(f"   month={m} prev_max={prev_max} cur_min={cur_min}")
    else:
        print("agg_id monotonic across months: OK")


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────
def run_ingest() -> None:
    cfg = load_config()
    init_schema(
        chunk_interval=cfg["database"]["chunk_interval_ticks"],
        compress_after=cfg["database"]["compress_after_ticks"],
    )

    binance_cfg = cfg["data"]["binance"]
    storage_dir = PROJECT_ROOT / binance_cfg["storage_dir"]
    storage_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(binance_cfg["start_date"])
    today = datetime.now(timezone.utc).date()
    months = months_to_ingest(start, today)

    LOG.info("Phase 0.1 ingest: %d months from %s to %s (excl. current)",
             len(months), months[0] if months else "—",
             months[-1] if months else "—")

    for symbol in binance_cfg["symbols"]:
        for month in months:
            t0 = time.perf_counter()
            res = ingest_one_month(symbol, month, storage_dir)
            elapsed = time.perf_counter() - t0
            LOG.info("M %s: %s, %d rows in %.1f s",
                     res["month"], res["status"], res["actual"], elapsed)
    # No try/except per DR: fail fast on first error rather than skip months.

    print_sanity_report(sanity_checks())


def _run_single_month(month_str: str) -> int:
    """Smoke-test entry: ingest exactly one month, hard-fail on any error."""
    cfg = load_config()
    init_schema(
        chunk_interval=cfg["database"]["chunk_interval_ticks"],
        compress_after=cfg["database"]["compress_after_ticks"],
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
    args = p.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.sanity:
            print_sanity_report(sanity_checks())
            return 0
        if args.month:
            return _run_single_month(args.month)
        run_ingest()
        return 0
    finally:
        close_pool()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
