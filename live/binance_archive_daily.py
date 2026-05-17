"""Binance Vision daily aggTrades fetcher for paper trading (DR v3.0.23).

Solves the api.binance.com geo-block problem for paper trading: instead of
the real-time REST API (which requires a non-US proxy), this module fetches
the publicly-available DAILY aggTrade archives from data.binance.vision.

URL format:
    https://data.binance.vision/data/spot/daily/aggTrades/{SYMBOL}/
        {SYMBOL}-aggTrades-{YYYY-MM-DD}.zip

Availability: each daily archive is published ~24h after the day's UTC close
(typically 01:00-03:00 UTC the next day).

Trade-off vs REST:
  - Latency: ~24h vs seconds
  - But our event bars form every ~2h on average, so the latency is small
    relative to trade horizon
  - SAME data source as our 7-year training history (no train/serve skew)
  - No auth, no rate limits, no geo-block

Strategy:
  - Determine last day fully present in DB
  - For each day from (last+1) up to (yesterday UTC):
      - Try to download the daily zip
      - If 404 (not yet released), stop
      - If 200, parse + insert into events.ticks_{sym}
  - Return counts dict

Compatible with the orchestrator's polling loop: call this every N minutes.
Most calls will be no-ops (data not yet released or already ingested).
Once per day around 02:00 UTC, the previous day's data appears and gets
ingested in one batch.
"""
from __future__ import annotations

import csv
import io
import logging
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import requests

from data.db import get_connection, ticks_table

LOG = logging.getLogger("live.binance_archive_daily")

ARCHIVE_HOST = "https://data.binance.vision"
DAILY_PATH = "/data/spot/daily/aggTrades"


def _daily_zip_url(symbol_binance: str, day: date) -> str:
    return (f"{ARCHIVE_HOST}{DAILY_PATH}/{symbol_binance}/"
            f"{symbol_binance}-aggTrades-{day:%Y-%m-%d}.zip")


def _download_zip(url: str, timeout: float = 30.0) -> Optional[bytes]:
    """Returns zip bytes on success, None on 404 (not yet published)."""
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def _parse_ts(raw: str) -> datetime:
    """Auto-detect ms vs µs (Binance Vision switched to µs in 2025).
    Mirrors data.ingest_ticks._parse_ts."""
    n = int(raw)
    secs = n / 1_000_000.0 if n > 10**14 else n / 1_000.0
    return datetime.fromtimestamp(secs, tz=timezone.utc)


def _iter_ticks_from_zip(zip_bytes: bytes, symbol_binance: str, day: date
                          ) -> Iterator[tuple]:
    """Yield tick tuples ready for COPY: (agg_id, ts, price, qty, quote_qty,
    is_buyer_maker, first_trade_id, last_trade_id).

    Binance daily aggTrade CSV columns:
      0: agg_id, 1: price, 2: qty, 3: first_trade_id, 4: last_trade_id,
      5: timestamp_ms, 6: is_buyer_maker, 7: best_match (ignored)
    """
    expected_name = f"{symbol_binance}-aggTrades-{day:%Y-%m-%d}.csv"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if expected_name in names:
            chosen = expected_name
        else:
            csvs = [n for n in names if n.endswith(".csv")]
            if not csvs:
                raise RuntimeError(f"No CSV in zip for {day}; contains {names}")
            chosen = csvs[0]
        with zf.open(chosen) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.reader(text)
            for row in reader:
                try:
                    agg_id = int(row[0])
                except ValueError:
                    continue  # header row
                price = float(row[1])
                qty = float(row[2])
                first_id = int(row[3])
                last_id = int(row[4])
                ts = _parse_ts(row[5])
                # is_buyer_maker — parse as bool ("True"/"False" or 0/1)
                ibm_raw = row[6].strip().lower()
                if ibm_raw in ("true", "1", "t"):
                    is_buyer_maker = True
                elif ibm_raw in ("false", "0", "f"):
                    is_buyer_maker = False
                else:
                    raise ValueError(f"Unexpected is_buyer_maker value: {row[6]!r}")
                yield (
                    agg_id, ts, price, qty, price * qty,
                    is_buyer_maker, first_id, last_id,
                )


def _last_tick_day(symbol: str) -> Optional[date]:
    """Return the UTC date of the most recent tick in events.ticks_{sym},
    or None if table empty."""
    tbl = ticks_table(symbol)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(ts) AS max_ts FROM {tbl}")
            row = cur.fetchone()
            if row is None or row["max_ts"] is None:
                return None
            return row["max_ts"].astimezone(timezone.utc).date()


def _ingest_day(symbol: str, symbol_binance: str, day: date, zip_bytes: bytes
                 ) -> dict:
    """Insert one day's worth of ticks via staging table + ON CONFLICT.

    Pattern mirrors data/ingest_ticks.py: COPY into TEMP staging,
    INSERT INTO target with ON CONFLICT DO NOTHING.
    """
    tbl = ticks_table(symbol)
    n_rows = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Staging table
            cur.execute(
                f"CREATE TEMP TABLE _staging_ticks (LIKE {tbl}) ON COMMIT DROP"
            )
            # COPY rows in
            with cur.copy(
                "COPY _staging_ticks "
                "(agg_id, ts, price, qty, quote_qty, is_buyer_maker, "
                "first_trade_id, last_trade_id) FROM STDIN"
            ) as copy:
                for row in _iter_ticks_from_zip(zip_bytes, symbol_binance, day):
                    copy.write_row(row)
                    n_rows += 1
            # INSERT into target with conflict skip
            cur.execute(
                f"INSERT INTO {tbl} "
                f"(agg_id, ts, price, qty, quote_qty, is_buyer_maker, "
                f"first_trade_id, last_trade_id) "
                f"SELECT agg_id, ts, price, qty, quote_qty, is_buyer_maker, "
                f"       first_trade_id, last_trade_id "
                f"FROM _staging_ticks "
                f"ON CONFLICT (agg_id, ts) DO NOTHING"
            )
            inserted = cur.rowcount
        conn.commit()
    return {"day": str(day), "rows_in_csv": n_rows, "rows_inserted": int(inserted)}


def poll_and_ingest_daily(
    symbol: str = "BTC",
    symbol_binance: str = "BTCUSDT",
    max_days_to_catch_up: int = 14,
    max_seconds: float = 300.0,
) -> dict:
    """One polling cycle: catch up DB to yesterday-UTC via daily archives.

    Returns aggregate counts. If no new archive is available, returns
    reason='no_new_archive' with 0 counts.
    """
    last_day = _last_tick_day(symbol)
    today_utc = datetime.now(timezone.utc).date()
    yesterday_utc = today_utc - timedelta(days=1)

    if last_day is None:
        LOG.warning("No prior ticks in %s — caller should backfill from monthly archives first",
                    ticks_table(symbol))
        return {"reason": "empty_table", "rows_inserted": 0,
                "days_ingested": 0, "last_day_before": None, "last_day_after": None}

    # Try days from last_day+1 up to yesterday_utc
    cursor_day = last_day + timedelta(days=1)
    days_done = 0
    total_inserted = 0
    t0 = time.perf_counter()
    skipped_404 = []

    while (cursor_day <= yesterday_utc
           and days_done < max_days_to_catch_up
           and (time.perf_counter() - t0) < max_seconds):
        url = _daily_zip_url(symbol_binance, cursor_day)
        try:
            zip_bytes = _download_zip(url)
        except requests.HTTPError as e:
            LOG.error("HTTP error downloading %s: %s", url, e)
            break
        except Exception as e:
            LOG.exception("Unexpected error downloading %s", url)
            break

        if zip_bytes is None:
            # 404 — archive not yet released for this day
            skipped_404.append(str(cursor_day))
            LOG.info("Archive not yet available for %s — stopping catch-up", cursor_day)
            break

        try:
            res = _ingest_day(symbol, symbol_binance, cursor_day, zip_bytes)
        except Exception as e:
            LOG.exception("Failed to ingest day %s", cursor_day)
            break

        total_inserted += res["rows_inserted"]
        days_done += 1
        LOG.info("Ingested %s: %d rows in CSV, %d inserted (skipped %d conflicts)",
                 cursor_day, res["rows_in_csv"], res["rows_inserted"],
                 res["rows_in_csv"] - res["rows_inserted"])
        cursor_day += timedelta(days=1)

    wall = time.perf_counter() - t0
    new_last = _last_tick_day(symbol)
    reason = "complete" if days_done > 0 else "no_new_archive"
    return {
        "reason": reason,
        "rows_inserted": total_inserted,
        "days_ingested": days_done,
        "skipped_404": skipped_404,
        "last_day_before": str(last_day),
        "last_day_after": str(new_last) if new_last else None,
        "wall_seconds": wall,
    }
