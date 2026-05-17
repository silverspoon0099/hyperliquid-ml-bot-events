"""Binance REST aggTrades fetcher for live/paper trading (DR v3.0.23).

Fetches recent BTCUSDT aggTrades incrementally and upserts into
events.ticks_btc. Designed for periodic polling (e.g., every 10 minutes)
to keep the tick stream fresh.

PROXY REQUIREMENT (DR v3.0.23): api.binance.com returns HTTP 451
(geo-blocked) from many regions including US. To preserve train/serve
parity with our training data (Binance.com archive), this client routes
through a user-provided proxy. Set EITHER:

  HTTPS_PROXY=http://your-proxy:port    (standard env, requests honors it)
or
  BINANCE_PROXY=http://your-proxy:port  (explicit, overrides HTTPS_PROXY for
                                          Binance calls only — useful if you
                                          want all traffic to bypass proxy
                                          except Binance)

If neither is set and your IP is geo-blocked, requests will fail with 451.

Strategy:
  - Query DB for latest agg_id in events.ticks_btc
  - Call Binance /api/v3/aggTrades with fromId = latest + 1, limit = 1000
  - Loop until either: empty response, response < limit, or caught-up flag
  - Bulk INSERT into events.ticks_btc with ON CONFLICT DO NOTHING

Endpoint reference:
  GET /api/v3/aggTrades?symbol=BTCUSDT&fromId={N}&limit=1000
  Returns list of:
    {a: aggId, p: price, q: qty, f: firstId, l: lastId, T: ts_ms, m: isBuyerMaker, M: bestMatch}

Rate limits (weight 4 per call, 6000/min budget → ~1500 calls/min for aggTrades).
For a 10-min poll covering 10×60×100 = 60k trades, we need 60 calls. Trivial.

Bootstrap: on first run, last_agg_id may not exist for the relevant recent
window. Caller should backfill via data/ingest_ticks.py (monthly archive)
before starting live polling.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from data.db import get_connection, ticks_table

LOG = logging.getLogger("live.binance_rest")

BINANCE_API = "https://api.binance.com/api/v3/aggTrades"
DEFAULT_LIMIT = 1000


def _proxy_dict() -> Optional[dict]:
    """Return proxies dict for requests if BINANCE_PROXY is set, else None
    (in which case `requests` falls back to HTTPS_PROXY env automatically)."""
    p = os.environ.get("BINANCE_PROXY")
    if p:
        return {"http": p, "https": p}
    return None


def fetch_aggtrades(
    symbol: str = "BTCUSDT",
    from_id: Optional[int] = None,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
    limit: int = DEFAULT_LIMIT,
    timeout: float = 15.0,
) -> list[dict]:
    """One REST call. Returns raw list of trades (may be empty)."""
    params: dict = {"symbol": symbol, "limit": limit}
    if from_id is not None:
        params["fromId"] = from_id
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    if end_time_ms is not None:
        params["endTime"] = end_time_ms
    r = requests.get(BINANCE_API, params=params, timeout=timeout, proxies=_proxy_dict())
    r.raise_for_status()
    return r.json()


def get_last_agg_id(symbol: str = "BTC") -> Optional[int]:
    """Return max(agg_id) in events.ticks_{sym} or None if table empty."""
    tbl = ticks_table(symbol)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(agg_id) AS max_id FROM {tbl}")
            row = cur.fetchone()
            return int(row["max_id"]) if row and row["max_id"] is not None else None


def insert_ticks(
    symbol: str, trades: list[dict],
) -> dict:
    """Bulk insert trades into events.ticks_{sym}. Returns counts dict.

    Each trade dict has Binance aggTrade keys (a, p, q, f, l, T, m).
    Insert with ON CONFLICT DO NOTHING for idempotency.
    """
    if not trades:
        return {"received": 0, "inserted": 0}

    tbl = ticks_table(symbol)
    rows = []
    for t in trades:
        rows.append((
            int(t["a"]),                                  # agg_id
            datetime.fromtimestamp(int(t["T"]) / 1000.0,  # ts (timestamptz)
                                    tz=timezone.utc),
            float(t["p"]),                                # price
            float(t["q"]),                                # qty
            float(t["p"]) * float(t["q"]),                # quote_qty
            bool(t["m"]),                                 # is_buyer_maker
            int(t["f"]),                                  # first_trade_id
            int(t["l"]),                                  # last_trade_id
        ))

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {tbl} "
                f"(agg_id, ts, price, qty, quote_qty, is_buyer_maker, "
                f"first_trade_id, last_trade_id) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                f"ON CONFLICT (agg_id, ts) DO NOTHING",
                rows,
            )
            conn.commit()
            inserted = cur.rowcount if cur.rowcount is not None else len(rows)

    return {"received": len(trades), "inserted": int(inserted)}


def poll_and_insert(
    symbol_for_db: str = "BTC",
    symbol_for_binance: str = "BTCUSDT",
    max_loops: int = 1000,
    max_seconds: float = 120.0,
) -> dict:
    """One polling cycle: catch up DB to latest Binance tick.

    Loops fetching 1000-trade batches starting from last+1 in DB. Stops on:
      - empty response, OR
      - response.length < limit (fully caught up), OR
      - max_loops reached, OR
      - max_seconds elapsed.

    Returns aggregate counts.
    """
    last_id = get_last_agg_id(symbol_for_db)
    if last_id is None:
        LOG.warning("No prior ticks in %s — caller should backfill from archive first",
                    ticks_table(symbol_for_db))
        return {"received": 0, "inserted": 0, "loops": 0, "wall_seconds": 0.0,
                "last_agg_id_before": None, "last_agg_id_after": None,
                "reason": "empty_table"}

    total_received = 0
    total_inserted = 0
    loop = 0
    t0 = time.perf_counter()
    cursor = last_id + 1

    while loop < max_loops and (time.perf_counter() - t0) < max_seconds:
        try:
            batch = fetch_aggtrades(symbol_for_binance, from_id=cursor, limit=DEFAULT_LIMIT)
        except requests.HTTPError as e:
            LOG.error("Binance HTTP error at fromId=%d: %s", cursor, e)
            break
        if not batch:
            break
        res = insert_ticks(symbol_for_db, batch)
        total_received += res["received"]
        total_inserted += res["inserted"]
        cursor = int(batch[-1]["a"]) + 1
        loop += 1
        if len(batch) < DEFAULT_LIMIT:
            # Caught up
            break

    wall = time.perf_counter() - t0
    new_last_id = get_last_agg_id(symbol_for_db)
    LOG.info("poll done: received=%d inserted=%d loops=%d wall=%.1fs  last_id %s → %s",
             total_received, total_inserted, loop, wall, last_id, new_last_id)
    return {
        "received": total_received,
        "inserted": total_inserted,
        "loops": loop,
        "wall_seconds": wall,
        "last_agg_id_before": last_id,
        "last_agg_id_after": new_last_id,
        "reason": "complete" if loop < max_loops else "max_loops",
    }
