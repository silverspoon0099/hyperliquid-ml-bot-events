"""PostgreSQL + TimescaleDB I/O layer for v3.0 event-bar pipeline.

Shared DB instance with v1.0 (Decision v2.27); credentials symlinked from
v1.0's .env per DR v3.0.2 §5. Schema "events" per spec §6.1.

Conventions mirror /nvme1/projects/trading/ml-bot/data/db.py:
  * Connection pool with row_factory=dict_row
  * SQLAlchemy engine alongside for pandas.read_sql_query
  * load_dotenv at module import (absolute path to repo root)

Phase 0.1 surface: init_schema(), get_connection(), ping(), close_pool().
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sqlalchemy import create_engine
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

_POOL: ConnectionPool | None = None
_ENGINE = None


def _build_url() -> str:
    return (
        f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@"
        f"{os.getenv('PG_HOST', '127.0.0.1')}:{os.getenv('PG_PORT', '5432')}/"
        f"{os.getenv('PG_DB')}"
    )


def get_pool() -> ConnectionPool:
    global _POOL
    if _POOL is None:
        _POOL = ConnectionPool(
            conninfo=_build_url(),
            min_size=int(os.getenv("PG_POOL_MIN", "2")),
            max_size=max(int(os.getenv("PG_POOL_MAX", "10")), 2),
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _POOL


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(
            f"postgresql+psycopg://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@"
            f"{os.getenv('PG_HOST', '127.0.0.1')}:{os.getenv('PG_PORT', '5432')}/"
            f"{os.getenv('PG_DB')}"
        )
    return _ENGINE


def get_connection() -> Connection:
    """Pooled connection — `with get_connection() as conn: ...`."""
    return get_pool().connection()


def close_pool() -> None:
    global _POOL, _ENGINE
    if _POOL is not None:
        _POOL.close()
        _POOL = None
    if _ENGINE is not None:
        _ENGINE.dispose()
        _ENGINE = None


# ─────────────────────────────────────────────────────────────────────────
# Schema bootstrap — DR v3.0.2 §1, §1b, §2
# ─────────────────────────────────────────────────────────────────────────
_DDL_SCHEMA = "CREATE SCHEMA IF NOT EXISTS events;"

_DDL_TICKS_BTC = """
CREATE TABLE IF NOT EXISTS events.ticks_btc (
    agg_id          BIGINT           NOT NULL,
    ts              TIMESTAMPTZ      NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    qty             DOUBLE PRECISION NOT NULL,
    quote_qty       DOUBLE PRECISION NOT NULL,
    is_buyer_maker  BOOLEAN          NOT NULL,
    first_trade_id  BIGINT           NOT NULL,
    last_trade_id   BIGINT           NOT NULL,
    PRIMARY KEY (agg_id, ts)
);
"""

_DDL_HYPERTABLE = """
SELECT create_hypertable(
    'events.ticks_btc', 'ts',
    chunk_time_interval => %s::interval,
    if_not_exists => TRUE
);
"""

_DDL_INGEST_LOG = """
CREATE TABLE IF NOT EXISTS events.ingest_log (
    symbol         TEXT        NOT NULL,
    month          DATE        NOT NULL,
    sha256         TEXT        NOT NULL,
    expected_rows  BIGINT      NOT NULL,
    actual_rows    BIGINT      NOT NULL,
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, month)
);
"""

_CHECK_COMPRESSION_ENABLED = """
SELECT 1
FROM timescaledb_information.compression_settings
WHERE hypertable_schema = 'events' AND hypertable_name = 'ticks_btc'
LIMIT 1;
"""

_DDL_ENABLE_COMPRESSION = """
ALTER TABLE events.ticks_btc SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'is_buyer_maker',
    timescaledb.compress_orderby   = 'ts, agg_id'
);
"""

_DDL_COMPRESSION_POLICY = """
SELECT add_compression_policy(
    'events.ticks_btc', %s::interval,
    if_not_exists => TRUE
);
"""

# DR v3.0.5 §8: bars table + hypertable. PK is (bar_id, bar_close_ts) — Timescale
# requires the partitioning column in every uniqueness constraint, same pattern
# as events.ticks_btc PK (agg_id, ts). UNIQUE(bar_close_ts, threshold_pct) per §6.3.
_DDL_BARS_BTC_CUSUM = """
CREATE TABLE IF NOT EXISTS events.bars_btc_cusum (
    bar_id        BIGSERIAL,
    bar_open_ts   TIMESTAMPTZ      NOT NULL,
    bar_close_ts  TIMESTAMPTZ      NOT NULL,
    open          DOUBLE PRECISION NOT NULL,
    high          DOUBLE PRECISION NOT NULL,
    low           DOUBLE PRECISION NOT NULL,
    close         DOUBLE PRECISION NOT NULL,
    volume        DOUBLE PRECISION NOT NULL,
    n_trades      INTEGER          NOT NULL,
    cusum_pos     DOUBLE PRECISION,
    cusum_neg     DOUBLE PRECISION,
    threshold_pct DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (bar_id, bar_close_ts)
);
"""

_DDL_BARS_HYPERTABLE = """
SELECT create_hypertable(
    'events.bars_btc_cusum', 'bar_close_ts',
    chunk_time_interval => %s::interval,
    if_not_exists => TRUE
);
"""


def init_schema(
    chunk_interval_ticks: str = "7 days",
    compress_after_ticks: str = "30 days",
    chunk_interval_bars: str = "180 days",
) -> None:
    """Create schema + tables + hypertables + compression policy. Idempotent.

    Defaults match config.yaml (DR v3.0.2 §2 for ticks; DR v3.0.5 §8 for bars).
    Pass overrides only for tests.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL_SCHEMA)
            cur.execute(_DDL_TICKS_BTC)
            cur.execute(_DDL_HYPERTABLE, (chunk_interval_ticks,))
            cur.execute(_DDL_INGEST_LOG)

            cur.execute(_CHECK_COMPRESSION_ENABLED)
            if cur.fetchone() is None:
                cur.execute(_DDL_ENABLE_COMPRESSION)
            cur.execute(_DDL_COMPRESSION_POLICY, (compress_after_ticks,))

            cur.execute(_DDL_BARS_BTC_CUSUM)
            cur.execute(_DDL_BARS_HYPERTABLE, (chunk_interval_bars,))
        conn.commit()


def ping() -> dict:
    """Verify DB + Timescale extension reachable."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT current_database() AS db, current_user AS usr, "
                "version() AS ver;"
            )
            info = cur.fetchone()
            cur.execute(
                "SELECT extversion FROM pg_extension WHERE extname='timescaledb';"
            )
            ts_row = cur.fetchone()
    return {
        "database": info["db"],
        "user": info["usr"],
        "postgres": info["ver"].split(",")[0],
        "timescaledb": ts_row["extversion"] if ts_row else None,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init_schema()
        print("Schema initialized.")
    print(ping())
