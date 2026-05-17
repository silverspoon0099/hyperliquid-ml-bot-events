"""PostgreSQL + TimescaleDB I/O layer for v3.0 event-bar pipeline.

Shared DB instance with v1.0 (Decision v2.27); credentials symlinked from
v1.0's .env per DR v3.0.2 §5. Schema "events" per spec §6.1.

DR v3.0.14: schema bootstrap parameterized by symbol. Tables follow
`events.ticks_{sym}` / `events.bars_{sym}_cusum` convention. Default
init_schema() still creates BTC tables for backward compatibility.
Phase B (SOL/LINK) and Path 3a (ETH) use init_schema(symbols=["ETH"]) etc.

Phase 0.1 surface: init_schema(), get_connection(), ping(), close_pool().
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

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
# Symbol → table-name helpers (DR v3.0.14)
# ─────────────────────────────────────────────────────────────────────────
def symbol_short(symbol: str) -> str:
    """Map Binance ticker (BTCUSDT/ETHUSDT) or bare symbol (BTC/ETH) → 'btc'/'eth'."""
    s = symbol.upper()
    if s.endswith("USDT"):
        s = s[:-4]
    return s.lower()


def ticks_table(symbol: str) -> str:
    return f"events.ticks_{symbol_short(symbol)}"


def bars_table(symbol: str) -> str:
    return f"events.bars_{symbol_short(symbol)}_cusum"


# ─────────────────────────────────────────────────────────────────────────
# Schema bootstrap — DR v3.0.2 §1, §1b, §2; DR v3.0.5 §8; DR v3.0.14
# ─────────────────────────────────────────────────────────────────────────
_DDL_SCHEMA = "CREATE SCHEMA IF NOT EXISTS events;"

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


def _ddl_ticks(symbol: str) -> str:
    table = ticks_table(symbol)
    return f"""
        CREATE TABLE IF NOT EXISTS {table} (
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


def _ddl_bars(symbol: str) -> str:
    table = bars_table(symbol)
    return f"""
        CREATE TABLE IF NOT EXISTS {table} (
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


def _init_symbol_tables(
    cur, symbol: str,
    chunk_interval_ticks: str, compress_after_ticks: str,
    chunk_interval_bars: str,
) -> None:
    """Create ticks_{sym} + bars_{sym}_cusum tables, hypertables, compression policy."""
    sym = symbol_short(symbol)
    t_ticks = ticks_table(symbol)
    t_bars = bars_table(symbol)

    cur.execute(_ddl_ticks(symbol))
    cur.execute(
        f"SELECT create_hypertable(%s, 'ts', "
        f"chunk_time_interval => %s::interval, if_not_exists => TRUE);",
        (t_ticks, chunk_interval_ticks),
    )

    cur.execute(
        "SELECT 1 FROM timescaledb_information.compression_settings "
        "WHERE hypertable_schema='events' AND hypertable_name=%s LIMIT 1",
        (f"ticks_{sym}",),
    )
    if cur.fetchone() is None:
        cur.execute(
            f"ALTER TABLE {t_ticks} SET ("
            f"timescaledb.compress, "
            f"timescaledb.compress_segmentby = 'is_buyer_maker', "
            f"timescaledb.compress_orderby = 'ts, agg_id');"
        )
    cur.execute(
        f"SELECT add_compression_policy(%s, %s::interval, if_not_exists => TRUE);",
        (t_ticks, compress_after_ticks),
    )

    cur.execute(_ddl_bars(symbol))
    cur.execute(
        f"SELECT create_hypertable(%s, 'bar_close_ts', "
        f"chunk_time_interval => %s::interval, if_not_exists => TRUE);",
        (t_bars, chunk_interval_bars),
    )


def init_schema(
    chunk_interval_ticks: str = "7 days",
    compress_after_ticks: str = "30 days",
    chunk_interval_bars: str = "180 days",
    symbols: Optional[list[str]] = None,
) -> None:
    """Create schema + per-symbol tables + hypertables + compression. Idempotent.

    Defaults match config.yaml. `symbols` defaults to ['BTC'] for backward
    compatibility; pass ['ETH'] or ['BTC','ETH'] for multi-asset (DR v3.0.14).
    """
    if symbols is None:
        symbols = ["BTC"]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL_SCHEMA)
            cur.execute(_DDL_INGEST_LOG)
            for sym in symbols:
                _init_symbol_tables(
                    cur, sym, chunk_interval_ticks,
                    compress_after_ticks, chunk_interval_bars,
                )
        conn.commit()


# ─────────────────────────────────────────────────────────────────────────
# Paper-trading schema (DR v3.0.23) — session metadata, trades, decisions
# ─────────────────────────────────────────────────────────────────────────
_DDL_PAPER_SESSIONS = """
CREATE TABLE IF NOT EXISTS events.paper_sessions (
    session_id              TEXT             PRIMARY KEY,
    asset                   TEXT             NOT NULL,
    model_artifact_path     TEXT             NOT NULL,
    bar_threshold           DOUBLE PRECISION NOT NULL,
    tp_pct                  DOUBLE PRECISION NOT NULL,
    sl_pct                  DOUBLE PRECISION NOT NULL,
    vertical_bars           INTEGER          NOT NULL,
    confidence_threshold    DOUBLE PRECISION NOT NULL,
    position_size_usd       DOUBLE PRECISION NOT NULL,
    cost_bps_round_trip     DOUBLE PRECISION NOT NULL,
    max_daily_loss_pct      DOUBLE PRECISION NOT NULL,
    started_at              TIMESTAMPTZ      NOT NULL DEFAULT now(),
    ended_at                TIMESTAMPTZ,
    end_reason              TEXT,
    notes                   TEXT
);
"""

_DDL_PAPER_TRADES = """
CREATE TABLE IF NOT EXISTS events.paper_trades (
    trade_id        BIGSERIAL        PRIMARY KEY,
    session_id      TEXT             NOT NULL REFERENCES events.paper_sessions(session_id),
    bar_id_entry    BIGINT           NOT NULL,
    entry_ts        TIMESTAMPTZ      NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    direction       SMALLINT         NOT NULL,
    p_long          DOUBLE PRECISION NOT NULL,
    p_short         DOUBLE PRECISION NOT NULL,
    p_neutral       DOUBLE PRECISION NOT NULL,
    entry_reason    TEXT             NOT NULL,
    status          TEXT             NOT NULL DEFAULT 'open',
    exit_bar_id     BIGINT,
    exit_ts         TIMESTAMPTZ,
    exit_price      DOUBLE PRECISION,
    exit_reason     TEXT,
    holding_bars    INTEGER,
    pnl_bps_gross   DOUBLE PRECISION,
    pnl_bps_net     DOUBLE PRECISION,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS paper_trades_session_status_idx
    ON events.paper_trades(session_id, status);
CREATE INDEX IF NOT EXISTS paper_trades_session_entry_ts_idx
    ON events.paper_trades(session_id, entry_ts);
"""

_DDL_PAPER_DECISIONS = """
CREATE TABLE IF NOT EXISTS events.paper_decisions (
    session_id      TEXT             NOT NULL REFERENCES events.paper_sessions(session_id),
    bar_id          BIGINT           NOT NULL,
    decided_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    p_long          DOUBLE PRECISION NOT NULL,
    p_short         DOUBLE PRECISION NOT NULL,
    p_neutral       DOUBLE PRECISION NOT NULL,
    max_prob        DOUBLE PRECISION NOT NULL,
    argmax_class    TEXT             NOT NULL,
    traded          BOOLEAN          NOT NULL,
    skip_reason     TEXT,
    trade_id        BIGINT,
    PRIMARY KEY (session_id, bar_id)
);
"""


def init_paper_schema() -> None:
    """Create paper-trading tables (DR v3.0.23). Idempotent."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL_SCHEMA)
            cur.execute(_DDL_PAPER_SESSIONS)
            cur.execute(_DDL_PAPER_TRADES)
            cur.execute(_DDL_PAPER_DECISIONS)
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
        symbols = sys.argv[2:] if len(sys.argv) > 2 else ["BTC"]
        init_schema(symbols=symbols)
        print(f"Schema initialized for: {symbols}")
    print(ping())
