# PROJECT_LOG — ml-bot-events (v3.0)

> Append-only decision log. Newest entry at top.
> Every code change must reference a Decision or DR here.

---

## 2026-05-06 — Decision v3.0.3 — Loader fix: multi-file Binance archive (DR)

**Context**: DR v3.0.2 STEP 3 full sweep failed at month 35 (2021-12) when
`_open_csv` asserted `len(zf.namelist()) == 1`. The 2021-12 BTCUSDT
aggTrades zip from Binance Vision contains TWO files:

    2,694,397,270 bytes  CRC=0x359574b8  BTCUSDT-aggTrades-2021-12.csv
    2,694,397,270 bytes  CRC=0x359574b8  fsx-data/collector_data/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2021-12.csv

Bytes-identical per CRC32. Second path looks like an AWS FSx collector
mount that leaked into Binance's archive structure. State at failure:
35 months done cleanly; 0 rows in DB for 2021-12 (atomic rollback per
DR v3.0.2 §3 step 8); resumable.

**Decisions**:

### 1. `_open_csv` selection logic

Selects the canonical root-level CSV matching `{zip_stem}.csv`. Falls
back to nested matches only if no root match. Hard-fails if no match
at all.

```python
def _open_csv(zip_path):
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    expected = zip_path.stem + ".csv"
    candidates = [n for n in names
                  if n == expected or n.endswith("/" + expected)]
    root = [n for n in candidates if "/" not in n]
    chosen = root[0] if root else (candidates[0] if candidates else None)
    if chosen is None:
        zf.close()
        raise ValueError(...)
    if len(candidates) > 1:
        ...  # log line — see §2
    return zf, zf.open(chosen)
```

### 2. Inline audit log (replaces rejected separate-sanity-pass option)

When `len(candidates) > 1`, emit one INFO log line per `_open_csv`
call. Format:

    [BTCUSDT 2021-12] zip contains 2 CSVs; using
    BTCUSDT-aggTrades-2021-12.csv, others: ['fsx-data/...']

Captures the multi-CSV case in `logs/ingest_full.log` at zero extra
cost. No second pass over the 500 GB pile.

Note: `_open_csv` is called twice per month in production (once from
`_count_data_rows`, once from `_iter_ticks` via `_ingest_month_atomic`),
so an affected month emits two log lines, not one. Accepted: the
duplication accurately reflects two distinct call sites; module-level
dedup state would violate "no abstractions for hypothetical future
requirements." Per-month affected count is recoverable via
`grep -c '"zip contains"' logs/ingest_full.log` divided by 2.

**Why safe**:
- Zip-level SHA256 (DR v3.0.2 §3 step 2) unchanged — same archive contract
- Two CSVs are bit-identical per CRC32 — picking either loads same data
- Root-level filename is Binance's canonical convention; nested paths
  are packaging artifacts
- Future divergence would surface in the per-month agg_id density check
  during STEP 3's sanity report

**Why NOT validate-both-copies-match at runtime**:
- Adds a full-decompress pass per affected month (multi-GB)
- Zip-SHA already proves archive is what Binance shipped
- Keep the loader simple

**Smoke-test result** (2021-12 zip, no DB write):
- Multi-CSV log line fires with correct format
- Chosen file: `BTCUSDT-aggTrades-2021-12.csv` (root-level)
- `_count_data_rows` and `_iter_ticks` both yield 32,269,900 — exact match
- First 5 ticks parse cleanly; prices ~$56,950 (consistent with Dec 1
  2021 BTC near $57k); ts in expected range
- Magnitude plausible for late-2021 BTC

**Resume plan**: idempotency (DR v3.0.2 §3) skips the 35 already-done
months instantly; sweep resumes from 2021-12. ETA for remaining 53
months: ~3 hr at observed rates.

**Approver**: User (`silverspoon0099`) — approved 2026-05-06; chose
"behavior B" (no module-level dedup) for the multi-CSV log line.

**References**: DR v3.0.2 §3; failure log `logs/ingest_full.log`;
smoke-test transcript (this conversation).

---

## 2026-05-05 — Decision v3.0.2 — Phase 0.1 raw tick ingestion contract (DR)

**Context**: Spec §6 defines the CUSUM-bar table (`events.bars_btc_cusum`) but
does not specify the raw aggTrades landing table, the Timescale chunking
policy for tick-scale volume, or the idempotency contract for a multi-month
loader. ~500 GB of BTCUSDT aggTrades 2019-01 → present need a deterministic,
resumable pipeline before `data/ingest_ticks.py` is written. This DR fills
those gaps; no frozen Phase A parameter (§10.1) is touched.

**Decisions**:

### 1. Raw tick table — `events.ticks_btc`

```sql
CREATE TABLE events.ticks_btc (
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
```

`agg_id` is Binance's aggregate-trade ID — unique and monotonic per symbol.
It is the idempotency key (see §3 below). No surrogate key, no checksum
column. PK is `(agg_id, ts)` not `(agg_id)` alone — Timescale requires the
partitioning column in any unique constraint.

`quote_qty` is **computed at insert time as `price * qty`**. The Binance
Vision spot aggTrades CSV publishes 8 columns
(`a, p, q, f, l, T, m, M` — agg_id, price, qty, first_trade_id,
last_trade_id, timestamp_ms, is_buyer_maker, was_best_match) and does not
include a quote-quantity field. Storing it materialized lets sanity queries
(`sum(quote_qty)` by date) avoid recomputing the product across hundreds of
millions of rows. The 8th CSV column (`was_best_match`) is dropped — not
useful for our purposes.

### 1b. Ingest audit table — `events.ingest_log`

```sql
CREATE TABLE events.ingest_log (
    symbol         TEXT        NOT NULL,
    month          DATE        NOT NULL,
    sha256         TEXT        NOT NULL,
    expected_rows  BIGINT      NOT NULL,
    actual_rows    BIGINT      NOT NULL,
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, month)
);
```

One row per (symbol, month) successfully ingested. The `sha256` column
stores the value from the Binance `.CHECKSUM` sidecar — if Binance later
republishes a month with corrections, the SHA will differ from what's
logged and the loader can detect the change and force re-ingestion. Without
this audit table there is no record of which archive version is currently
in the DB. The loader writes this row in the same transaction as the data
`COPY` (see §3) — atomic with the ingest itself.

### 2. TimescaleDB hypertable + compression

```sql
SELECT create_hypertable(
    'events.ticks_btc', 'ts',
    chunk_time_interval => INTERVAL '7 days'
);

ALTER TABLE events.ticks_btc SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'is_buyer_maker',
    timescaledb.compress_orderby   = 'ts, agg_id'
);

SELECT add_compression_policy('events.ticks_btc', INTERVAL '30 days');
```

**Why 7-day chunks** (not the 180-day default in [config.yaml:18](config.yaml#L18),
which is for bar tables): at ~500 GB / ~6.5 yr ≈ 77 GB/yr ≈ 1.5 GB/week of
uncompressed ticks. 7-day chunks keep each chunk in the low-GB range so
ALTER/REINDEX/compression operations stay tractable; 180-day chunks would
produce 40+ GB chunks that block on any maintenance op.

**Why compress after 30 days**: only the most recent ~30 days are read at
tick resolution during CUSUM-bar regeneration sweeps; older data is
read-only after the first bar build. Timescale native columnar compression
typically achieves 10–20× on tick data — drops the on-disk footprint from
~500 GB to ~25–50 GB.

`config.yaml` key `database.chunk_interval_bars` is left as-is (it
governs the future bar table). Two new keys will be added to `config.yaml`
as part of this DR's implementation: `database.chunk_interval_ticks: "7 days"`
and `database.compress_after_ticks: "30 days"`.

### 3. Idempotency contract

**Loop bound — complete months only**: the loader processes months strictly
older than the current UTC calendar month. The in-flight current month is
never ingested, to avoid partial-month data. For example, on 2026-05-05 UTC
the loader processes 2019-01 through 2026-04 inclusive; 2026-05 becomes
eligible on 2026-06-01 UTC.

Per-month, the loader:

1. Downloads `BTCUSDT-aggTrades-YYYY-MM.zip` and the sibling
   `BTCUSDT-aggTrades-YYYY-MM.zip.CHECKSUM` from Binance Vision.
2. Verifies the SHA256 of the `.zip` matches the `.CHECKSUM` file.
   On mismatch: abort, log, do not ingest.
3. Counts expected rows: `expected_count = wc -l <unzipped CSV>`
   (Binance CSVs are headerless).
4. Queries actual rows already in the DB for that month:

   ```sql
   SELECT COUNT(*) FROM events.ticks_btc
   WHERE ts >= :month_start AND ts < :next_month_start;
   ```

5. Looks up the prior ingest log entry:
   `SELECT sha256 FROM events.ingest_log WHERE symbol=:s AND month=:m`.
6. **Skip rule**: if `actual >= expected` AND the logged `sha256` matches
   the current `.CHECKSUM` value → log "month YYYY-MM already complete"
   and continue to the next month.
7. **Force-re-ingest rule**: if `actual >= expected` but logged `sha256`
   differs from current `.CHECKSUM` → treat as a Binance-republished
   archive; proceed to step 8.
8. Run a **single transaction** per month:
   - `DELETE FROM events.ticks_btc WHERE ts >= :month_start AND ts < :next_month_start`
   - `COPY events.ticks_btc (...) FROM STDIN` for the full month
   - `INSERT INTO events.ingest_log (...) VALUES (...)
      ON CONFLICT (symbol, month) DO UPDATE SET
        sha256        = EXCLUDED.sha256,
        expected_rows = EXCLUDED.expected_rows,
        actual_rows   = EXCLUDED.actual_rows,
        ingested_at   = now()`
   - commit

No partial commits, no batch-by-batch resume *within* a month — the unit
of resumability is one calendar month. Loader is safe to re-run after any
kind of crash.

### 4. Sanity-check refinement

The Appendix A row "no gaps > 1 hr" is refined to operate on minute-
aggregated tick volume:

```sql
WITH minute_bins AS (
    SELECT date_trunc('minute', ts) AS m
    FROM events.ticks_btc
    GROUP BY 1
),
gaps AS (
    SELECT m, m - LAG(m) OVER (ORDER BY m) AS gap
    FROM minute_bins
)
SELECT m, gap FROM gaps WHERE gap > INTERVAL '1 hour';
```

Rationale: raw inter-tick gaps of seconds-to-minutes are normal in low-vol
periods (e.g. weekend Asia-session lulls in 2019) and would flood any
"no gap > 1 hr" check with false positives. The interesting signal is
*minutes with zero trade activity* clustered together — a continuous
60-minute window with no ticks indicates a real outage (exchange downtime,
archive truncation, or a missed month).

Other Phase 0.1 sanity outputs (unchanged from Appendix A):
- Total tick count by month
- Daily volume distribution (`sum(quote_qty)` by date)
- First/last `agg_id` per month, confirm strictly monotonic across the full
  range with no resets

### 5. `.env` credentials — symlink confirmed

The shared-DB decision is Decision v2.27 (30m repo); credentials already
exist in v1.0's `.env` at `/nvme1/projects/trading/ml-bot/.env`. User
confirmed symlink rather than duplicate:

```bash
ln -s /nvme1/projects/trading/ml-bot/.env \
      /nvme1/projects/trading/hyperliquid-ml-bot-events/.env
```

Single source of truth; rotation in v1.0 propagates automatically; no risk
of v3.0 drifting onto stale creds. Trade-off accepted: portability to a
non-VPS environment (e.g. local Windows box) requires manual credential
copy — out of Phase 0.1 scope.

### 6. Appendix A clarification (no spec edit)

The Appendix A row under Phase 0.1 — "Postgres schema
`events.bars_btc_cusum` created" — is a checklist mislabel. That table
holds CUSUM-bar output and is the artifact of Phase 0.2 (`bars/cusum.py`).
Spec §6.3's schema definition is correct and stays put. Phase 0.1's table
artifacts are `events.ticks_btc` and `events.ingest_log` per §1 and §1b
of this DR. No edit to the spec body is required; this DR is the
canonical reference for anyone reading Appendix A.

---

**Approver**: User (`silverspoon0099`) — approved 2026-05-05 in
conversation, with two folds: (a) `events.ingest_log` audit table,
(b) complete-months-only loop bound. Both folded in above.

**References**:
- Spec §6.1, §6.2, §6.3, §15, Appendix A (Phase 0.1)
- 30m repo Decision v2.27 (shared DB)
- Spec §10.1 — frozen parameters NOT modified by this DR
- TimescaleDB docs: `create_hypertable`, `add_compression_policy`
- Binance Vision archive: per-zip `.CHECKSUM` SHA256 sidecar files

---

## 2026-05-05 — Decision v3.0.1 — Architecture finalized

**Decision**: Three-layer model
- L0: LightGBM pass-gate (fast premise check)
- L1: ResNet-LSTM primary (NOT Transformer per Lessmann §"Conclusions")
- L2: LightGBM meta-filter (Phase B only)

Frozen Phase A parameters (no tuning until DR):
- CUSUM threshold = 0.02
- Triple-barrier tp/sl = 0.05 / 0.05 (symmetric)
- Vertical barrier = 24 bars
- Confidence threshold = 0.60 (long if P>0.60, short if P(SHORT)>0.60)
- Costs = 11 bps round-trip (3.5 bps Hyperliquid taker + 2 bps slippage, both sides)

**Approver**: User (`silverspoon0099`)

**Reference**: Spec §3.3, §5.2, §10.1.

---

## 2026-05-05 — Decision v3.0.0 — Project inception

**Context**: 30m v2.0 Phase 2.2 multi-asset OOT FAIL (BTC pre-gate 0.9913, SOL 1.0027, LINK 1.0041 — all ≥ 1.0 random-prior gate per 30m repo Decision v2.69). User and Claude conducted Phase B-bis literature review on 2026-05-05 covering 3 PDFs + 4 URLs:

| Source | TF | Verdict |
|---|---|---|
| Performer+BiLSTM (arxiv 2403.03606) | daily | Methodologically flawed (R²=0.99 = autocorrelation) |
| Meta-RL-Crypto (arxiv 2509.09751) | daily | LLM agents; Sharpe 0.30 bull, −0.05 bear; not deployable |
| PMformer (arxiv 2512.04099) | daily | "Disconnect between accuracy and trading utility" — ETH Sharpe −0.84 |
| Medium meta-labeling (Nguyen) | volume bars | Sound architecture; no costs modeled |
| **Lessmann 2025** (Springer FinInnov) | **CUSUM bars** | **+91.6% ETH, +20.4% BTC after costs** ← keystone |
| ScienceDirect (Izadi/Hajizadeh) | daily | 57% accuracy; no Sharpe; paywalled body |
| NIH PMC (TFT) | daily | No walk-forward, no costs; research-only |

**Decision**:
- Fresh rewrite, new repo at `c:/Users/1/Documents/Workspace/trading/hyperliquid-ml-bot-events/`
- No inherit-and-patch from 30m v2.0 repo
- Architecture anchored to Lessmann: CUSUM filter + triple-barrier + ResNet-LSTM
- BTC-only Phase A; SOL/LINK conditional on Phase A pass
- Spec `Project Spec EventBars.md` v3.0.0 created

**Stop conditions**:
- Phase A pre-gate fails after 1 parameter sweep → ship signal-provider mode (alerts only)
- Phase A passes pre-gate but cost-adj Sharpe < 1.0 → Phase B sweep then signal-provider
- Phase B fails on SOL+LINK → ship BTC-only bot

**Approver**: User (`silverspoon0099`)

**References**:
- Spec §1, §2.3, §3
- 30m repo (lineage): `../hyperliquid-ml-bot-30m/PROJECT_LOG.md` Decision v2.69
- Memory: `~/.claude/.../memory/reference_springer_lessmann_paper.md`
- Local paper extracts: `../TradingView/research/paper_springer.txt`, `paper3.txt`

---

(future entries below this line)
