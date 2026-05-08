# PROJECT_LOG — ml-bot-events (v3.0)

> Append-only decision log. Newest entry at top.
> Every code change must reference a Decision or DR here.

---

## 2026-05-07 — Decision v3.0.7 — Phase 0.3 feature builder contract (DR)

**Context**: Phase 0.3 implements `features/builder.py` per spec §7.1
(replicate Lessmann's 33-feature set), reading from
`events.bars_btc_cusum` and writing to parquet. Spec leaves several
mechanics unspecified — this DR pins them. The §7.2-frozen 33-feature
list is NOT modified, only its concrete column names + semantics.

**Decisions**:

### 1. Output destination — parquet only

`data/storage/features/features_btc.parquet`. Per spec §15 +
config.yaml:59. At 18,629 × 35 cols × float64 ≈ 5 MB, the DB pays
hypertable overhead for no downstream benefit. Per-asset filename
matches future Phase B SOL/LINK pattern.

### 2. Builder produces RAW features only

Per-fold z-score standardization (spec §7.1) is a trainer concern
(`cv/walk_forward.py` Phase 1). Builder writes raw, unstandardized.

### 3. Warmup handling — NaN at builder, filter at trainer

Indicator implementations return NaN for early bars before each
indicator ramps up. Builder writes NaN through; trainer drops the
first 100 bars per config.yaml:58 `features.warmup_bars: 100`.
Rejected: forward-fill (loses info), builder-side drop (couples warmup
to wrong stage).

### 4. "EMA + std" — std of CLOSE over the same N-period window

`pandas.Series.rolling(N).std()` on close prices — NOT std of EMA
values. Lessmann pairs an EMA trend measure with a rolling-volatility
measure of the underlying.

### 5. Exact 33 feature names (snake_case)

| # | Name | Input | Source / formula |
|---|------|-------|------------------|
| 1–5  | ema_{5,10,15,20,50} | close | EMA, α = 2/(N+1) |
| 6–10 | std_{5,10,15,20,50} | close | `close.rolling(N).std()` |
| 11   | macd_line   | close | EMA(12) − EMA(26) |
| 12   | macd_signal | close | EMA(9) of macd_line |
| 13   | macd_hist   | close | macd_line − macd_signal |
| 14–16 | rsi_{6,10,14} | close | Wilder, α = 1/N |
| 17   | stoch_k    | h,l,c   | %K(14) smoothed by 3 |
| 18   | stoch_d    | h,l,c   | SMA(3) of stoch_k |
| 19   | williams_r | h,l,c   | −100 · (HH−c)/(HH−LL) over 14 |
| 20   | bb_upper   | close   | SMA(5) + 2.0·std(5) |
| 21   | bb_lower   | close   | SMA(5) − 2.0·std(5) |
| 22   | ret_1      | close   | `np.log(close / close.shift(1))` |
| 23   | cmf_21     | h,l,c,v | sum(MFV,21) / sum(v,21) |
| 24   | mfi_14     | h,l,c,v | 100 − 100/(1+pos/neg) |
| 25   | hour_sin   | bar_close_ts | sin(2π·h/24) |
| 26   | hour_cos   | bar_close_ts | cos(2π·h/24) |
| 27   | dow_sin    | bar_close_ts | sin(2π·d/7) |
| 28   | dow_cos    | bar_close_ts | cos(2π·d/7) |
| 29   | bar_duration_sec | open_ts/close_ts | `(close − open).total_seconds()` |
| 30   | n_trades   | bars passthrough | (cast float64) |
| 31   | volume     | bars passthrough | base-asset BTC, per DR v3.0.5 §7 |
| 32   | cusum_pos  | bars passthrough | |
| 33   | cusum_neg  | bars passthrough | |

Plus 2 key columns prepended (NOT features, identifying):
- `bar_id` — int64, BIGSERIAL from bars_btc_cusum
- `bar_close_ts` — datetime64[ns, UTC]

Total parquet schema: **35 columns** (2 keys + 33 features).

### 6. sin/cos encoding

```python
h = bar_close_ts.hour          # 0..23
d = bar_close_ts.weekday()     # 0=Mon .. 6=Sun (Python convention)
hour_sin, hour_cos = sin(2π·h/24), cos(2π·h/24)
dow_sin,  dow_cos  = sin(2π·d/7),  cos(2π·d/7)
```

`bar_close_ts` is the time anchor (not `bar_open_ts`) — the moment a
bar's signal becomes available for action.

### 7. dtype

`bar_id`: int64. `bar_close_ts`: datetime64[ns, UTC]. All 33 features:
float64 (including `n_trades` cast — keeps feature matrix homogeneous).

### 8. Bar read order — `ORDER BY bar_close_ts, bar_id`

bars_btc_cusum PK is `(bar_id, bar_close_ts)`; we read chronologically.
`bar_id` is monotonic with insert order (BIGSERIAL during the rebuild)
and matches `bar_close_ts` ordering for all but the same-ts cascade
bars (DR v3.0.6); tie-breaking on `bar_id` gives a total deterministic
order.

### 9. Reproducibility — md5 fingerprint

Same pattern as DR v3.0.6: at end of build, compute md5 over the
canonicalized feature matrix (sorted by bar_id, all columns
text-serialized) and log it. Re-runs produce identical fingerprints.

### 10. pandas-ta — fallback (b) taken

First attempt: `pandas-ta==0.3.14b0` against `numpy==2.1.3`.

**Outcome (2026-05-07)**: pandas-ta is unavailable for our environment
— `pandas-ta==0.3.14b0` is not on PyPI for Python 3.10.12, and the
newer 0.4.x line (0.4.67b0, 0.4.71b0) requires Python ≥3.12.
Per-user-decided fallback path **(b)** is taken: the 11 distinct
indicator types (EMA / std / MACD / RSI-Wilder / Stoch / Williams %R /
Bollinger / log-return / CMF / MFI / sin-cos seasonality) are
implemented directly in `features/builder.py` using pandas + numpy
(~150 lines total). `requirements.txt` does NOT include pandas-ta.

The four golden-value tests (§13) validate the hand-rolled
implementation against canonical formulas: RSI(14) Wilder, EMA(20)
α=2/(N+1), MACD hist consistency, and BB symmetry.

Rejected alternatives: (a) downgrading numpy is regressive — psycopg +
future ML libs want 2.x; (c) coming-back-to-ask was avoidable since
(b) is the only sensible long-term answer the user had pre-authorized.

### 11. CLI surface

```
python -m features.builder                  # full build
python -m features.builder --dry-run        # in-memory, print summary, no file
python -m features.builder --month YYYY-MM  # smoke (single month)
```

Match the `bars.cusum` CLI shape.

### 12. Sanity report (post-build)

- 35 columns present, in the order specified above
- Row count == bar count (18,629)
- **Explicit assertion** (per user fold): `assert features_df.iloc[50:]
  .isna().sum().sum() == 0` — NaN density drops to 0 after the longest
  ramp (ema_50 / std_50, needs 50 bars). Catches silent ramp-up bugs.
- Feature range plausibility: rsi ∈ [0, 100], williams_r ∈ [-100, 0],
  sin/cos ∈ [-1, 1]
- md5 fingerprint of the full matrix

### 13. Test fixtures — synthetic + 4 golden-value/consistency tests

Synthetic-OHLCV fixture set:
- Expected shape (35 columns × N rows)
- NaN counts at warmup edges
- Indicator range plausibility
- Determinism (run twice → identical output)

Plus four golden-value / consistency tests (per user 2026-05-07 fold):
- **RSI(14) golden**: hand-compute expected RSI at bars 14, 20, 25
  from a deterministic synthetic close series; assert builder output
  matches within float64 epsilon. RSI carries the highest definition-
  drift risk (Wilder vs simple smoothing varies across libraries).
- **EMA(20) golden**: hand-compute EMA at bars 20, 30, 40 with
  α = 2/(N+1); assert match.
- **MACD hist consistency**: assert `macd_hist == macd_line −
  macd_signal` exactly (derived column; no float epsilon).
- **BB symmetry**: assert `(bb_upper − sma_5) ≈ (sma_5 − bb_lower)`
  within epsilon — symmetric ±2σ around the SMA-5 midpoint.

These earn their place twice: catch pandas-ta version drift if path
(a) is taken; validate the hand-rolled implementation against
canonical formulas if path (b) is taken.

### Implementation surface (informational, not a contract)

- `features/builder.py`: indicator functions, `load_bars()`,
  `build_features(bars_df)` returning the 35-column DataFrame,
  `write_parquet(df)`, `sanity_report()`, CLI.
- `features/tests/test_builder.py`: synthetic fixture set + 4 golden
  tests above.

**Approver**: User (`silverspoon0099`) — approved 2026-05-07; three
folds: §12 explicit warmup assertion, §10 decided fallback path (b),
§13 four golden-value tests.

**References**: Spec §7.1, §7.2, §7.3, §15; config.yaml:57-69;
DR v3.0.5 (bars source schema), DR v3.0.6 (sub-µs ordering);
Lessmann §"Feature engineering".

---

## 2026-05-07 — Decision v3.0.6 — Drop UNIQUE(bar_close_ts, threshold_pct) on bars_btc_cusum (DR)

**Context**: DR v3.0.5 STEP "full sweep" failed at the bulk INSERT step
after a 6.5-hour clean scan (3.87B ticks → 18,629 bars → 0 force-closed):

    psycopg.errors.UniqueViolation: duplicate key value violates unique
    constraint "bars_btc_cusum_bar_close_ts_threshold_pct_key"
    DETAIL:  Key (bar_close_ts, threshold_pct)=
        (2019-06-26 20:35:08.818+00, 0.02) already exists.
    CONTEXT:  COPY bars_btc_cusum, line 894

Diagnostic (`--dry-run --month 2019-06`): 512 bars total, 7 close_ts
values shared by >1 bar (8 extra bars from duplication). All cluster on
2019-06-26 20:35 UTC during BTC's $13.8k → $12.6k flash crash. Sample
triple-bar at `20:35:08.818000+00:00`:

  - bar 1: 20:35:03.437 → 20:35:08.818, 1,691 ticks, $13340→$13134
  - bar 2: 20:35:08.818 → 20:35:08.818, 1,095 ticks, $13134→$12873
  - bar 3: 20:35:08.818 → 20:35:08.818,   663 ticks, $12871→$12644

Bars 2 and 3 each contain hundreds of aggregate trades, all stamped to
the same ts (Binance matcher's sub-µs sequencing during cascades). Each
is a legitimate, ordered, distinct CUSUM crossing — real microstructure,
not an algorithm artifact.

State at failure: bars table empty (atomic DELETE+COPY rolled back as
designed). Re-run required.

**Decisions**:

### 1. Drop UNIQUE(bar_close_ts, threshold_pct)

```sql
ALTER TABLE events.bars_btc_cusum
DROP CONSTRAINT bars_btc_cusum_bar_close_ts_threshold_pct_key;
```

Update `_DDL_BARS_BTC_CUSUM` in `data/db.py` to omit the UNIQUE clause.
Bar identity is fully captured by `bar_id` alone (BIGSERIAL, part of
the existing PK `(bar_id, bar_close_ts)`).

### 2. Authorize spec §6.3 edit

Remove the `UNIQUE(bar_close_ts, threshold_pct)` line from the §6.3
schema and strip the now-trailing comma on `threshold_pct`. Same
single-line spec-edit pattern as DR v3.0.5 §3 (diagram alignment).
No semantic change to the spec body.

### 3. Why safe

- The atomic DELETE + COPY + commit semantics from DR v3.0.5 §1 already
  prevent accidental double-insertion across rebuilds — the UNIQUE was
  redundant defense.
- `bar_id` PK provides unique identity per bar.
- Bars sharing `bar_close_ts` are semantically valid: each is a
  separate CUSUM crossing within a single ts grain. Ordering by
  `(bar_close_ts, bar_id)` preserves chronology.

### 4. Why NOT change the algorithm to coalesce same-ts bars

Two alternatives considered and rejected:

(a) Force the next bar's close to be strictly later than the previous
    bar's close. Would artificially stretch bars during cascades —
    losing the genuine microstructure the algorithm is detecting.
(b) Coalesce same-ts emits into one larger bar. Loses the directional
    information (e.g., the triple-bar cascade above is three separate
    SHORT bars; coalesced into one would mask the magnitude of the move).

The DB schema must accommodate the data the algorithm produces, not the
other way around.

### 5. Re-run cost; recovery options explicitly deferred

The 6.5-hour scan must be repeated — the in-memory bars list was lost
when the failed transaction rolled back the process. Two recovery
options are deferred:

- **Disk-backed parquet checkpoint** (write bars after scan, before
  INSERT): user decision — UNIQUE violation was the realistic failure
  mode; remaining modes (disk full, conn timeout, OOM) aren't well
  mitigated by parquet anyway. 30-line code + permanent artifact-
  lifecycle burden was a poor ratio for now.
- **Per-month atomic commits** (mirror DR v3.0.2 §3 pattern for bars):
  substantive contract change to DR v3.0.5 §1 ("full atomic rebuild").
  Earns its place if a fourth failure surfaces during the re-run — new
  DR before the fifth attempt.

### 6. Bar-count target — there is no validated target

The 18,629 bars from the failed-INSERT scan (≈6.96 bars/day across
2019-01..2026-04) sits in the low end of Lessmann's qualitative 5–20
bars/day range. The earlier "33k–47k extrapolation" was speculative:
Lessmann's paper does not tabulate raw per-threshold bar counts for
BTC CUSUM 2% in the figures we have; the 25–35k anchor attributed to
him was an approximation, not a measured value. Treat the bar count
from the next clean run as the empirical truth from our data, not a
deviation requiring investigation. The post-sweep sanity report
(`bars/cusum.py:sanity_report`) reports bar count + per-month density
observationally, without comparing to any target.

**Approver**: User (`silverspoon0099`) — approved 2026-05-07; one fold
added (§6 bar-count honesty); checkpoint and per-month-commits options
explicitly deferred per user direction.

**References**: DR v3.0.5 §1 (atomic rebuild contract), spec §6.3
(schema), failure log `logs/bars_full.log` (archived as
`logs/bars_full.log.20260507-failed-uniqueviolation`), diagnostic
dry-run on 2019-06.

---

## 2026-05-07 — Decision v3.0.5 — Phase 0.2 CUSUM bar construction contract (DR)

**Context**: Phase 0.2 implements `bars/cusum.py` per spec §6.4 algorithm,
writing into `events.bars_btc_cusum` per §6.3, with the §10.1-frozen
CUSUM threshold of 0.02 for BTC. The spec leaves several mechanics
unspecified — this DR pins them. The §10.1 frozen Phase A parameter
(CUSUM threshold = 0.02) is NOT touched.

**Decisions**:

### 1. Resumability — full rebuild

`bars/cusum.py` rebuilds the entire bar series from scratch on every
run: `DELETE FROM events.bars_btc_cusum WHERE threshold_pct = :t` →
single-pass tick scan → bulk INSERT. All in one transaction.

Why rebuild over incremental:
- Bar count is small (~25–35k for BTC over 6.5 yr per Lessmann); a
  full rebuild on the existing 3.87B ticks runs once on Phase 0.2 pass,
  then again only when ticks update or threshold changes.
- Atomicity: DELETE + INSERT in one tx → no partial state if a crash
  hits mid-run; same idempotency-by-rebuild discipline as the tick
  loader's per-month tx (DR v3.0.2 §3 step 8).
- Incremental is stateful (resume from `max(bar_close_ts)`, recover
  in-flight `s_pos`/`s_neg`), error-prone, and offers no payoff at
  this scale.

For Phase B (multi-asset / threshold sweep) this becomes
"per-(asset, threshold)" rebuild — same pattern, parameterized.

### 2. Force-close bar marking — derived from cusum_pos / cusum_neg

Per §6.5, a bar is force-closed when no 2% CUSUM move occurred in 168 h.
The spec table already includes `cusum_pos` and `cusum_neg` at close;
the close reason is recoverable:

    close_reason = 'cusum'   if max(cusum_pos, -cusum_neg) >= threshold_pct
                 = 'timeout' otherwise

No schema change. Recoverable in one query:

    SELECT bar_id,
           CASE WHEN GREATEST(cusum_pos, -cusum_neg) >= threshold_pct
                THEN 'cusum' ELSE 'timeout' END AS close_reason
    FROM events.bars_btc_cusum;

Rejected: adding `close_reason TEXT CHECK IN ('cusum','timeout')`. The
extra column is explicit but spec-divergent for negligible benefit at
~30k rows.

### 3. Output destination — DB only

`events.bars_btc_cusum` is canonical (per §6.3). The §5.1 ASCII diagram
mention of `bars_BTC.parquet` is illustrative; parquet snapshot is
NOT produced in Phase 0.2.

**Spec edit authorized**: §5.1 ASCII diagram annotation
`→ bars_BTC.parquet` is updated to `→ events.bars_btc_cusum` in the
same commit as this DR. No semantic change to the spec body — pure
annotation alignment to the §6.3 contract.

### 4. Streaming pattern — `COPY (...) TO STDOUT (FORMAT BINARY)` + Python CUSUM loop

3.87B ticks cannot be loaded into RAM. Read pattern:

```python
with cur.copy(
    "COPY (SELECT ts, price, qty FROM events.ticks_btc "
    "ORDER BY ts, agg_id) TO STDOUT (FORMAT BINARY)"
) as copy:
    copy.set_types(["timestamptz", "float8", "float8"])
    builder = CusumBuilder(threshold, max_duration_h=168)
    for ts, price, qty in copy.rows():
        bar = builder.step(ts, price, qty)
        if bar is not None:
            bars.append(bar)
```

Why COPY over server-side cursor: ~2–3× faster on bulk reads, simpler
loop, no cursor housekeeping. Single linear pass, monotonic stateful
iteration. Bars are accumulated in an in-memory Python list (~30k rows
× ~12 cols ≈ a few MB) and bulk-inserted via binary `COPY ... FROM
STDIN` at end of scan, inside the same DELETE+INSERT transaction.

If pure-Python iteration over 3.87B rows proves too slow in practice
(likely 1–3 hours), `numba @njit` on the inner loop is a drop-in
optimization — defer until measurement justifies.

### 5. Warmup — none at bar stage

Bar construction emits every bar from the first tick onward. The
[config.yaml:58](config.yaml#L58) `warmup_bars: 100` parameter belongs
to feature engineering (§7) — the feature builder drops the first 100
bars to absorb its EMA/MACD/RSI ramp-up. Bar construction itself has
no warmup; the first bar takes however many ticks are needed for
either CUSUM ≥ 0.02 or 168 h to elapse.

### 6. Tick ordering — `ORDER BY ts, agg_id`

§6.5 demands bit-identical output across re-runs. Tick `ts` is not
strictly monotonic — sub-microsecond timestamps collide (sample from
2026-02: agg_id 3856672511 had ts=00:00:00.008822, multiple ticks
within the same `ts` value are routine). Tie-break on `agg_id` ASC.
`(ts, agg_id)` is unique by construction (the table's PK columns), so
ordering is total and deterministic.

### 7. Volume semantic — `SUM(qty)` (base-asset BTC)

Spec §6.3 column `volume DOUBLE PRECISION` is ambiguous. Decision:
volume is base-asset units (BTC), i.e. `SUM(qty)` over the bar's
ticks. Matches Lessmann §"Bar construction"; matches the v2.0 30m
project; quote-asset (USDT) volume is recoverable as `SUM(quote_qty)`
from a JOIN if ever needed.

### 8. Bars hypertable — chunk_interval 180 days; PK includes partitioning column

[config.yaml:18](config.yaml#L18) sets `chunk_interval_bars: "180 days"`,
which conflicts with §6.3's default-7-days `create_hypertable` call.
For ~30k total bars over 6.5 yr (~13 chunks at 180 d vs ~340 chunks at
7 d), 180-day chunks are appropriate. Pass
`chunk_time_interval => INTERVAL '180 days'` explicitly.

PK in §6.3 spec is `bar_id` alone, but Timescale requires the
partitioning column in every uniqueness constraint. Use
`PRIMARY KEY (bar_id, bar_close_ts)` — same pattern we used for
`events.ticks_btc` PK `(agg_id, ts)` in DR v3.0.2 §1. The
`UNIQUE(bar_close_ts, threshold_pct)` from the spec is preserved
unchanged (already includes `bar_close_ts`).

Compression policy for bars: defer to Phase 0.3+. Bars table is
small enough that compression isn't load-bearing.

### Implementation surface (informational, not a contract)

- `data/db.py`: extend `init_schema()` to also create
  `events.bars_btc_cusum` and the bars hypertable per §8. Signature
  renamed: `init_schema(chunk_interval_ticks, compress_after_ticks,
  chunk_interval_bars)` — old generic `chunk_interval` parameter is
  now ticks-specific by name. Two existing callers in `data/ingest_ticks.py`
  updated.
- `bars/cusum.py`: `CusumBuilder` class (stateful, testable) +
  `cusum_bars(ticks_iter, threshold, max_duration_h)` generator +
  `build_bars(threshold, month_filter, dry_run)` DB driver +
  CLI: `python -m bars.cusum [--month YYYY-MM] [--dry-run]`.
- `bars/tests/test_cusum.py`: pytest fixtures covering the agreed set:
  empty input, single tick, threshold edge ≥, threshold edge < ε,
  monotonic up, monotonic down, sideways force-close, mixed walk
  invariants, reset after CUSUM-triggered close (asserts
  `s_pos == s_neg == 0`), explicit OHLC/volume/n_trades, cusum_pos/neg
  at close == trigger state, determinism (same input → identical bars).

### Sanity checks (Phase 0.2 post-build)

- Total bar count plausible: ~25k–40k (Lessmann anchor 25k–35k for BTC
  CUSUM 2% over 6.5 yr — adjust upward for our extra year + the
  high-vol 2022Q3–2023Q1 segment)
- Median bars/day per regime: high-vol 2022Q3 → 20+/d; low-vol 2023Q2
  → 3–5/d
- All bars: `n_trades >= 1`, `high >= max(open,close)`, `low <=
  min(open,close)`, `volume > 0`
- Force-closed bars (max(cusum_pos, -cusum_neg) < threshold): all have
  duration ≈ 168h (zero or few in well-traded BTC)
- No bar duration > 168h
- Determinism: re-run, compare row count + ordered hash → identical

**Approver**: User (`silverspoon0099`) — approved 2026-05-07; all 8
decisions accepted; one fold added (spec §5.1 diagram annotation
update authorized in §3); test fixtures extended per user's add list.

**References**: Spec §6.3, §6.4, §6.5, §10.1, §15;
[config.yaml:48-53](config.yaml#L48-L53); DR v3.0.2 §1
(events.ticks_btc — source); Lessmann §"CUSUM filter and range bars".

---

## 2026-05-06 — Decision v3.0.4 — Loader fix: dedup Binance source-data duplicates (DR)

**Context**: After DR v3.0.3 patched the multi-CSV archive case, STEP 3
sweep continued and failed at month 51 (2026-02) with a fresh failure
mode: `psycopg.errors.UniqueViolation: duplicate key value violates
unique constraint "ticks_btc_pkey", Key (agg_id, ts)=(3856672511,
2026-02-11 00:00:00.008822+00) already exists`. The 2026-02 BTCUSDT
aggTrades CSV contains internal duplicate rows — a Binance publishing
artifact:

    total CSV data rows:           52,474,665
    unique agg_ids:                52,471,665
    agg_ids that appear >1 time:   2,000   (1,000 at 2x; 1,000 at 3x)
    extra rows from dupes:         3,000   (0.006% of file)

Sample: agg_id `3856672511` appears 3× at lines 25,562,325 / 25,564,325
/ 25,565,325 — bytes-identical (same price, qty, ts, trade IDs, flags).
The "1,000 of each multiplicity" pattern looks like a batch-processing
artifact, not random corruption — likely affects more recent months too
as the publisher continues operating.

State at failure: 85 of 88 months done cleanly (2019-01..2026-01); 0
rows in DB for 2026-02 (atomic rollback per DR v3.0.2 §3 step 8 worked
again); resumable.

**Decisions**:

### 1. Staging-table dedup pattern in `_ingest_month_atomic`

Replace direct `COPY events.ticks_btc FROM STDIN` with:

1. `CREATE TEMP TABLE _staging_ticks (LIKE events.ticks_btc) ON COMMIT DROP`
   — `LIKE` copies columns + NOT NULL but NOT the PK; staging accepts
   duplicate `(agg_id, ts)` rows.
2. `COPY _staging_ticks ... FROM STDIN WITH (FORMAT BINARY)` — same
   binary COPY as before, but into the no-PK staging table.
3. `INSERT INTO events.ticks_btc SELECT … FROM _staging_ticks
   ON CONFLICT (agg_id, ts) DO NOTHING` — Postgres handles dedup; first
   row wins, subsequent occurrences silently dropped. `cur.rowcount` =
   post-dedup count.
4. `INSERT INTO events.ingest_log` (unchanged), then commit.

All steps in one transaction. Same atomicity contract as DR v3.0.2 §3.
TEMP table auto-drops on commit/rollback.

Rejected alternatives:
- Python-side dedup set: ~2 GB memory for 30M+ agg_ids; defeats COPY's
  memory-efficiency.
- Sliding-window dedup: assumes locality of duplicates (false — sample
  shows dupes 1,000–2,000 lines apart).
- Schema column for `staged_rows`: extra column, recoverable from
  `expected_rows - actual_rows`.

### 2. Three contract changes

a) **`actual_rows` in `events.ingest_log` is post-dedup count** (was raw
   COPY rowcount). The 85 already-done months had no dupes (else the
   pre-DR-v3.0.4 code would have failed on them, as 2026-02 did) so
   their `actual_rows == expected_rows`. New months with dupes will
   have `actual_rows < expected_rows`. Diagnostic query:

   ```sql
   SELECT month, expected_rows, actual_rows,
          expected_rows - actual_rows AS source_dupes
   FROM events.ingest_log
   WHERE actual_rows < expected_rows
   ORDER BY month;
   ```

b) **Hard assertion `actual == expected` relaxes to `actual <= expected`**.
   Hard-fail only if `actual > expected` (impossible by construction;
   firing = bug). When `actual < expected`, log INFO line:
   `[BTCUSDT 2026-02] dedup: 3000 duplicate rows in source CSV (kept 52471665 / 52474665)`.

c) **Skip rule simplifies from `existing >= expected AND sha matches` to
   `sha matches`**. After dedup, `existing == actual_post_dedup <
   expected_raw` for affected months — the count check would force
   every-dup-month to re-ingest forever, breaking idempotency. The
   SHA256 match is the cryptographic proof that the same archive was
   processed before; atomicity guarantees no partial COPY ever lands in
   `events.ingest_log`, so a logged SHA implies a complete prior ingest.
   Republish detection unchanged: SHA mismatch → force re-ingest.

### 3. Performance estimate

Per-month time goes from `COPY only` to `COPY into staging + INSERT…ON
CONFLICT into target`. For 50M-row months: ~250s COPY + ~100–200s
INSERT pass = **~1.5–2× original time**. For the 3 remaining months,
total cost is ~10 min above the unpatched baseline.

### 4. Sanity-report addition

`sanity_checks` adds a `source_dupes` query; `print_sanity_report` adds
a "source-dupe diagnostic" section listing all months where
`actual_rows < expected_rows`, the gap per month, and total dupes
across all months. Surfaces the long-tail of Binance publishing
artifacts in one place.

### 5. Why no formal smoke test before resume

User decision: live ingest of 2026-02 = implicit smoke test. The
atomicity contract has demonstrably worked twice in production (DR
v3.0.2 + DR v3.0.3 failures both rolled back cleanly). Cost of a
hidden bug = another atomic rollback at month 51 (~$0 to recover).
Cost of a 5-min smoke test = 5 min. Marginal call; lean toward resume.

If 2026-03 or 2026-04 fail with a third failure mode, fail-fast and
report — do not handle a third class of edge case inline.

**Approver**: User (`silverspoon0099`) — approved 2026-05-06; all three
contract changes accepted.

**References**: DR v3.0.2 §3 (atomicity contract), DR v3.0.3 (prior
loader fix), failure log `logs/ingest_full.log` (2026-02 traceback +
duplicate analysis: 2,000 affected agg_ids / 3,000 extra rows).

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
