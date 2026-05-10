# PROJECT_LOG — ml-bot-events (v3.0)

> Append-only decision log. Newest entry at top.
> Every code change must reference a Decision or DR here.

---

## 2026-05-10 — Decision v3.0.14 — Path 3a: ETH walk-forward (Phase A architecture transfer) (DR)

**Context**: After §16.4 ladder steps (1) TB sweep and (2) Tier 1
features both failed to lift the BTC L0 architecture past the §16.1
Sharpe ≥ 1.0 gate (DR v3.0.10/11/12/13), the user invokes Path 3a:
transfer the Phase A pipeline to ETHUSDT and test under user's research
on ETH 2026 regime conditions.

**User research (2026-05-09)**: ETH 2026 is qualitatively different
from Lessmann's 2022Q2–2023Q2 test period:
- ETH has lost alt-leader status to SOL (RWA tokenization, active
  addresses)
- Institutional flows favor BTC ETF (~$30B) vs ETH's "modest and
  inconsistent" inflows
- Price action is grinding underperformance (-27% YTD) not capitulative
  whipsaw
- ETH/BTC ratio at 0.0313 (well below 0.040 trend-reversal level)
- Lessmann's published 1.42 ETH Sharpe came from LUNA + FTX collapse
  era; modern ETH is calmer institutional flow

The architectural transfer is still the right test (best published
prior of any path to clearing 1.0 gate), but result interpretation
must account for regime evolution.

**Decision**: Transfer Phase A pipeline to ETHUSDT:

### 1. Symbol-parameterized multi-asset architecture

Existing modules refactored to accept `symbol` parameter (default `BTC`
for backward compatibility):
- `data/db.py`: schema bootstrap creates `events.ticks_{sym}` and
  `events.bars_{sym}_cusum` per symbol
- `data/ingest_ticks.py`: `--symbol` flag (BTCUSDT or ETHUSDT)
- `bars/cusum.py`: `--symbol` flag
- `features/builder.py`: `--symbol` flag, writes `features_{sym}.parquet`
- `labels/triple_barrier.py`: `--symbol` flag, writes `labels_{sym}.parquet`
- `scripts/run_phase_1_lgbm.py`: `--asset` flag (BTC|ETH) reads
  symbol-specific parquets

Existing v3.0.11-phase1-baseline tag preserved; BTC artifacts untouched.

### 2. Phase A pipeline for ETH

Same parameters as BTC (per spec §10.1 freeze for BTC; ETH inherits
the same defaults at this stage; per-asset parameter sweeps are Phase B):
- CUSUM threshold: 0.02
- TB tp/sl: 0.05 / 0.05
- vertical_bars: 24
- Confidence threshold: 0.60

Run sequence:
1. ETH tick ingest (~6-8h wall): 2019-01 → 2026-04, same Binance Vision
   pipeline, all DR v3.0.2/3/4/6 fixes inherited
2. ETH CUSUM bars (~6.5h wall): same algorithm, same `apply_triple_barrier`
3. ETH features (33-feature parquet, ~5s)
4. ETH labels (~1s)
5. L0 walk-forward (default config; ~5 min)
6. Joint TB × threshold sweep on ETH (DR v3.0.12 mechanics) for direct
   comparison to BTC v3.0.12 result

### 3. Regime-segmented Sharpe diagnostic (per user research)

Split per-fold metrics into 3 eras anchored to user's market-research
era boundaries:

| Era | Description | Folds (by OOT_end) |
|---|---|---|
| Era 1 (2021–2022) | ETH high-beta era; Lessmann's strongest signal | 1–6 (OOT 2021-07 to 2022-10) |
| Era 2 (2023–2024) | Post-collapse normalization | 7–14 (OOT 2023-01 to 2024-10) |
| Era 3 (2025–2026) | Alt-leadership shift; institutional ETH | 15–20 (OOT 2025-01 to 2026-04) |

Apply §16.1 gate to aggregate AND to Era 3 separately. The era
breakdown reveals whether ETH's edge has persisted into modern regime
or whether (like BTC) it's hit a recent regime wall.

### 4. Output

- `data/storage/features/features_eth.parquet` (33 cols × ~ETH bar count)
- `data/storage/labels/labels_eth.parquet`
- `reports/phase_1/lgbm_results_eth.json`
- `reports/phase_1/joint_tb03_threshold_sweep_eth.json` (TB sweep ON ETH
  using DR v3.0.12 mechanics, since TB=0.03 was best for BTC; Phase B
  per-asset CUSUM/TB sweeps deferred per spec §16.4 step 2)
- Regime-segmented Sharpe table in sanity report

### 5. Decision tree (per user research-anchored spec)

| Outcome | Action |
|---|---|
| Aggregate Sharpe ≥ 1.0 AND Era 3 Sharpe ≥ 1.0 | Architecture transfers cleanly. GO L1 ResNet-LSTM on ETH (3 days) |
| Aggregate ≥ 1.0 BUT Era 3 ≤ 0.5 | Architecture worked historically; regime-fragile in 2025–2026 ETH. Ship 3b signal-provider on ETH historical model with explicit "may not work in current ETH regime" caveat |
| Aggregate ≤ 0.5 | Architecture doesn't transfer. Ship 3b signal-provider on BTC TB=0.03 + thr=0.62 baseline (DR v3.0.12 best operating point) |

### 6. Cost / budget

- Refactor: ~3 hours of code (parameterize 6 files, run BTC tests)
- ETH ingest: ~6-8h wall (background)
- ETH bars: ~6.5h wall (background)
- ETH features + labels: ~10s
- L0 sweep + joint sweep + regime analysis: ~10 min

Total wall clock: **~14-18 hours**. Within user's hard cap of 2 days.

### 7. Phase B (SOL/LINK) note

User flagged: SOL and LINK are structurally different. Lessmann's data
required CUSUM 5% + TB 8% for LINK vs ETH's 2%/5%. Phase B port is NOT
"rerun on different symbols" — it's "rerun per-asset parameter sweep."
~2-3 days per asset, not 0.5. The multi-asset refactor in this DR
makes that future work cleaner; DR v3.0.14 itself is ETH-only.

### 8. Result (post-run, 2026-05-10)

**ETH ingest**: 1.93B aggTrades across 88 months (2019-01 → 2026-04),
zero agg_id gaps, perfect tape. ~9h wall.

**ETH bars**: 28,917 CUSUM bars (md5 c3d19abe...), 100% cusum-triggered
closes, all invariants OK. Bar density similar to BTC overall but
concentrated in early-2019 (286 bars in 2019-01 vs BTC's 87) and
2025-10 (1200 bars — vol spike confirms recent regime shift). ~33min
wall.

**ETH features**: 28,917 rows × 33 cols (md5 f3e2ca90...). Parquet
8.29 MB.

**ETH labels**: 28,889 labels, class balance LONG 44.05% / SHORT
39.04% / NEUTRAL 16.92% (slightly long-skewed vs BTC's tighter
distribution; §8.3 informational fail — known ETH directional bias
2019–2021). Path-dependence 5.7–5.9% (clean sustained moves, comparable
to BTC).

**L0 walk-forward (thr=0.60)**: 20/20 folds evaluated. Pre-gate 5/6.
Aggregate Sharpe **−0.205 ± 1.594**. Only 5/20 folds traded (0
trades on 15 folds — calibrated probabilities rarely cross 0.60 on
ETH).

**Joint TB=0.03 × threshold sweep**: best aggregate at thr=0.55:
Sharpe **+0.111**, 869 trades, 53.1% win, annret +0.607. All other
threshold settings net-negative.

**Regime-segmented Sharpe (L0 default thr=0.60)**:

| Era | Folds | Active | Sharpe (mean ± std) | Trades | Annret |
|---|---|---|---|---|---|
| Era 1 (Lessmann 2021–22) | 6 | 1/6 | −0.512 ± 1.253 | 3 | −0.063 |
| Era 2 (recovery 2023–24) | 8 | 2/8 | −0.537 ± 1.956 | 32 | −0.051 |
| Era 3 (recent 2025–26) | 6 | 2/6 | **+0.543** ± 1.330 | 6 | +0.309 |

Era 3 is the only positive era but driven by only 2/6 active folds
and 6 total trades — statistically thin. Joint sweep at thr=0.55
gives Era 3 Sharpe +0.263 (more trades, lower per-trade quality).

Notable: Era 2 (recovery) is uniformly bad across all configurations
— consistent with user's research that 2023–2024 ETH was sideways
recovery with no clean directional signal.

**Verdict per §5 decision tree**:

Best aggregate Sharpe across configs = **+0.111** (joint thr=0.55).
Best Era 3 Sharpe at that config = +0.263. Both well below 0.5
threshold for "ship 3b ETH historical" branch.

→ **Aggregate ≤ 0.5: Ship 3b BTC TB=0.03 + thr=0.62 baseline**
(DR v3.0.12 best operating point). The Phase A architecture does
not transfer cleanly to ETH; user's research thesis that modern ETH
regime is harder than Lessmann's era is empirically validated.

**Status**: ETH Phase A architecture transfer — **NEGATIVE**. §16.4
fallback ladder exhausted at step (3) Path 3a; proceeding to step
(4) Path 3b deployment on BTC v3.0.12 baseline. ResNet-LSTM Phase 1.1
deferred indefinitely until a path to ≥0.5 aggregate Sharpe is found.

**Approver**: User (`silverspoon0099`) — approved 2026-05-10 in
strategic-checkpoint message with research-anchored regime-segmented
test design.

**References**: Spec §4.2, §16.4 ladder; DR v3.0.2/3/4/6 (loader fixes
inherited), DR v3.0.7/8 (features/labels pipeline), DR v3.0.9 (L0
walk-forward), DR v3.0.12 (joint sweep mechanics), DR v3.0.13 (Tier 1
result); Lessmann §"Extensibility to other cryptocurrencies"; user
2026-05-09 ETH 2026 market research.

---

## 2026-05-10 — Decision v3.0.13 — §16.4 step (2) Tier 1 features (DR)

**Context**: After DR v3.0.12 joint TB=0.03 × threshold sweep, two
findings:

- TB=0.03 + thr=0.62 is the most-robust operating point: 222 trades,
  +84 bps mean, 67.9% win, Sharpe(nonzero) +1.477, 8 of 18 folds active.
- Recent folds (14–20, OOT 2024-04 → 2026-04) are mostly inactive at
  thr=0.62 and entirely inactive at thr=0.65 — suggests regime shift
  between Lessmann's published era (2018–2023) and post-2024 BTC
  (ETF era).

Per spec §16.4 fallback ladder for "Phase A passes pre-gate but Sharpe
< 1.0" → step (2) "feature additions per §7.2". This DR executes step
(2) with all 4 §7.2 candidate categories.

**Decision**: Add 15 new features to a new parquet
`features_btc_tier1.parquet` (existing `features_btc.parquet`
unchanged; tag `v3.0.11-phase1-baseline` preserved). 33 → 48 columns.

### New feature inventory (per §7.2 evidence-strength order)

**(1) Event-memory — `bars_since_*` (6 features)**
30m project's strongest tabular signal per spec §7.2. Computed from
existing features parquet:
- `bars_since_rsi_ob_14` — bars since `rsi_14 > 70`
- `bars_since_rsi_os_14` — bars since `rsi_14 < 30`
- `bars_since_macd_cross` — bars since `macd_line` sign-flipped
- `bars_since_volume_spike` — bars since `volume > rolling_50_median × 3`
- `bars_since_close_gt_ema50` — bars since `close > ema_50`
- `bars_since_close_lt_ema50` — bars since `close < ema_50`

`close` re-loaded from `events.bars_btc_cusum` (not in features parquet).

**(2) HTF context — log-returns at standard horizons (3 features)**
Lessmann's primary feature category in the 30m project's top-20.
Implementation: `merge_asof` lookup of close-price-at-time-T-X for
each bar at time T:
- `htf_ret_4h` = `log(close[T] / close[bar_close_ts ≤ T − 4h])`
- `htf_ret_1d` = `log(close[T] / close[bar_close_ts ≤ T − 24h])`
- `htf_ret_5d` = `log(close[T] / close[bar_close_ts ≤ T − 5d])`

Simpler than full 4H/1D EMA pipeline (which would need new HTF bar
table); captures the same "where is price relative to recent-history"
information.

**(3) Volatility regime — ATR + percentile (2 features)**
Addresses Lessmann-documented low-vol weakness + user's recent-fold
inactivity observation. `high - low` from bars table:
- `atr_14` = mean(high − low) over rolling 14 bars
- `atr_pct_rank_100` = percentile rank of `atr_14` over rolling 100 bars

`atr_pct_rank_100` is the regime-classifier-equivalent — tells the
model "which volatility regime is this."

**(4) Pivot proximity — Fibonacci (4 features)**
User's chart-reading observation per §7.2. Daily-aggregated H/L/C from
CUSUM bars; pivot point P = (H_d + L_d + C_d) / 3:
- `pivot_distance` = `log(close / P)`
- `r1_distance` = `log(close / R1)` where `R1 = 2P − L_d`
- `s1_distance` = `log(close / S1)` where `S1 = 2P − H_d`
- `fib_618_distance` = `log(close / fib_618)` where `fib_618 = P + 0.618 × (H_d − L_d)`

Forward-deterministic from prior period H/L/C; no leakage.

### Mechanics

`features/tier1_builder.py`:
1. Load `features_btc.parquet` (33 cols)
2. Load `events.bars_btc_cusum` (bar_id, bar_close_ts, close, high, low)
3. Compute the 15 new features, merge on `bar_id`
4. Write `data/storage/features/features_btc_tier1.parquet` (48 cols)

`scripts/run_phase_1_lgbm.py` extended with `--tier1-features` flag
that switches `_load_features()` to read the tier1 parquet.

### Output

- `data/storage/features/features_btc_tier1.parquet` (48 features × 18,629 bars)
- `reports/phase_1/joint_tb03_threshold_sweep_tier1.json` — same schema as
  DR v3.0.12 output, for direct apples-to-apples comparison
- Side-by-side report: per-threshold aggregates with Δ vs
  v3.0.12 baseline; per-fold n_trades comparison; top-10 feature
  importance (do new features rank?)

### CLI

```
python -m features.tier1_builder                 # build extended parquet
python -m scripts.run_phase_1_lgbm --joint-sweep --tier1-features
```

### Why safe

- Existing features parquet untouched; v3.0.11-phase1-baseline tag pinned
- §10.1 frozen Phase A parameters unchanged (CUSUM 0.02, TB 0.05/0.05 in
  config; this DR uses TB=0.03 in-memory only via the joint-sweep
  mechanics from DR v3.0.11/v3.0.12)
- Pivot features' "prior period H/L/C" are deterministic from past
  bars only — no future leakage (validated by leakage-detection test
  pattern from DR v3.0.9 §16(a) if needed)

### Decision tree on result

| Outcome (vs v3.0.12 baseline at TB=0.03 + thr=0.62) | Action |
|---|---|
| Sharpe lift ≥ +0.3 AND ≥ 4 of 7 recent folds (14–20) activate | Strong evidence — proceed to Tier 2 (full HTF EMA pipeline) or strategic L1 conversation |
| Sharpe lift +0.1–0.3 OR recent-fold activation modest | Modest improvement; commit to signal-provider on extended-feature TB=0.03+thr=0.62 baseline; skip L1 |
| Sharpe negligible / regresses; new features don't rank in top-10 | Features aren't load-bearing for this architecture; revert to 33-feature baseline; ship 3b signal-provider on DR v3.0.12 best operating point |

Top-10 feature importance is the second key diagnostic: if the new 15
features don't rank in top-10 in any fold, the additions don't carry
signal regardless of Sharpe movement.

**Approver**: User (`silverspoon0099`) — approved 2026-05-10 in
strategic message; mechanics + 4-category scope + thr=0.62 evaluation
point specified by user.

**References**: Spec §7.2 feature candidates, §16.4 step (2);
DR v3.0.7 (features baseline), DR v3.0.11 (TB sweep), DR v3.0.12
(joint sweep result); 30m v2.0 project (`bars_since_*` validation).

---

## 2026-05-09 — Decision v3.0.12 — Joint TB=0.03 × threshold sweep (DR)

**Context**: DR v3.0.11 TB sweep result (commit `08edee0`,
tag `v3.0.11-phase1-baseline`):

- TB=0.03 emerged as substantially best operating point at default 0.60
  confidence threshold: 351 trades (3.8× default), Sharpe(nonzero)
  +0.564, mean +48 bps net, both LONG and SHORT working
- Still below 1.0 Phase A gate (§16.1)
- DR v3.0.10 threshold sweep was done on TB=0.05 default labels — the
  optimal threshold for TB=0.03's *different* label distribution and
  *different* calibrated probability distribution is unmeasured

The 0.60 threshold inherited from spec §8.4 + Lessmann is anchored to
his label distribution. Our TB=0.03 produces a different prior
(more LONG/SHORT, less NEUTRAL → calibrated probs land in different
band) — the joint optimum may be at a different threshold.

This is the **last cheap close** before days-long commitments. After
this DR lands we go to either §16.4 step (2) features (Tier 1 plan
already discussed) or one of the strategic forks (3a ETH, 3b
signal-provider).

**Decision**: TB=0.03 held constant; sweep threshold across
**{0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65}** (7 values; brackets the
DR v3.0.10 threshold sweep range). In-memory relabel once at TB=0.03
(reusing DR v3.0.11 mechanics), then training shared per fold (one
LightGBM + Platt fit per fold), backtest re-runs per threshold. Same
purge/embargo, same model, same Platt — only post-prediction trade
rule varies.

### Mechanics

1. Relabel bars with `apply_triple_barrier(bars, tp=0.03, sl=0.03, vertical=24)` — in-memory, once
2. Merge with features parquet on `bar_id`, drop UNLABELABLE
3. Generate 20 folds (18 evaluated; 2 skipped on n<100 guard)
4. Per fold: train + Platt fit (once), then for each of 7 thresholds:
   simulate_trades + metrics
5. Aggregate per-threshold across all folds

Wall time: ~3–5 min (training once per fold = ~3 min; backtest per
threshold is fast).

### Output

`reports/phase_1/joint_tb03_threshold_sweep.json`. Schema mirrors
`threshold_sweep.json` (DR v3.0.10) for direct apples-to-apples
comparison; only the underlying labels differ (TB=0.03 vs TB=0.05).

### CLI

```
python -m scripts.run_phase_1_lgbm --joint-sweep
```

### Why safe

- Default tp/sl=0.05 and threshold=0.60 in `config.yaml` unchanged
- §10.1 freeze unchanged; production change requires separate DR
- Sensitivity analysis to characterize joint TB×threshold curve, not
  pick a winner
- Same discipline as DR v3.0.10 / v3.0.11 (don't optimize on test;
  characterize)

### Decision tree on result

| Outcome at any threshold under TB=0.03 | Action |
|---|---|
| Sharpe ≥ 0.8 (surprise) | TB=0.03 + best-threshold close enough to gate that L1 ResNet-LSTM (3 days) might bridge — real conversation, possible L1 GO |
| Sharpe peaks 0.5–0.7 (most-likely outcome) | BTC ceiling confirmed at 4 independent operating points. Strategic fork: 3a (ETH) vs 3b (signal-provider) vs Tier 1 features (modest probabilistic upside). Real conversation. |
| Sharpe regresses below 0.4 anywhere | TB=0.03 finding doesn't generalize; revert to TB=0.05 / threshold=0.58 baseline; ship 3b directly |

**Approver**: User (`silverspoon0099`) — approved 2026-05-09 in
strategic-checkpoint message; mechanics + thresholds + output schema
specified by user; cheap-close-first discipline preserved.

**References**: DR v3.0.10 (threshold sweep methodology), DR v3.0.11
(TB sweep result), spec §16.4 fallback ladder, §10.1 frozen Phase A.

---

## 2026-05-08 — Decision v3.0.11 — TB sweep (§16.4 step 1) (DR)

**Context**: After commit `2c71b43` (DR v3.0.10 threshold sweep) the
state is:

- Pre-gate 6/6 first-6 folds passed; model demonstrably learns
- Threshold sweep characterized: 0.58 is the local economics optimum
  (Sharpe nonzero +0.519 ≈ Lessmann's BTC anchor 0.51); lowering
  threshold further DEGRADES per-trade economics
- Aggregate Sharpe across all folds remains < 1.0 at every threshold;
  Phase A pass gate (§16.1: Sharpe ≥ 1.0 mean, ≥ 75% folds positive)
  is **NOT** met

Per spec **§16.4 fallback ladder** for "Phase A passes pre-gate but
Sharpe < 1.0":

> *"(1) TB sweep first; (2) feature additions per §7.2; if still
> fails → ship signal-provider mode"*

This DR executes **step (1)**. Replicates the static-TB sensitivity
analysis on our specific BTC tick stream rather than blindly accepting
Lessmann's 5% as optimal for our data.

**Decision**: Re-run the full 18-fold L0 walk-forward at five TB values
**{0.03, 0.04, 0.05, 0.06, 0.07}** (symmetric tp/sl per spec §8.2;
`vertical_bars=24` unchanged; default 0.60 confidence threshold per
§10.1 unchanged). In-memory relabeling per TB value via
`labels.triple_barrier.apply_triple_barrier(bars, tp, sl, 24)` — no
disk parquet artifacts written for sweep variants.

### Mechanics

For each TB value t ∈ {0.03, 0.04, 0.05, 0.06, 0.07}:
1. `apply_triple_barrier(bars_full, tp_pct=t, sl_pct=t, vertical_bars=24)`
   → fresh labels DataFrame (in-memory only)
2. Merge with features parquet on `bar_id`; drop UNLABELABLE
3. Run the standard 18-fold L0 walk-forward (training, Platt
   calibration on val, OOT prediction, backtest at default 0.60
   confidence threshold)
4. Aggregate per-TB metrics

This is the §16.4-mandated step (1) execution. The §10.1 freeze on
`tp/sl=0.05` stays in place; the sweep CHARACTERIZES the
TB-vs-economics curve. Production change to TB requires a separate
DR.

### Output

`reports/phase_1/tb_sweep.json`. Schema mirrors
`threshold_sweep.json` (DR v3.0.10) for side-by-side comparison
ergonomics:

```json
{
  "tb_values_swept": [0.03, 0.04, 0.05, 0.06, 0.07],
  "n_folds_total": 20,
  "wall_clock_seconds": ...,
  "by_tb": {
    "0.03": {"aggregate": {n_trades_total, mean_pnl_bps_net,
                            median_pnl_bps_net, win_pct_mean,
                            long_win_pct_mean, short_win_pct_mean,
                            sharpe_mean_across_folds, sharpe_std,
                            sharpe_mean_nonzero_folds,
                            n_folds_zero_trades,
                            annual_return_mean},
             "per_fold": [{fold, n_trades, sharpe, ...}, ...]},
    "0.04": {...}, "0.05": {...}, "0.06": {...}, "0.07": {...}
  }
}
```

Sanity report: side-by-side aggregate table across all 5 TB values +
per-fold n_trades by TB.

### CLI

```
python -m scripts.run_phase_1_lgbm --tb-sweep
```

### Why safe (no test-set optimization, same discipline as v3.0.10)

- Default tp/sl=0.05 in `config.yaml` and §10.1 freeze are unchanged
- Sensitivity analysis to characterize the curve, not pick a winner
- All TB values × all evaluated folds reported. No selection
- Same purge/embargo, same training, same Platt, same threshold —
  TB is the ONLY variable

### Decision tree on the result (per user 2026-05-08 strategic message)

| Outcome | Action |
|---|---|
| Any TB value yields Sharpe ≥ 0.7 with healthy per-trade economics (mean ≥ +40 bps, win% ≥ 55%) | §16.4 step (2): proceed to feature additions on BTC (HTF context, ATR percentile, pivot proximity, bars-since-event) per spec §7.2 |
| All TB values stuck in Sharpe 0.4–0.6 range with no meaningful per-trade lift | Skip §7.2 features on BTC (low marginal hypothesis); go to user's strategic fork: **3a** ETH switch (~2 days) **OR** **3b** BTC signal-provider mode (~3 days) |
| Mixed (one TB shows partial improvement, others don't) | Real conversation again before committing more time |

**L1 ResNet-LSTM on BTC remains explicitly OFF the menu** — the
threshold-sweep evidence (DR v3.0.10) weakens its marginal hypothesis;
3-day commitment is not justified.

**Approver**: User (`silverspoon0099`) — approved 2026-05-08 in
strategic-checkpoint message; mechanics + TB values + output schema
specified by user; in-memory relabeling per agent's implementation
note (acknowledged).

**References**: Spec §16.4 fallback ladder, §16.1 Phase A pass gate,
§8.2 frozen Phase A labeling parameters, §10.1 frozen Phase A;
DR v3.0.8 (labeler), DR v3.0.9 (L0 walk-forward), DR v3.0.10
(threshold sweep — methodological precedent).

---

## 2026-05-08 — Decision v3.0.10 — Confidence threshold sweep (sensitivity analysis) (DR)

**Context**: Phase 1.0 L0 LightGBM full sweep result (commit `2003e06`):

- Pre-gate: 6/6 first-6 folds passed (ratios 0.91–0.96 < 0.99)
- Per-trade economics: **positive** — mean +45.96 bps net, median +287.75 bps,
  55.4% net winners across 92 trades; LONG side 59.0% win at +73.8 bps mean
- Aggregate OOT Sharpe: -0.091 ± 1.18 (noise-dominated)
- 13 of 18 folds produced 0 trades → 0.49% of bars traded

The Sharpe-near-zero is a daily-resample artifact: equity curve is flat
~99.5% of days → tiny daily mean / tiny daily std → noise-dominated
ratio. Lessmann's BTC Sharpe 0.51 was achieved at ~20–25% time in market
(many trades smoothing the curve); we have 0.5%.

The economically meaningful question is whether per-trade economics
(+46 bps net, 55% win) survive at lower confidence thresholds. If yes,
more trades raise Sharpe via N-scaling (~√N). If no, the 0.60 threshold
IS the binding constraint and the model has hit its lift ceiling.

**Decision**: Re-run the same 18-fold L0 sweep at confidence thresholds
**{0.50, 0.52, 0.55, 0.58, 0.60}**. Same model, same features, same
labels, same purge/embargo, same Platt calibration. Only the
post-prediction trade-take rule changes per threshold.

This is a **sensitivity analysis, NOT a deviation from the §10.1 freeze**.
The 0.60 threshold remains the default. If a lower value is later chosen
for production, that requires a separate DR. The purpose here is to
characterize the threshold-vs-economics tradeoff with eyes open before
deciding whether to commit ~3 days to L1 ResNet-LSTM.

### Mechanics

Per fold:
1. Train LightGBM on TRAIN, fit Platt on VAL (unchanged).
2. Predict OOT raw probs → apply Platt → calibrated probs (unchanged).
3. **For each threshold t in {0.50, 0.52, 0.55, 0.58, 0.60}**: run
   `simulate_trades(preds, ..., confidence_threshold=t)` → compute
   metrics → record under that threshold.

Training is shared across thresholds (single LightGBM + Platt fit per
fold); only backtest re-runs. Total cost ≈ original sweep + (5 × O(N)
backtest passes) ≈ +1 minute over the ~3.4-min single-threshold run.

### Output

`reports/phase_1/threshold_sweep.json`. Per-threshold sub-dict:

```json
{
  "thresholds_swept": [0.50, 0.52, 0.55, 0.58, 0.60],
  "n_folds_evaluated": 18,
  "by_threshold": {
    "0.50": {
      "aggregate": {
        "n_trades_total": ..., "trades_per_fold_mean": ...,
        "mean_pnl_bps_net": ..., "median_pnl_bps_net": ...,
        "win_pct": ..., "long_win_pct": ..., "short_win_pct": ...,
        "sharpe_mean_across_folds": ..., "sharpe_std": ...,
        "annual_return_mean": ...
      },
      "per_fold": [{"fold": ..., "n_trades": ..., "sharpe": ...,
                    "mean_pnl_bps_net": ..., "win_pct": ...}, ...]
    },
    "0.52": {...}, "0.55": {...}, "0.58": {...}, "0.60": {...}
  }
}
```

Sanity report: side-by-side comparison table across all 5 thresholds.

### CLI

```
python -m scripts.run_phase_1_lgbm --threshold-sweep
```

Without the flag, existing single-threshold behavior is unchanged
(the v3.0.9 default remains 0.60 from `config.yaml`).

### Why safe (no test-set optimization)

- Default threshold (0.60 per §10.1) is unchanged. Any production change
  requires a separate DR.
- We are CHARACTERIZING the curve, not picking a value. The L1 vs
  signal-provider vs bail decision will be made on the curve's shape,
  not on cherry-picking the best threshold's Sharpe.
- All folds, all thresholds reported. No selection. The numbers are
  what they are.

### Decision tree on the sweep result

| Outcome | Action |
|---|---|
| Lower threshold preserves +40+ bps per trade with 3-5× more trades | Strong signal; **GO L1 ResNet-LSTM** (Phase 1.1) |
| Per-trade economics collapse at any threshold below 0.60 | **BAIL signal-provider mode** with default 0.60 model |
| Mixed (e.g. economics drift but trades scale enough to net positive) | Real conversation again |

**Approver**: User (`silverspoon0099`) — approved 2026-05-08 in
strategic-checkpoint message; mechanics + thresholds + output schema
specified by user.

**References**: DR v3.0.7 (features), DR v3.0.8 (labels), DR v3.0.9
(L0 walk-forward); commit `2003e06` (L0 baseline result); spec §8.4
(confidence threshold), §10.1 (frozen Phase A parameters).

---

## 2026-05-08 — Decision v3.0.9 — Phase 1.0 L0 LightGBM walk-forward contract (DR)

**Context**: Phase 1.0 implements the L0 LightGBM walk-forward pre-gate
per spec §9.1, §9.2, §10.3, §10.4 + §11.1, §11.3, §11.5, §13. **NO L1
ResNet-LSTM in this phase** — that decision is gated on the L0 result
per the user's Phase A strategy. The §10.1-frozen parameters and §9.2
LightGBM hyperparams are NOT touched.

**Decisions**:

### 1. Source — features ⨝ labels (⨝ bars for backtest) on bar_id

Read `data/storage/features/features_btc.parquet` (18,629 × 35) and
`data/storage/labels/labels_btc.parquet` (18,629 × 6). INNER JOIN on
`bar_id`; drop rows where `label == -1` (UNLABELABLE; 24 rows). Yields
18,605 labelable rows for train/val/OOT.

For backtest (entry price), also load `events.bars_btc_cusum`
(`bar_id, bar_close_ts, close`) — close is NOT a feature column in the
features parquet (DR v3.0.7 §5). 3-way merge on `bar_id`.

### 2. Walk-forward fold construction (calendar-anchored, expanding)

Per spec §9.1 + config.yaml `walk_forward`:
- `initial_train_months=24`, `val_months=3`, `oot_months=3`,
  `step_months=3`
- Train start fixed at 2019-01-01; train_end advances by 3 months per
  fold; val and OOT slide forward.

Fold N: train [2019-01-01, val_start), val [val_start, val_end),
OOT [val_end, oot_end). Stop when oot_end > data_end (2026-05-01).
Estimated ~20 folds.

### 3. Purge / embargo (bar-count, applied within calendar boundaries)

Per spec §9.1: `purge_bars = embargo_bars = 24` (= vertical_bars).
- **Purge**: drop the last 24 train bars before val starts (their
  labels' `exit_bar_id` could fall inside val).
- **Embargo**: drop the first 24 OOT bars after val ends (those bars
  could have been adjacent to val-fitted Platt scaler).

### 4. Per-fold sample-size guard

Skip a fold (logged warning, excluded from aggregate) if val OR OOT
has < 100 labelable bars after purge/embargo. Worst-case 3-month
window in our data (2025-Q3): 107 bars — should not trigger.

### 5. Class weighting — default (no balancing)

Class distribution 42.69 / 36.44 / 20.88 reflects natural BTC bull-
market prior. Re-weighting biases the model AWAY from the prior;
calibration (§6) corrects probability scale downstream.

### 6. Probability calibration — Platt (sigmoid) on val fold

Order:
1. Train LightGBM on TRAIN with early-stopping on VAL (built-in eval)
2. Predict VAL → raw probs; fit per-class one-vs-rest sigmoid
   (`sklearn.linear_model.LogisticRegression`) on (raw_prob_k, y_val==k)
3. Predict OOT → raw probs → apply per-class Platt → renormalize rows
   to sum to 1
4. Apply 0.60 confidence threshold (§8.4) for trade signal

If a class has 0 examples in val, skip Platt for that class (use raw
probs); flag in fold report.

### 7. Pre-gate H(p) — train-fold class proportions

Per spec §10.3:
```
ratio = val_logloss / H(p_train)
H(p) = -Σ p_i · ln(p_i) over class proportions in TRAIN
```

Pre-gate passes for a fold if `ratio < 0.99`. Aggregate pass if
**≥4 of first 6 folds** pass (per config.yaml `pre_gate.required_pass_folds`).

### 8. Trading signal rule + position management

Per §8.4 + §13:
- `p_long > 0.60` → LONG at close[t]
- `p_short > 0.60` → SHORT at close[t]
- Else → no trade
- **Max 1 concurrent position per asset** (§13). Signals that fire
  while a prior position is still open (its `exit_bar_id` not yet
  reached) are skipped.

### 9. PnL via label exit_price; cost 11 bps round-trip

Subtle: the **label's `exit_price` IS the trade outcome**, regardless
of model prediction. The label was computed from the same triple-
barrier rule the model is trained against:
- predicted LONG, label LONG (TP) → win:  `+(exit/entry − 1)`
- predicted LONG, label SHORT (SL) → loss: `+(exit/entry − 1)` ≈ −5%
- predicted LONG, label NEUTRAL (timeout) → small win/loss: actual
  price diff
- predicted SHORT → mirror (sign flipped)

Cost: 11 bps subtracted from each completed trade's return (spec §11.1).
Position size: $10k fixed (spec §11.3).

### 10. Sharpe — daily-resample equity curve × √252

Per spec §11.5 + standard convention:
1. Build equity curve `(timestamp, equity)` indexed by trade exit time
2. Resample to daily, forward-fill between trades
3. `Sharpe = mean(daily_log_ret) / std(daily_log_ret) × √252`

Sortino: same but std → negative-side deviation. max_dd: peak-to-trough
on equity. pct_time_in_market: Σ(holding_bars) / OOT_bar_count × 100.
n_trades, profitable_trade_pct (net PnL > 0): direct counts.

Comparable to Lessmann's BTC Sharpe 0.51 (after 20 bps; we use 11 bps
so should land ≥ his on the same model architecture).

### 11. No standardization for L0

LightGBM is gradient-boosted trees → scale-invariant. Per-fold
standardization is meaningful only for L1 ResNet-LSTM (Phase 1.1).
L0 reads raw features from `features_btc.parquet` directly.

### 12. Reproducibility — md5 fingerprints

- Per-fold OOT predictions: md5 over calibrated-prob array
- Aggregate JSON: contains md5 fingerprints + RNG seed (42, per §9.2)

### 13. Output schema

```
reports/phase_1/
├── lgbm_results.json          # aggregate + per-fold metrics
├── fold_01/
│   ├── equity_curve.csv       # ts, equity, position, signal
│   ├── trades.csv             # entry/exit/direction/exit_reason/pnl
│   └── predictions.parquet    # bar_id, p_long, p_short, p_neutral
├── fold_02/...
```

`lgbm_results.json` per fold includes:
- val_logloss, H(p_train), ratio, pre_gate_pass
- oot_sharpe, oot_sortino, oot_max_dd, oot_pct_time_in_market,
  oot_n_trades, oot_profitable_trade_pct, oot_annual_return
- **feature_importance_top10** (per user 2026-05-08 fold (b)): list
  of `{feature, gain, split}` ordered by `gain` descending; gain is
  the loss-improvement contribution (more meaningful than split count)
- oot_md5

Aggregate: mean ± std across evaluated folds; pre-gate verdict
(first-6 folds pass-count vs required).

### 14. CLI surface

```
python -m scripts.run_phase_1_lgbm                  # full sweep
python -m scripts.run_phase_1_lgbm --first-n 3      # smoke (first 3 folds)
python -m scripts.run_phase_1_lgbm --dry-run        # build folds, no train
```

### 15. Sanity report

Per-fold:
- val_logloss / H(p_train), pre_gate pass/fail
- Sample sizes (n_train, n_val, n_oot after purge/embargo)
- OOT metrics per spec §10.4
- Top-10 feature importance (gain)
- LightGBM trees used (early-stopping)

Aggregate:
- mean ± std of each metric across evaluated folds
- pre-gate verdict (k of first 6 passed; required ≥ 4)
- **Interpretation note** (per user 2026-05-08 operational fold):
  "Per-fold Sharpe is high-variance for thin OOT (~90 daily returns
  per 3-month window). Mean across folds is the meaningful aggregate;
  individual fold swings are not over-interpreted."

### 16. Test fixtures

- `cv/tests/test_walk_forward.py`: synthetic 60-month range → expected
  fold count + boundaries; purge/embargo geometry on synthetic bars
- `cv/tests/test_pre_gate.py`: hand-computed H(p); ratio at known
  val_logloss; aggregate ≥4/6 logic
- `model/tests/test_lgbm.py`: train with seed=42 twice → identical
  predictions; Platt calibrated probs sum to 1.0; **leakage-detection
  test** (per user 2026-05-08 fold (a)):
    - inject synthetic `future_ret_5 = log(close[t+5]/close[t])`
      feature
    - train L0 with leak feature → assert val_logloss < 0.5 × H(p_train)
      (pipeline lets the model use the leak; otherwise a different bug)
    - train SAME pipeline without leak → assert val_logloss / H(p) > 0.7
      (no leak means no implausibly low logloss)
    - catches look-ahead in feature computation or full-dataset fit
- `backtest/tests/test_runner.py`: synthetic trades → known equity
  curve + Sharpe; cost application = 11 bps subtracted; no-trade bars
  contribute 0; max 1 concurrent honored

### 17. Decision tree at L0 result (per user 2026-05-08)

After full sweep + sanity report lands, decide:

| L0 outcome | Action |
|---|---|
| Pre-gate ≥4/6 AND OOT Sharpe mean ≥ 0.5 | Proceed to Phase 1.1 — L1 ResNet-LSTM |
| Pre-gate ≤3/6 AND OOT Sharpe ≤ 0.2 | Bail to signal-provider mode; skip the L1 week |
| Mixed (pre-gate passes but Sharpe 0.2-0.5, or vice versa) | STOP, send numbers, real conversation before more time |

NO L1 implementation begins without explicit user GO after L0 numbers
land.

### Implementation surface (informational, not a contract)

- `cv/__init__.py`, `cv/walk_forward.py`, `cv/pre_gate.py`
- `model/__init__.py`, `model/lgbm.py`
- `backtest/__init__.py`, `backtest/runner.py`
- `scripts/__init__.py`, `scripts/run_phase_1_lgbm.py`
- Tests for each module
- requirements.txt: `lightgbm==4.5.0`, `scikit-learn==1.5.2`

**Approver**: User (`silverspoon0099`) — approved 2026-05-08; two folds:
(a) leakage-detection test, (b) per-fold top-10 feature importance.
Operational note: per-fold Sharpe high-variance, interpret aggregate.

**References**: Spec §9.1, §9.2, §10.3, §10.4, §11.1, §11.3, §11.5,
§13; config.yaml:88-141; DR v3.0.7 (features), DR v3.0.8 (labels);
Lessmann §"Detailed results", §"Experiment setup"; López de Prado
2018 Ch. 7 (purge/embargo).

---

## 2026-05-08 — Decision v3.0.8 — Phase 0.4 triple-barrier labeler contract (DR)

**Context**: Phase 0.4 implements `labels/triple_barrier.py` per spec
§8.1, reading bars from `events.bars_btc_cusum` and writing a labels
parquet. The §8.2 frozen parameters (tp_pct=0.05, sl_pct=0.05,
vertical_bars=24 for BTC) are NOT touched. The §8.4 confidence threshold
(0.60) is a Phase 1 trainer concern, not a labeler concern.

**Decisions**:

### 1. Output destination + schema

`data/storage/labels/labels_btc.parquet`. Separate from features parquet
— downstream JOIN on `bar_id` at training time (cleaner layering;
features and labels evolve independently).

| Column | dtype | nullable | meaning |
|---|---|---|---|
| `bar_id` | int64 | no | from bars_btc_cusum |
| `label` | int8 | no | {0=LONG, 1=SHORT, 2=NEUTRAL, -1=UNLABELABLE} |
| `exit_bar_id` | Int64 | yes | bar at which barrier hit; null if UNLABELABLE |
| `exit_reason` | string | yes | {'tp','sl','timeout','ambiguous'}; null if UNLABELABLE |
| `holding_bars` | Int8 | yes | exit_bar_index − t (1..24); null if UNLABELABLE |
| `exit_price` | float64 | yes | close at exit; NaN if UNLABELABLE |

`exit_reason='ambiguous'` distinguishes the both-hit-same-bar case
(label=NEUTRAL by tie-break, see §3) from a clean vertical timeout
(`exit_reason='timeout'`). Avoids a 5th label class (spec freezes 4
label values) while preserving diagnostic fidelity.

### 2. Intra-bar barrier detection (HIGH/LOW vs TP/SL)

For each labeled bar `t` with `P_t = bars[t].close`:
- `TP_price = P_t * (1 + 0.05)`
- `SL_price = P_t * (1 - 0.05)`

Walk forward `k = t+1 .. t+24` (inclusive both ends; vertical at
k=t+24 forces NEUTRAL/timeout if no barrier hits earlier):
- `tp_hit = bars[k].high >= TP_price`  (inclusive)
- `sl_hit = bars[k].low  <= SL_price`  (inclusive)

The labeled bar `t` itself is NOT checked — labels reflect what
happens AFTER the signal is observed at close of bar `t`.

### 3. Both-hit-same-bar tie-break — NEUTRAL with exit_reason='ambiguous'

If `tp_hit AND sl_hit` at some bar `k`: `label=NEUTRAL (2)`,
`exit_reason='ambiguous'`. `exit_bar_id`, `holding_bars`, `exit_price`
populated normally.

Rationale: with 5% barriers on CUSUM-2% bars, a single bar reaching
both ±5% from `P_t` is a wide-range whipsaw — most honest is "we can't
determine direction without sub-bar data". Sanity reports the
frequency; expected to be rare (<1%) with our ~5h median bar duration.

### 4. UNLABELABLE rule (last 24 bars)

Bar at index `t` is UNLABELABLE iff `t + 24 >= N`. For N=18,629:
indices 18,605..18,628 (24 bars) → label=-1, all diagnostic fields null.
Labelable count = 18,605.

### 5. Reproducibility — md5 fingerprint

Same pattern as DR v3.0.6 / v3.0.7. md5 over canonicalized labels
parquet (sorted by bar_id, all columns text-serialized). Re-runs
identical given identical bars source.

### 6. Class balance reporting + hard-fail bounds

Sanity report prints class percentages (UNLABELABLE excluded from
denominator — structural property, not a model class).

Three-tier check:
- **Clean**: each class within §8.3 expected range (35-40 LONG/SHORT,
  20-30 NEUTRAL) → green
- **Warn**: any class outside §8.3 but within 10–50% → print warning,
  proceed
- **Hard-fail**: any class > 50% OR any class < 10% →
  `AssertionError` with message including "see spec §8.3 for expected
  class balance ranges; investigate label-config (TP/SL vs CUSUM
  threshold) before proceeding to Phase 1." (per user 2026-05-08 fold:
  message references §8.3 explicitly so future debuggers land at the
  right spec section). Parquet is still written (for inspection) but
  the build exits non-zero so a CI/shell pipeline catches imbalance
  before Phase 1 reads stale labels.

In smoke mode (`--month YYYY-MM`), class-balance assertions are
relaxed to warnings (not hard-fail). The hard-fail is meaningful only
for the full sweep where statistics across the entire 7.4-yr window
are stable.

### 7. Iteration pattern — naive O(N × vertical_bars) Python loop

18,629 × 24 ≈ 447k iterations is ~1s in Python. No vectorization.

### 8. Frozen-parameter runtime check

At entry to `run_label()`, assert:
```python
assert cfg["labeling"]["tp_pct"]["BTC"] == 0.05
assert cfg["labeling"]["sl_pct"]["BTC"] == 0.05
assert cfg["labeling"]["vertical_bars"] == 24
```

Discipline guard — any drift from §10.1 frozen values requires a DR
explicitly changing them.

### 9. Reference price + comparison semantics

`P_t = bars[t].close`. Forward checks start at `t+1`. Comparisons
inclusive (`>=`, `<=`).

### 10. Source — bars table, not features parquet

Reads `events.bars_btc_cusum` directly (`bar_id, close, high, low`
ordered by `bar_close_ts, bar_id`). Features parquet has no OHLC.
Labels parquet keyed by `bar_id` for downstream JOIN with features.

### 11. CLI surface

```
python -m labels.triple_barrier                  # full build
python -m labels.triple_barrier --dry-run        # in-memory + sanity, no parquet
python -m labels.triple_barrier --month YYYY-MM  # smoke (single month)
```

### 12. Sanity report (post-build)

- Row count == bar count
- Schema check (6 columns in DR §1 order)
- Class distribution (LONG/SHORT/NEUTRAL/UNLABELABLE) — UNLABELABLE
  separate; not in % denominator
- exit_reason distribution: tp / sl / timeout / ambiguous / null
- holding_bars histogram (1..24 + null)
- §8.3 expected-range check (35-40/35-40/20-30)
- §6 hard-fail thresholds (>50% any / <10% any) — full sweep only
- **Path-dependence diagnostic** (per user 2026-05-08 fold):
  - count of LONG-labeled bars where SL was also touched in
    `[exit_bar_id+1 .. t+24]`, as % of all LONG labels
  - mirror for SHORT (TP touched after SL exit)
  - threshold guidance (informational, no enforcement):
    `<10%` clean (first-touch reflects sustained move),
    `>40%` noisy (many "wins" would have whipsawed back). If high,
    Phase 0.5 DR could revisit with López de Prado-style double-touch
    labels.
- md5 fingerprint

### 13. Test fixtures

Synthetic-bars set:
- TP-then-SL within window (TP wins) → LONG
- SL-then-TP within window (SL wins) → SHORT
- Neither hit → NEUTRAL (timeout)
- Whipsaw same bar (both hit) → NEUTRAL (ambiguous)
- Last 24 bars → UNLABELABLE
- Determinism (run twice → identical)

Plus 4 boundary tests (per user 2026-05-08 fold):
- **TP at exactly t+1** → LONG, holding_bars=1, exit_reason='tp' (catches
  off-by-one in walk-forward start)
- **TP at exactly t+24** → LONG, holding_bars=24, exit_reason='tp'
  (catches the vertical-vs-tp boundary semantic — 'tp' NOT 'timeout')
- **TP miss through t+24, no SL** → NEUTRAL, holding_bars=24,
  exit_reason='timeout' (corollary of above; both must work)
- **SL at exactly t+24, no TP** → SHORT, holding_bars=24,
  exit_reason='sl' (mirror of t+24 TP test)

Plus 30-bar golden: hand-computed labels for an engineered price path.

### Implementation surface (informational)

- `labels/triple_barrier.py`: `apply_triple_barrier(bars_df, tp_pct,
  sl_pct, vertical_bars)` returns labels DataFrame; `_path_dependence_check()`
  computes the §12 path-dep diagnostic; `run_label()` orchestrates;
  CLI.
- `labels/tests/test_triple_barrier.py`: original synthetic set + 4
  boundary tests + 30-bar golden.

**Approver**: User (`silverspoon0099`) — approved 2026-05-08; three
folds: §6 message references spec §8.3, §13 adds 4 boundary tests,
§12 adds path-dependence diagnostic.

**References**: Spec §8.1, §8.2, §8.3, §8.4, §10.1; config.yaml:73-83;
DR v3.0.5 (bars source schema), DR v3.0.6 (sub-µs ordering),
DR v3.0.7 (parquet pattern); López de Prado 2018 Ch. 3.

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
