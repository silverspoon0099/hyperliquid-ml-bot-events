# Project Spec — Event-Bar ML Trading Bot (v3.0)

> **Status**: Draft v3.0.0 — Phase 0 (planning). All decisions herein are anchored to published evidence (primary: Grądzki/Wójcik/Lessmann 2025) or explicit Decision Log entries. **Drift = failure.** Every change is a Deviation Request (DR) recorded in `PROJECT_LOG.md`, never an independent action.
>
> **Lineage**: Successor to `hyperliquid-ml-bot-30m/` (v2.0, archived after Phase 1.14 freeze + Phase 2 multi-asset OOT FAIL per v2.0 Decision v2.69). Fresh rewrite — no inherit-and-patch. Code may be ported by explicit reference; no implicit reuse.

---

## Table of Contents

1. Executive Summary
2. Why Event Bars — Diagnosis of 30m Failure
3. Research Findings (Literature Anchor)
4. Goals, Constraints, Non-Goals
5. Architecture
6. Data Pipeline
7. Feature Engineering
8. Labeling
9. Training & Validation
10. Anti-Overfit Discipline
11. Backtesting
12. Execution Layer
13. Risk Management
14. Project Phases & Timeline
15. Directory Structure
16. Success Criteria
17. Alpha Decay & Regime Adaptation Plan
18. Decision Log
19. References

Appendix A — Phase Checklist
Appendix B — PROJECT_LOG.md Template
Appendix C — Differences vs 30m Spec (fast onboarding)

---

## 1. Executive Summary

**Goal**: Build a profitable, automated, ML-based trading bot for Hyperliquid perpetuals on BTCUSD, with a fallback to signal-provider mode if the cost-adjusted Sharpe target is not met.

**Approach** (anchored to Lessmann 2025):
- **Sampling**: CUSUM filter on log-returns (NOT 30m time bars; NOT volume/dollar bars)
- **Labels**: Triple-barrier 3-class (LONG / SHORT / NEUTRAL) with static horizontal barriers
- **Model**: ResNet-LSTM (ensemble of top-3 validation configs); LightGBM as fast baseline + meta-filter
- **Validation**: Purged-embargo walk-forward across ≥6 quarterly OOT folds
- **Costs**: 11 bps round-trip (Hyperliquid taker 3.5 + 2 bps slippage, both sides)
- **Asset**: BTC-only Phase A; SOL/LINK only after BTC passes

**Hard pass gate (Phase A)**: Cost-adjusted Sharpe ≥ 1.0 averaged across all OOT folds AND positive Sharpe in ≥ 75% of folds.

**Stop conditions**:
- Phase A fails after exhausting parameter sweep → pivot to signal-provider mode (publish entries/exits as alerts; no auto-execution)
- Phase A passes but Phase B fails on SOL+LINK → ship BTC-only bot

**Honest budget**: 5–7 working days to Phase A pass/fail decision after data + pipeline ready. **No "profitable bot tomorrow" promise.**

---

## 2. Why Event Bars — Diagnosis of 30m Failure

### 2.1 What v2.0 (30m) actually demonstrated

The v2.0 30m project ran through 14 phases of feature engineering, label tuning, walk-forward CV, and per-asset parameter optimization. After exhaustive iteration:
- Phase 1.14 froze tp/sl/min_atr_pct as PER-ASSET dicts (BTC: 2.4/2.4/0.275, SOL/LINK: 2.7/2.7/0.60)
- Phase 2.2 multi-asset OOT result: BTC pre-gate 0.9913, SOL 1.0027, LINK 1.0041 — **all above the 1.0 random-prior gate** = no robust signal

The model did learn something — top-20 feature importance was dominated by `bars_since_*` event-memory features and HTF context. **What it did not learn was a directional edge that survives transaction costs.**

### 2.2 Three diagnoses, one root cause

1. **Wrong sampling**: 30m time bars sample on the wall clock, not on market activity. Crypto trades 24/7 with volume clustering — there is no inherent reason a 13:00 close encodes more signal than 13:13. Lessmann (2025) shows volume distribution by minute is roughly flat; if traders were optimizing for 30m closes, we'd see spikes at the 0/30 mark — we don't.
2. **Wrong target**: Next-bar prediction forces the model to learn many tiny noise-dominated movements. Triple-barrier labels learn meaningful price excursions.
3. **Wrong assumption**: Tabular ML on continuous indicator panels predicts continuous-distribution noise. Crypto edge appears in *event* structure, not in *time* structure.

The 30m project addressed (3) partially via event-memory features, but not (1) or (2). Lessmann's evidence shows you need all three.

### 2.3 Lessmann's evidence we are committing to

- **Trading at fixed time intervals** with next-bar labels was negative-Sharpe in **all 210 next-bar experiments** across 5 bar types (when systematic time sampling was used).
- **CUSUM 2% + Triple Barrier 5% + ResNet-LSTM** on ETH delivered **+91.6% annual, Sharpe 1.42 after 20 bps round-trip** during 2022Q2–2023Q2 (a period where buy-and-hold ETH was −44%).
- Same setup on BTC delivered **+20.4% annual, Sharpe 0.51** — BTC is harder than ETH, but still positive. **Our gate is set at this level.**

---

## 3. Research Findings (Literature Anchor)

### 3.1 Primary anchor — Grądzki/Wójcik/Lessmann (2025)

*"Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning."* Financial Innovation 11:136. DOI 10.1186/s40854-025-00866-w. Open access.

**Tested**: 5,400 model trainings on Binance tick BTC/ETH 2018-01 → 2023-06.
**Won**: CUSUM filter (2%) + Triple Barrier (5% sym, 24-bar vertical) + ResNet-LSTM, ensemble of top-3 validation configs.
**Lost**: Volume bars, dollar bars, range bars, time bars; Transformer / FEDformer / Autoformer / TSMixer; XGBoost (highest accuracy, lowest profit); dynamic vol-adjusted barriers.

Local copy: `../TradingView/research/paper_springer.txt`. Full reference memory: `~/.claude/.../memory/reference_springer_lessmann_paper.md`.

### 3.2 Secondary — corroborating evidence

- **PMformer (Tokajuk/Chudziak 2025, ScienceDirect)**: Best MSE on ETH but **Sharpe −0.84** on ETH despite winning on accuracy. Authors explicitly: *"a pronounced disconnect between statistical accuracy and practical trading utility."* → confirms: select on Sharpe, not on logloss.
- **Meta-RL-Crypto (2025, arxiv 2509.09751)**: LLM actor/judge daily BTC/ETH/SOL. Sharpe 0.30 bull, −0.05 bear. Not deployable. Daily TF + LLM is overkill.
- **Performer + BiLSTM (arxiv 2403.03606)**: Claimed R²=0.99 on BTC price-level prediction = autocorrelation artifact, not trading edge. Methodologically flawed.
- **NIH PMC TFT (Farooq et al. 2024)**: Daily BTC, no Sharpe, no walk-forward, no costs. Research-only.
- **Medium meta-labeling (Nguyen 2024)**: Bollinger primary + LightGBM filter on volume bars; Sharpe 1.07 (but no costs). Architecture is sound (López de Prado meta-labeling). **Useful as Phase B template.**

### 3.3 What the literature collectively concludes

| Decision | Evidence |
|---|---|
| Use information-driven sampling, specifically CUSUM | Lessmann §"Detailed results"; rejected: volume/dollar/range bars |
| Use triple-barrier labeling, static thresholds | Lessmann §"Sensitivity analysis" — dynamic barriers performed worse |
| Use CNN-LSTM family, not Transformers | Lessmann §"Comparison of Transformer vs DL" — ResNet-LSTM beat Transformer/FEDformer/Autoformer/TSMixer |
| Per-asset parameters, not universal config | Lessmann §"Extensibility to other cryptocurrencies" — LINK needs CUSUM 5% + TB 8% vs BTC/ETH 2%/5% |
| Select on cost-adjusted Sharpe, not accuracy | Lessmann + PMformer + 30m project Phase 2.2 evidence |
| Account for transaction costs in CV, not just final test | All cited papers; multiple 30m project DRs |

---

## 4. Goals, Constraints, Non-Goals

### 4.1 Goals
1. Cost-adjusted Sharpe ≥ 1.0 on BTC OOT (Phase A pass gate)
2. Positive Sharpe in ≥ 75% of OOT folds (robustness)
3. Reproducible: every seed/config/parameter is in `config.yaml`; results bit-identical given same data + config
4. Honest deliverable: if (1) fails, ship signal provider; do not force live execution

### 4.2 Constraints
- **Universe Phase A**: BTC only. SOL + LINK gated on Phase A pass.
- **Venue**: Hyperliquid perpetuals (BTCUSD-PERP). Data sourced from Binance spot tick data (BTCUSDT) for training (Lessmann's source).
- **Costs**: 3.5 bps taker + 2 bps slippage = 5.5 bps each side, **11 bps round-trip**, both sides modeled in walk-forward. Funding cost optional Phase A, mandatory Phase C.
- **Compute**: VPS (existing nvme1) for tick processing + walk-forward; local for development.
- **Spec discipline**: Every code change traceable to a numbered section here OR a logged DR. No drift.

### 4.3 Non-Goals
- News data ingestion (per user: by news report time, price has moved)
- LLM agents / RL frameworks (overkill for liquid majors per Meta-RL-Crypto evidence)
- Multi-exchange data (Binance is sufficient — Lessmann §"Data" justification)
- Sub-bar tick-level predictions (Lessmann showed moderate sampling sufficient)
- Pattern recognition on chart images (per user: ML, not chart pattern matching)

### 4.4 Explicit "do not" list (anti-drift)
- Do NOT add Transformer / FEDformer / Autoformer / TSMixer in Phase A (Lessmann §"Conclusions" rejected)
- Do NOT use volume bars / dollar bars / range bars in Phase A (Lessmann §"Detailed results" rejected)
- Do NOT use dynamic LdP-volatility-adjusted barriers (Lessmann §"Dynamic triple barrier labeling" rejected)
- Do NOT optimize parameters on the test set
- Do NOT skip transaction-cost modeling at any stage

---

## 5. Architecture

### 5.1 System diagram (text)

```
                    ┌──────────────────────────────────┐
                    │  Binance aggTrades archive (BTC) │
                    │  Jan 2019 – present, ~6 yrs      │
                    └────────────────┬─────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ data/ingest_ticks.py│
                          │  → Postgres (raw)   │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  bars/cusum_bar.py  │
                          │  CUSUM filter 2%    │
                          │  → bars_BTC.parquet │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  features/builder.py│
                          │  ~33 features       │
                          │  → features.parquet │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  labels/triple_bar.py│
                          │  TP/SL/timeout      │
                          │  → labeled.parquet  │
                          └──────────┬──────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
   ┌──────────▼─────────┐ ┌──────────▼─────────┐ ┌──────────▼─────────┐
   │ baseline/lgbm.py   │ │ model/resnet_lstm.py│ │ filter/meta_lgbm.py│
   │  fast pass-gate    │ │  Phase A primary    │ │ Phase B meta-label │
   └──────────┬─────────┘ └──────────┬─────────┘ └──────────┬─────────┘
              └──────────────────────┼──────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ cv/walk_forward.py  │
                          │ purged-embargo CV   │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ backtest/runner.py  │
                          │ +costs +slippage    │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ exec/hyperliquid.py │
                          │ (Phase D only)      │
                          └─────────────────────┘
```

### 5.2 Three-layer model (Phase A → C)

| Layer | Component | Role | Pass condition |
|---|---|---|---|
| L0 | LightGBM baseline | Cheap pass-gate; reject premise fast if no signal | Sharpe > 0.3 on val fold |
| L1 | ResNet-LSTM primary | Phase A main predictor | Sharpe ≥ 1.0 cost-adjusted, OOT |
| L2 | LightGBM meta-filter | Phase B; trust/distrust L1 signals | improves L1 Sharpe by ≥ 0.3 |

L0 exists to reject premise fast: if LightGBM on event-bars + triple-barrier can't beat random on val, we save weeks by not training L1.

### 5.3 Per-asset config schema

Following the per-asset dict pattern from 30m v2.0 Decision v2.64:

```yaml
cusum_threshold:
  BTC: 0.02   # Lessmann finding
  # SOL, LINK populated after Phase B sweep
triple_barrier:
  BTC: { tp: 0.05, sl: 0.05, vertical_bars: 24 }
```

---

## 6. Data Pipeline

### 6.1 Source

**Binance Vision archive** — `https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/`. Monthly aggTrades CSV (price, qty, side, timestamp_ms).

Why aggTrades over klines: aggTrades are tick-level; klines are 1m aggregates. Lessmann uses tick (with 1m fallback for compute reasons); we'll use 1m aggTrades-based imputation if tick processing is too slow.

**Storage**: Postgres + TimescaleDB (existing v1.0 instance, shared per Decision v2.27). New tables under schema `events`.

### 6.2 Coverage

- **BTC**: Jan 2019 → most recent complete month (~6.5 yrs as of 2026-05)
- Train + val + 6+ OOT quarters (≥ 18 mo OOT total)
- Storage estimate: ~500 GB raw aggTrades for BTC alone; CUSUM-bar parquets ~50 MB

### 6.3 Schema

```sql
CREATE TABLE events.bars_btc_cusum (
    bar_id        BIGSERIAL PRIMARY KEY,
    bar_open_ts   TIMESTAMPTZ NOT NULL,
    bar_close_ts  TIMESTAMPTZ NOT NULL,
    open          DOUBLE PRECISION NOT NULL,
    high          DOUBLE PRECISION NOT NULL,
    low           DOUBLE PRECISION NOT NULL,
    close         DOUBLE PRECISION NOT NULL,
    volume        DOUBLE PRECISION NOT NULL,
    n_trades      INTEGER NOT NULL,
    cusum_pos     DOUBLE PRECISION,  -- S+ at close
    cusum_neg     DOUBLE PRECISION,  -- S- at close
    threshold_pct DOUBLE PRECISION NOT NULL,
    UNIQUE(bar_close_ts, threshold_pct)
);
SELECT create_hypertable('events.bars_btc_cusum', 'bar_close_ts');
```

### 6.4 CUSUM bar construction

Reference: Lessmann §"CUSUM filter and range bars" (Eqs. 10–12). López de Prado 2018, Ch. 2.

```python
def cusum_bars(ticks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    ticks: columns = [ts, price, volume]
    threshold: e.g. 0.02 for 2%
    Returns: OHLCV bars where each bar closes when |cumulative log-return| ≥ threshold.
    """
    s_pos = 0.0
    s_neg = 0.0
    bar_open_idx = 0
    last_price = ticks.iloc[0].price
    bars = []
    for i, row in enumerate(ticks.itertuples()):
        r = math.log(row.price / last_price) if i > 0 else 0.0
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if max(s_pos, -s_neg) >= threshold:
            slc = ticks.iloc[bar_open_idx : i + 1]
            bars.append({
                'bar_open_ts':  slc.iloc[0].ts,
                'bar_close_ts': slc.iloc[-1].ts,
                'open':         slc.iloc[0].price,
                'high':         slc.price.max(),
                'low':          slc.price.min(),
                'close':        slc.iloc[-1].price,
                'volume':       slc.volume.sum(),
                'n_trades':     len(slc),
            })
            s_pos = 0.0
            s_neg = 0.0
            bar_open_idx = i + 1
        last_price = row.price
    return pd.DataFrame(bars)
```

**Threshold for BTC Phase A**: 0.02 (2%) — Lessmann's anchor. Sweep range Phase B: {0.015, 0.020, 0.025, 0.030}.

**Expected bar density**: Lessmann reports BTC CUSUM 2% generates roughly 5–20 bars/day depending on regime (high-vol Q2 2022 → 20+/day; low-vol Q1 2023 → 3-5/day). Over 6.5 yrs we expect ~25,000–35,000 bars total.

### 6.5 Validation

- Bar count by month (sanity: expect higher density in volatile months)
- No bar may span > 7 days (vertical fail-safe; if no 2% move in 7 days, force bar close)
- All bars must have `n_trades ≥ 1`
- Reproducibility: `cusum_bars(ticks, 0.02)` must produce bit-identical bars given identical ticks (no randomness)

---

## 7. Feature Engineering

### 7.1 Replicate Lessmann's 33-feature set first

Per Lessmann §"Feature engineering", computed *after* CUSUM sampling (i.e. on the event-bar series, not the underlying tick stream):

| # | Feature | Periods | Notes |
|---|---|---|---|
| 1–10 | EMA + std of close | 5, 10, 15, 20, 50 | 2 features per period |
| 11 | MACD line | (12, 26) fast/slow | |
| 12 | MACD signal | 9 | |
| 13 | MACD histogram | derived | |
| 14–16 | RSI | 6, 10, 14 | |
| 17–18 | Stochastic %K, %D | 14 | |
| 19 | Williams %R | 14 | |
| 20–21 | Bollinger Bands upper/lower | 5 bars, 2σ | |
| 22 | Historical return | 1 bar | log-return |
| 23 | Chaikin Money Flow | 21 | |
| 24 | Money Flow Index | 14 | |
| 25–28 | sin/cos of hour, weekday | — | seasonality |
| 29–33 | Bar metadata | — | bar_duration_sec, n_trades, volume, cusum_pos, cusum_neg |

All features computed via `pandas-ta`. All features standardized (z-score) using **training-fold statistics only** (no leakage). Standardization parameters saved per fold.

### 7.2 Feature additions — deferred to remediation only

Phase A trains and tests on the **33 features above, exactly**. No additions.

If Phase A fails (per §16.4), feature additions are one remediation lever, evaluated only after CUSUM/TB parameter sweeps. The candidate list — ordered by evidence strength — is:

1. **HTF context** (4H, 1D EMAs at event-bar close) — 30m project's top-20 feature importance was dominated by HTF context (direct prior-project evidence)
2. **Volatility regime** (ATR percentile rolling 100-bar) — addresses the Lessmann-documented low-volatility regime weakness
3. **Pivot proximity** (Fibonacci pivot distance) — pivots are forward-deterministic from prior period H/L/C, no leakage; user's chart-reading observation
4. **Bars-since-event** features (`bars_since_rsi_ob`, `bars_since_volume_spike`, etc.) — 30m project's strongest tabular signal

**Not added speculatively** (no evidence): VWAP, TSI, KAMA, fractal-dim, Hurst, sentiment, on-chain, news. May be reconsidered if the four above don't lift performance, but each addition requires a logged DR.

### 7.3 Feature exclusion list

- **No future-looking features**: TP/SL/timeout outcomes leak labels
- **No raw price levels**: only relative quantities (price changes, ratios, spreads)
- **No additions in Phase A**: even features that worked at 30m are out — Phase A is strict Lessmann replication

---

## 8. Labeling

### 8.1 Triple-barrier specification

Reference: Lessmann §"Target labeling"; López de Prado 2018, Ch. 3.

Per event bar at time `t` with close price `P_t`:
- **Upper barrier**: `P_t × (1 + 0.05)` — take-profit
- **Lower barrier**: `P_t × (1 - 0.05)` — stop-loss
- **Vertical barrier**: 24 bars forward (max holding ≈ 24 CUSUM bars; in time, ≈ 1–5 days depending on vol regime)

Walk forward bar-by-bar from `t+1` until one of three exits triggers:

| First exit hit | Label |
|---|---|
| Upper barrier (TP) | `LONG = 0` |
| Lower barrier (SL) | `SHORT = 1` |
| Vertical barrier (timeout) | `NEUTRAL = 2` |

Sentinel values (not in `classes` dict, emitted by labeler):
- `UNLABELABLE = -1`: insufficient forward data (last 24 bars of dataset)

### 8.2 Phase A parameters (frozen for BTC)

```yaml
labeling:
  method: triple_barrier
  tp_pct:           0.05    # Lessmann anchor for BTC
  sl_pct:           0.05    # symmetric
  vertical_bars:    24      # Lessmann anchor
  classes: { LONG: 0, SHORT: 1, NEUTRAL: 2 }
```

These are STATIC. Per Lessmann §"Dynamic triple barrier labeling", dynamic vol-adjusted thresholds *underperformed* static ones on his data. We freeze static for Phase A.

### 8.3 Class balance check

Expected from Lessmann's tables: roughly 35–40% LONG, 35–40% SHORT, 20–30% NEUTRAL on CUSUM 2% + TB 5% for BTC. If our labeler produces materially different balance (e.g., 80% NEUTRAL), the TP/SL is too wide vs the CUSUM threshold — investigate before proceeding.

### 8.4 Confidence threshold (decision rule)

Per Lessmann §"Experiment setup": at inference,
- Predicted P(LONG) > 0.60 → take long
- Predicted P(SHORT) > 0.60 → take short (i.e., P(SHORT) > 0.60, not "P(LONG) < 0.40")
- Otherwise no trade

This 0.60 threshold is **fixed pre-experiment** per Lessmann's anti-overfitting protocol. We do not tune it.

---

## 9. Training & Validation

### 9.1 Walk-forward split (purged-embargo)

Following 30m project's §10 cascade contract + Lessmann's expanding-window approach:

| Param | Value | Notes |
|---|---|---|
| Train window | ≥ 18 months (expanding) | At least 6,000 CUSUM bars expected |
| Validation window | 3 months | Used for early stopping + ensemble selection |
| Test (OOT) window | 3 months | Untouched until final scoring |
| Step | 3 months | Roll forward |
| Purge | 24 bars | = vertical_bars (López de Prado) |
| Embargo | 24 bars | Symmetric with purge |
| Total folds | ≥ 6 | Covers ≥ 18 months OOT |

Schedule (6.5 yrs of data, BTC):
- Initial train: 2019-01 → 2020-12 (24 mo)
- Fold 1: train 2019-01 → 2020-12, val 2021-01 → 2021-03, OOT 2021-04 → 2021-06
- Fold 2: train 2019-01 → 2021-03, val 2021-04 → 2021-06, OOT 2021-07 → 2021-09
- ... (expanding window) ...
- Fold N: ending in most recent complete quarter

### 9.2 L0 baseline — LightGBM

Identical hyperparams to 30m v2.0 §9.3:
```yaml
lightgbm:
  objective:         multiclass
  num_class:         3
  metric:            multi_logloss
  num_leaves:        63
  learning_rate:     0.05
  feature_fraction:  0.8
  bagging_fraction:  0.8
  min_child_samples: 50
  lambda_l1:         0.1
  lambda_l2:         0.1
  num_boost_round:   1000
  early_stopping_rounds: 50
  seed:              42
```
Calibration: sigmoid (Platt) on val fold. Then triple-barrier-adjusted PnL backtest.

### 9.3 L1 primary — ResNet-LSTM

Per Lessmann §"CNN-LSTM" (Fig. 1):
- Input: 96 event-bars × 33 features
- 3 × 1D-conv layers (Conv1D + BatchNorm + ReLU + Dropout); residual skip connection between input and last conv output
- LSTM stack on conv output
- Dense head → softmax (3 classes)

Hyperparameter search (Hyperband — Keras Tuner, per Lessmann):
- Conv kernel sizes: {3, 5, 7}
- Conv channels: {32, 64, 128}
- LSTM hidden: {64, 128, 256}
- LSTM layers: {1, 2}
- Dropout: {0.1, 0.2, 0.3}
- Learning rate: log-uniform [1e-5, 1e-3]
- Batch size: {32, 64, 128}

Per fold, train **3 seeds** × top-3 validation configs = 9 models; final prediction is **ensemble vote** of 9.

### 9.4 Ensemble vote rule

For each test bar, each of 9 models produces P(LONG, SHORT, NEUTRAL). Average probabilities, then apply 0.60 confidence threshold from §8.4.

### 9.5 Optuna budget

Single Hyperband search per coin (BTC Phase A). Budget: 50 trials max, 8 hr wall-clock cap on VPS T4-equivalent GPU.

---

## 10. Anti-Overfit Discipline

### 10.1 Frozen parameters (set BEFORE training, never tuned on test)

- CUSUM threshold: 0.02 (Lessmann anchor)
- TB tp/sl: 0.05 / 0.05 (Lessmann anchor)
- Vertical: 24 bars (Lessmann anchor)
- Confidence threshold: 0.60 (Lessmann anchor)
- Costs: 11 bps round-trip (Hyperliquid live values)

These five values are the spec. Changing any of them requires a logged DR.

### 10.2 What gets tuned

Only ResNet-LSTM hyperparameters (§9.3 search space), and only on the validation fold within each walk-forward step. Test fold is sacred.

### 10.3 Pre-gate ("did we learn anything?") — adapted from 30m §10.3.1 DR-014

Before running full ResNet-LSTM training, gate on LightGBM L0:

```
val_logloss / random_prior_logloss
```

For 3-class with empirical class prior `p`, random prior logloss = `H(p) = -Σ p_i × ln(p_i)`. If `val_logloss / H(p) < 0.99` on ≥ 4 of the first 6 folds → proceed to L1 training.

If pre-gate fails on BTC: **stop**. Do not waste compute on L1. Proceed to fallback (signal-provider mode planning) or DR a parameter sweep.

### 10.4 Result reporting (mandatory in PROJECT_LOG.md)

Per fold:
- `val_logloss / H(p)` ratio
- `oot_sharpe_after_costs`
- `oot_sortino_after_costs`
- `oot_max_dd`
- `oot_pct_time_in_market`
- `oot_n_trades`
- `oot_profitable_trade_pct`

Aggregate (across folds): mean ± std of each metric.

### 10.5 Phase 1 freeze contract

After Phase A pass, all of (CUSUM threshold, TB params, vertical, confidence threshold, model architecture, model hyperparams) are FROZEN. Phase B may add features but must not touch any of the above.

---

## 11. Backtesting

### 11.1 Cost model

```
position_size_usd × bps_per_side / 10_000

bps_per_side = 5.5  # 3.5 (Hyperliquid taker) + 2.0 (slippage)
round_trip   = 11.0 # both sides
```

Slippage is fixed Phase A. Phase C may upgrade to size-dependent slippage from Hyperliquid orderbook snapshots.

### 11.2 Funding (Phase C only)

Hyperliquid funding settles hourly. For Phase A, ignore funding (Lessmann ignored it; Binance spot has no funding). For Phase C, integrate funding from Hyperliquid funding-rate history.

### 11.3 Position sizing

Phase A: fixed $ per trade (e.g., $10k notional). No Kelly, no vol-targeting. Just a clean signal-vs-noise test.

### 11.4 Leverage

Phase A: 1× (spot-equivalent). Phase D execution may use Hyperliquid 5× or 10× per Decision v2.16 risk constraint, but with `max_risk_per_trade_pct = 1.0%`.

### 11.5 Equity curve outputs

For each fold, save:
- `equity_curve.csv` (timestamp, equity, position, signal)
- `trades.csv` (entry_ts, exit_ts, entry_price, exit_price, exit_reason ∈ {TP, SL, TIMEOUT}, pnl_bps, pnl_usd)

---

## 12. Execution Layer

**Phase D scope only.** Out of scope for Phase A/B.

When triggered:
- Hyperliquid SDK (Python) — REST for orders, WS for fills
- One open position at a time per asset (Phase A constraint)
- Entry: market order on signal, IOC if liquidity
- Exit: TP/SL set as native Hyperliquid stop-loss/take-profit triggers; vertical timeout monitored client-side
- Reconnect: 5s initial, exponential backoff to 60s
- Heartbeat: server-side cron alerts user if bot offline > 60 min

---

## 13. Risk Management

```yaml
risk:
  max_risk_per_trade_pct:   1.0    # of equity (Decision v2.16)
  max_position_size_pct:    10.0
  max_daily_loss_pct:       5.0    # halt for the day
  max_concurrent_positions: 1      # per asset, Phase A
  daily_drawdown_kill:      8.0    # full bot pause + email
```

---

## 14. Project Phases & Timeline

| Phase | Goal | Duration | Pass criterion |
|---|---|---|---|
| 0 | Spec + repo init | 1 day | This document signed off + repo created |
| 0.1 | Tick data ingestion | 2 days | BTC 2019-01 → present in Postgres + sanity checks |
| 0.2 | CUSUM bar pipeline | 1 day | bars_btc_cusum populated; sanity checks pass |
| 0.3 | Feature builder | 1 day | features parquet built; no NaNs after warmup |
| 0.4 | Labeler | 0.5 day | labels emitted; class balance reasonable |
| 1.0 | L0 LightGBM walk-forward | 1 day | First L0 results; pre-gate decision |
| 1.1 | If L0 pre-gate passes → L1 ResNet-LSTM Hyperband | 2 days | Top-3 configs identified |
| 1.2 | L1 walk-forward across all folds | 1 day | Final L1 results |
| **A** | **Phase A pass/fail decision** | — | **Sharpe ≥ 1.0 mean, ≥ 75% folds positive** |
| 2.0 | Phase B SOL + LINK if A passes | 3 days | Sweep CUSUM/TB params per Lessmann LINK example |
| 3.0 | Meta-labeling overlay if needed | 2 days | Improve L1 by ≥ 0.3 Sharpe |
| 4.0 | Production hardening (regime detector, funding, paper trade) | 5 days | 30 days paper trading positive |
| 5.0 | Live with $1k seed, increase quarterly | continuous | — |

**Total to Phase A pass/fail**: ≈ 8 working days.
**Total to live capital**: ≈ 25 working days IF every gate passes first try.

---

## 15. Directory Structure

```
hyperliquid-ml-bot-events/
├── Project Spec EventBars.md          # this document
├── PROJECT_LOG.md                     # decision log (append-only)
├── config.yaml                        # all params, single source of truth
├── .env                               # secrets (DB, Hyperliquid keys)
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── ingest_ticks.py                # Binance Vision archive → Postgres
│   ├── db.py                          # schema, connection
│   └── storage/                       # parquets, gitignored
│
├── bars/
│   ├── cusum.py                       # §6.4 implementation
│   └── tests/test_cusum.py
│
├── features/
│   ├── builder.py                     # §7 features
│   └── tests/
│
├── labels/
│   ├── triple_barrier.py              # §8 implementation
│   └── tests/
│
├── cv/
│   ├── walk_forward.py                # §9.1 purged-embargo
│   └── pre_gate.py                    # §10.3 logloss/H(p)
│
├── model/
│   ├── lgbm.py                        # L0 baseline
│   ├── resnet_lstm.py                 # L1 primary
│   └── meta_filter.py                 # L2 (Phase B)
│
├── tuning/
│   └── hyperband.py                   # Keras Tuner search
│
├── backtest/
│   ├── runner.py
│   └── metrics.py
│
├── exec/                              # Phase D
│   └── hyperliquid.py
│
├── scripts/
│   ├── run_phase_0_ingest.py
│   ├── run_phase_0_2_bars.py
│   ├── run_phase_1_lgbm.py
│   ├── run_phase_1_resnet_lstm.py
│   └── run_phase_a_decision.py        # final pass/fail gate
│
├── reports/                           # generated, gitignored
│   ├── phase_0/
│   ├── phase_1/
│   └── phase_a_decision.md            # the actual decision
│
├── tests/
│   └── (pytest)
│
└── logs/                              # gitignored
```

Conventions:
- All scripts read `config.yaml` for parameters; no CLI flags for params
- All outputs to `reports/<phase>/`; never to source-controlled dirs
- All intermediate data to `data/storage/`; gitignored
- All logs rotated 100MB / retained 30 days

---

## 16. Success Criteria

### 16.1 Phase A (BTC only) — pass to continue, fail to fallback

| Metric | Threshold |
|---|---|
| Cost-adjusted Sharpe (mean across OOT folds) | ≥ 1.0 |
| Folds with positive Sharpe | ≥ 75% (e.g., ≥ 5 of 6) |
| Max drawdown (worst fold) | ≤ 35% |
| Profitable trade % (mean) | ≥ 52% |
| Annual return after costs (mean) | ≥ 10% |
| pre-gate val_logloss / H(p) | < 0.99 |

### 16.2 Phase B (SOL + LINK)

Same thresholds as Phase A applied to SOL and LINK independently. Pass requires ≥ 2 of 3 assets meeting all thresholds.

### 16.3 Phase D (production)

- 30 consecutive days paper trading with positive cumulative PnL after costs
- Live monitoring dashboard operational
- Daily drawdown kill-switch tested

### 16.4 Failure modes & fallback

Remediation order is fixed: sweep parameters first, then add features, then accept fallback. No skipping, no parallel experimentation.

| Failure | Remediation order |
|---|---|
| Phase A pre-gate fails | (1) CUSUM sweep ∈ {1.5, 2.0, 2.5, 3.0}% (single DR); if still fails (2) TB sweep ∈ {3, 4, 5, 6, 7}% (single DR); if still fails (3) feature additions per §7.2 (single DR); if still fails → **ship signal-provider mode** |
| Phase A passes pre-gate but Sharpe < 1.0 | (1) TB sweep first; (2) feature additions per §7.2; if still fails → **ship signal-provider mode** |
| Phase B fails on SOL+LINK | Ship BTC-only bot |
| Live drawdown > 8% in a day | Pause bot, root-cause analysis, DR before resuming |

**Anti-drift rule**: each remediation step is one DR with one change. No "try CUSUM 2.5% with TB 6% and added VWAP at the same time." That's how v1.0 failed.

---

## 17. Alpha Decay & Regime Adaptation Plan

### 17.1 Lessmann's known weakness — low-volatility regimes

Per Lessmann §"Detailed results of best-performing model": equity curve **stagnated in Q1–Q2 2023** (low volatility). The strategy is a volatility-regime model — it makes money when there's price action and loses time when there isn't.

Mitigations (Phase C):
1. **Volatility regime detector**: Rolling 30-day realized volatility. If 30-day vol < 30th percentile (history), reduce position size by 50% or skip trades.
2. **Reduce CUSUM threshold in low vol**: CUSUM 2% may produce too few bars in low vol; consider 1.5% in low-vol regimes — but parameter switching itself is risky and must be DR'd.

### 17.2 Retraining cadence

Quarterly retrain with expanding window (mirrors walk-forward). New OOT fold added each quarter; oldest training data NOT dropped (per Lessmann expanding-window approach).

### 17.3 Decay monitoring

Live weekly Sharpe (rolling 30-day). If rolling-30d Sharpe < 0 for 4 consecutive weeks → halt bot, re-train, root-cause.

### 17.4 What to watch externally

- Hyperliquid fee schedule changes (cost model assumes 3.5 bps taker)
- Binance reduces archive availability (alt: own tick recording)
- Major BTC market structure shift (halving, ETF flows, regulatory) — re-evaluate spec relevance

---

## 18. Decision Log

> Append-only in `PROJECT_LOG.md`. Section 18 here mirrors structure but stays terse.

### v3.0.0 (2026-05-05) — Spec creation
- Decision: Fresh rewrite per user agreement, after 30m v2.0 Phase 2.2 multi-asset OOT FAIL (Decision v2.69 in 30m repo)
- Anchor: Lessmann 2025 Financial Innovation paper
- Scope: BTC-only Phase A; SOL/LINK Phase B; new repo per user direction

### v3.0.1 (2026-05-05) — Architecture choice
- L0: LightGBM (fast pre-gate)
- L1: ResNet-LSTM (Lessmann's winning architecture)
- L2: LightGBM meta-filter (Phase B only, López de Prado meta-labeling)
- Reasoning: Lessmann §"Comparison of Transformer vs DL" — Transformers underperformed; ResNet-LSTM was best across all configurations

### Future DRs to be appended in PROJECT_LOG.md.

---

## 19. References

### 19.1 Primary anchor
1. Grądzki, P., Wójcik, P., Lessmann, S. (2025). "Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning." *Financial Innovation* 11:136. DOI 10.1186/s40854-025-00866-w. Open access. Local copy: `../TradingView/research/paper_springer.txt`.

### 19.2 Methodology references
2. López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. Triple-barrier (Ch. 3), purged-embargo CV (Ch. 7), meta-labeling (Ch. 3).
3. Tokajuk, A., Chudziak, J. (2025). "Partial multivariate transformer as a tool for cryptocurrencies time series prediction." arXiv 2512.04099. Local copy: `../TradingView/research/paper3.txt`. (Negative result on accuracy ≠ trading utility.)

### 19.3 Confirming evidence
4. Nguyen, L. (2024). "Meta-labeling in cryptocurrencies market." Medium. (Bollinger primary + LightGBM filter; volume bars.)
5. Borges, T.A., Neves, R.F. (2020). "Ensemble of machine learning algorithms for cryptocurrency investment with different data resampling methods." *Applied Soft Computing* 90:106187.

### 19.4 Architecture references
6. He, K., et al. (2016). "Deep residual learning for image recognition." CVPR. (ResNet)
7. Hochreiter, S., Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation* 9(8). (LSTM)

### 19.5 Internal lineage
8. v2.0 30m project — `../hyperliquid-ml-bot-30m/Project Spec 30min.md` and `PROJECT_LOG.md` (decisions v2.0 → v2.69).
9. v3.0 user feedback memory — `~/.claude/.../memory/feedback_spec_discipline.md`, `feedback_btc_correlation.md`.

---

## Appendix A — Phase Checklist

Use this to track progress; each row is a single tickable task.

### Phase 0 — Foundations
- [ ] Repo created at `c:/Users/1/Documents/Workspace/trading/hyperliquid-ml-bot-events/`
- [ ] `.gitignore` written (data/, logs/, reports/, .env, *.parquet)
- [ ] `config.yaml` skeleton with all spec values
- [ ] `requirements.txt` with pinned versions
- [ ] PROJECT_LOG.md initialized

### Phase 0.1 — Data
- [ ] `data/ingest_ticks.py` downloads Binance Vision aggTrades for BTC 2019-01 to present
- [ ] Postgres schema `events.bars_btc_cusum` created
- [ ] Sanity: total tick count, daily volume distribution, no gaps > 1 hr

### Phase 0.2 — CUSUM
- [ ] `bars/cusum.py` implements §6.4 algorithm
- [ ] Unit test: synthetic tick stream → known bars
- [ ] Run on full BTC history, generate `bars_btc_cusum_2pct.parquet`
- [ ] Sanity: bar count by month, mean bars/day, max bar duration

### Phase 0.3 — Features
- [ ] `features/builder.py` produces all 33 features per §7.1
- [ ] Unit test: each indicator matches reference values from pandas-ta on small fixture
- [ ] Run on full BTC bars, generate `features_btc.parquet`
- [ ] Sanity: no NaNs after warmup, z-score stats per column

### Phase 0.4 — Labels
- [ ] `labels/triple_barrier.py` per §8.1
- [ ] Unit test: synthetic price series with known TP/SL/timeout cases
- [ ] Run on full BTC features, attach labels
- [ ] Sanity: class balance LONG/SHORT/NEUTRAL within expected range

### Phase 1 — L0 Pre-gate
- [ ] `cv/walk_forward.py` implements purged-embargo per §9.1
- [ ] `cv/pre_gate.py` computes val_logloss / H(p) per §10.3
- [ ] `model/lgbm.py` wraps LightGBM with §9.2 hyperparams
- [ ] `scripts/run_phase_1_lgbm.py` produces `reports/phase_1/lgbm_results.json`
- [ ] **Decision point**: pre-gate pass on ≥ 4/6 folds?

### Phase 1.1 — L1 Hyperband
- [ ] `model/resnet_lstm.py` implements §9.3 architecture
- [ ] `tuning/hyperband.py` Keras Tuner search
- [ ] Top-3 configs identified per fold

### Phase 1.2 — L1 Walk-forward
- [ ] All N folds × 9 seeds-and-configs trained
- [ ] Ensemble vote per §9.4
- [ ] `reports/phase_a_decision.md` generated with all metrics from §10.4

### **Phase A pass/fail meeting** ← go/no-go for the project

---

## Appendix B — PROJECT_LOG.md Template

Use this as the initial content of `PROJECT_LOG.md`:

```markdown
# PROJECT_LOG — ml-bot-events (v3.0)

> Append-only decision log. Newest entry at top.
> Every code change must reference a Decision/DR here.

---

## 2026-05-05 — Decision v3.0.0 — Project inception

**Context**: 30m v2.0 Phase 2.2 multi-asset OOT FAIL (BTC pre-gate 0.9913, SOL 1.0027, LINK 1.0041 — all ≥ 1.0 random-prior gate). User and Claude conducted Phase B-bis literature review (3 PDFs + 4 URLs) over 2026-05-05 session. Lessmann 2025 paper provides published, cost-validated evidence for an alternative architecture.

**Decision**:
- Fresh rewrite, new repo `hyperliquid-ml-bot-events/`
- Architecture anchored to Lessmann 2025: CUSUM bars + triple barrier + ResNet-LSTM
- BTC-only Phase A; SOL/LINK Phase B
- Spec: `Project Spec EventBars.md` v3.0 created

**Approver**: User (`silverspoon0099`)

**Reference**: This file Section 3.1; memory `reference_springer_lessmann_paper.md`.

---

## 2026-05-05 — Decision v3.0.1 — Architecture finalized

**Decision**: Three-layer model
- L0: LightGBM pass-gate
- L1: ResNet-LSTM primary (NOT Transformer per Lessmann §"Conclusions")
- L2: LightGBM meta-filter (Phase B only)

CUSUM threshold = 0.02; TB tp/sl = 0.05; vertical = 24 bars; confidence = 0.60. All five values FROZEN until DR.

**Approver**: User

**Reference**: Spec §3.3, §10.1.

---

(future entries below)
```

---

## Appendix C — Differences vs 30m Spec (fast onboarding)

| 30m Spec | Event-Bar Spec | Why |
|---|---|---|
| 30m time bars | CUSUM 2% bars | Lessmann §"Detailed results" — time bars failed |
| Triple-barrier with chop filter (per asset min_atr_pct) | Triple-barrier static (5%/5%/24bars) | Lessmann §"Sensitivity" — dynamic worse |
| LightGBM only | LightGBM (L0) + ResNet-LSTM (L1) + LightGBM meta (L2) | Lessmann §"Comparison" — DL beat classical, ResNet-LSTM beat Transformer |
| 250+ features (22 categories) | 33 features (Lessmann replication) | Phase A is replication, not extension |
| BTC + SOL + LINK parallel | BTC only Phase A | User direction; Lessmann's BTC was harder than ETH |
| max_holding_bars: 12 (4–6 hr) | vertical_bars: 24 (1–5 days dep on vol) | CUSUM bars are slower than 30m — different time scale |
| atr_mult sl/tp (volatility-scaled) | static % barriers | Lessmann §"Dynamic triple barrier" — static beat dynamic |
| Hyperliquid taker 3.5 bps + 2 bps slippage | Same | inherited |
| OOT one-shot Mar 2026 | OOT walk-forward across ≥ 6 quarters | More robust |

**End of Project Spec EventBars.md v3.0.0**
