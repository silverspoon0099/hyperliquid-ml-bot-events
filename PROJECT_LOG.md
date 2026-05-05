# PROJECT_LOG — ml-bot-events (v3.0)

> Append-only decision log. Newest entry at top.
> Every code change must reference a Decision or DR here.

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
