# P2-TDA-REPLAY

**Continual Rehearsal with Generated Topological Scenarios**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine prevents catastrophic forgetting in market prediction models by replaying **synthetically generated data of past topological regimes** instead of old real data.

### Theory

**Topological Data Analysis (TDA):**
- Extracts persistence diagrams from market data
- Identifies topological features (H0 components, H1 loops)
- Creates a "fingerprint" of market regimes

**Topo-Score-Diffusion:**
- Generates synthetic market paths conditioned on persistence diagrams
- Creates unlimited, private rehearsal data
- No look-ahead bias (no real historical data stored)

**Continual Rehearsal:**
- Model rehearses on generated scenarios
- Prevents catastrophic forgetting
- Maintains memory of past regimes

---

## Key Metrics

| Metric | What it tells you |
|--------|-------------------|
| **z-score** | Quality of generated scenarios for rehearsal |
| **Sharpe Ratio** | Risk-adjusted return of generated scenarios |
| **N Scenarios** | Number of synthetic scenarios generated |
| **Regime Type** | Which regime was used for generation |

---

## Windows

| Window | Purpose |
|--------|---------|
| 63d | Short-term regime rehearsal |
| 252d | Core signal (primary) |
| 504d | Medium-term regimes |
| 1008d | Structural regimes |
| 2016d+ | Secular regimes |

---

## Universes

| Universe | Tickers |
|----------|---------|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-TDA-REPLAY
cd P2-TDA-REPLAY
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
