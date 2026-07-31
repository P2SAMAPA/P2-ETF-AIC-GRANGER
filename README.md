# P2-AIC-GRANGER

**Algorithmic Information Flow — Non-Parametric Causality with Kolmogorov Complexity**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine implements a **true non-parametric causality test** using **Kolmogorov complexity** and **Normalized Compression Distance (NCD)**.

Instead of using linear models or Shannon entropy (which requires estimating a PDF), it uses **compression as a universal feature extractor**. A Lempel-Ziv or Gzip compressor acts as the feature extractor, detecting non-linear, high-dimensional causal links invisible to standard methods.

### Theory

**Kolmogorov Complexity:** The length of the shortest program that produces a string. It's a theoretical measure of information content.

**Normalized Compression Distance (NCD):**
NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

text

Where C(x) is the compressed length of x. NCD = 0 means identical, NCD = 1 means unrelated.

**Causality Test:**
- If the presence of Y's compressed past makes X's present **harder to compress** (conditional on X's past), then **Y is causally informative for X**
- This captures non-linear, high-dimensional causal links invisible to linear Granger causality

**Algorithmic Information Flow:**
- Higher causality score = stronger information flow from one asset to another
- Positive net causality = asset receives more information than it sends → potential leader
- Negative net causality = asset sends more information than it receives → potential follower

---

## Key Metrics

| Metric | What it tells you | Trading Implication |
|--------|-------------------|---------------------|
| **z-score** | Cross-sectional ranking of causality strength | Higher = stronger causal driver |
| **Net Causality** | Incoming - Outgoing information flow | Positive = information receiver (leader) |
| **Incoming Causality** | How much others cause this asset | High = this asset is a causal hub |
| **Outgoing Causality** | How much this asset causes others | High = this asset drives others |
| **Significance** | p-value from permutation test | p < 0.05 = statistically significant |

---

## Windows

| Window | Purpose |
|--------|---------|
| 63d | Short-term information flow |
| 252d | Core signal (primary) |
| 504d | Medium-term causal structure |
| 1008d | Structural information flow |
| 2016d+ | Secular causality |

---

## Universes

| Universe | Tickers |
|----------|---------|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

---

## Compression Methods

| Method | Description |
|--------|-------------|
| **LZ77** | Lempel-Ziv 77 compression (simulated) |
| **LZMA** | Lempel-Ziv-Markov chain algorithm |
| **zlib** | Deflate compression |
| **gzip** | GNU zip compression |

The engine uses **multiple compressors** and takes the **minimum compressed length** as the best approximation of Kolmogorov complexity.

---

## Outputs

The engine produces two JSON files:

### Tab 1 — `aic_granger_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-07-31",
  "universes": {
    "FI_COMMODITIES": {
      "top_buys": [
        {"ticker": "TLT", "z_score": 1.42},
        {"ticker": "GLD", "z_score": 0.98}
      ],
      "top_sells": [
        {"ticker": "SLV", "z_score": -1.23}
      ],
      "full_scores": {
        "TLT": {
          "z_score": 1.42,
          "best_window": 252,
          "net_causality": 0.85,
          "incoming": 0.92,
          "outgoing": 0.07,
          "action": "STRONG BUY"
        }
      }
    }
  }
}
Tab 2 — aic_granger_breakdown_YYYY-MM-DD.json
json
{
  "run_date": "2026-07-31",
  "universes": {
    "FI_COMMODITIES": {
      "windows": {
        "252": {
          "top_buys": [
            {"ticker": "TLT", "z_score": 1.42}
          ],
          "full_ranking": [
            ["TLT", 1.42, "STRONG BUY"],
            ["GLD", 0.98, "BUY"],
            ["SLV", -1.23, "STRONG SELL"]
          ]
        }
      }
    }
  }
}
Dashboard Features
Tab	What it shows
Best Window per ETF	Each ETF's highest z-score window, with net causality
Explore by Window	All ETFs ranked for a selected window
Setup
bash
git clone https://github.com/P2SAMAPA/P2-AIC-GRANGER
cd P2-AIC-GRANGER
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
GitHub Actions
Runs automatically at 00:30 UTC Monday–Saturday via .github/workflows/daily.yml.

Required secret: HF_TOKEN

Advantages Over Standard Granger Causality
Feature	Linear Granger	AIC-Granger
Linearity	Assumes linear relationships	✅ No linearity assumption
Dimensionality	Limited by data	✅ Handles high-dimensional data
Non-linearity	Misses non-linear links	✅ Captures non-linear links
Distribution	Assumes normal errors	✅ No distribution assumption
Feature extraction	Manual	✅ Automatic via compression
Interpretation	Easy	✅ Information-theoretic
References
Lempel, A., & Ziv, J. (1976). On the Complexity of Finite Sequences. IEEE Transactions on Information Theory.

Li, M., Chen, X., Li, X., Ma, B., & Vitányi, P. (2004). The Similarity Metric. IEEE Transactions on Information Theory.

Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory. Wiley.

Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-spectral Methods. Econometrica.

Ziv, J., & Lempel, A. (1978). Compression of Individual Sequences via Variable-Rate Coding. IEEE Transactions on Information Theory.
