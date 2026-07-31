"""
config.py  —  Configuration for TDA-Replay Engine
==================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - TDA: Topological data analysis parameters
  - DIFFUSION: Topo-Score-Diffusion model parameters
  - WINDOWS: Time windows for regime detection
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-tda-replay-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Windows ──────────────────────────────────────────────────────────────────

WINDOWS = [63, 252, 504, 1008, 2016, 4032, 4536]
WINDOW_LABELS = {
    63: "63d  (~3 months) — Short-term",
    252: "252d (~1 year) — Core Signal",
    504: "504d (~2 years) — Medium-term",
    1008: "1008d (~4 years) — Structural",
    2016: "2016d (~8 years) — Secular",
    4032: "4032d (~16 years) — Long-term",
    4536: "4536d (~18 years) — Full History",
}
PRIMARY_WINDOW = 252


# ── TDA Parameters ──────────────────────────────────────────────────────────

TDA = {
    "max_dimension": 2,          # 0=components, 1=loops, 2=voids
    "persistence_threshold": 0.1, # Minimum persistence to keep
    "n_landmarks": 100,          # For landmark sampling (speed)
    "distance_metric": "euclidean", # or "manhattan", "chebyshev"
}


# ── Diffusion Model Parameters ──────────────────────────────────────────────

DIFFUSION = {
    "n_steps": 100,              # Diffusion steps
    "n_samples": 50,             # Samples per scenario
    "scenario_count": 5,         # Number of scenarios to generate
    "noise_scale": 0.1,          # Noise scale for diffusion
    "regime_labels": [           # Predefined regime types
        "LIQUIDITY_CRUNCH",
        "BULL_MARKET",
        "BEAR_MARKET",
        "HIGH_VOLATILITY",
        "LOW_VOLATILITY",
        "CREDIT_CRISIS",
    ]
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
