"""
config.py  —  Configuration for AIC-Granger Engine
===================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - COMPRESSION: Lempel-Ziv compression parameters
  - WINDOWS: Time windows for causality testing
  - CAUSALITY: Thresholds and parameters
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-aic-granger-results"


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


# ── Compression Parameters ──────────────────────────────────────────────────

COMPRESSION = {
    "method": "lz77",           # lz77, lzma, zlib
    "normalize": True,          # Use NCD (Normalized Compression Distance)
    "compress_ratio": 1.0,      # Scaling factor for compression
    "window_size": 256,         # LZ77 window size
    "lookahead_size": 32,       # LZ77 lookahead buffer
}


# ── Causality Parameters ────────────────────────────────────────────────────

CAUSALITY = {
    "lag": 5,                   # Lag for causality testing
    "threshold": 0.1,           # NCD threshold for significance
    "min_samples": 50,          # Minimum samples for reliable compression
    "shuffle_permutations": 100, # Permutations for significance testing
    "significance_level": 0.05, # p-value threshold
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
