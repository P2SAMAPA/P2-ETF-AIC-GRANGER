"""
aic_granger.py  —  AIC-Granger Engine
======================================

Implements:
- Lempel-Ziv (LZ77) compression as universal feature extractor
- Normalized Compression Distance (NCD) for causality testing
- Algorithmic Information Flow between time series
- Non-parametric Granger causality using Kolmogorov complexity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
import zlib
import lzma
import gzip
import io
from collections import deque
import hashlib

warnings.filterwarnings("ignore")


class LempelZivCompressor:
    """
    LZ77-style compressor for Algorithmic Information Theory.
    
    Estimates Kolmogorov complexity via compression ratio.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.window_size = config.get("window_size", 256)
        self.lookahead_size = config.get("lookahead_size", 32)
        self.method = config.get("method", "lz77")
        self.normalize = config.get("normalize", True)
        
    def compress(self, data: str) -> int:
        """
        Compress data and return compressed length.
        
        Uses multiple compression methods and takes the minimum
        as an approximation of Kolmogorov complexity.
        """
        if not data:
            return 0
        
        data_bytes = data.encode('utf-8')
        
        # Try multiple compressors and take minimum (best compression)
        compressions = []
        
        # LZMA
        try:
            lzma_comp = lzma.compress(data_bytes)
            compressions.append(len(lzma_comp))
        except:
            pass
        
        # zlib
        try:
            zlib_comp = zlib.compress(data_bytes, level=9)
            compressions.append(len(zlib_comp))
        except:
            pass
        
        # gzip
        try:
            gzip_buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=gzip_buffer, mode='wb', compresslevel=9) as f:
                f.write(data_bytes)
            compressions.append(len(gzip_buffer.getvalue()))
        except:
            pass
        
        # LZ77 simulation (simplified)
        lz77_len = self._simulate_lz77(data)
        compressions.append(lz77_len)
        
        # Take minimum compression length as K-complexity approximation
        return min(compressions) if compressions else len(data_bytes)
    
    def _simulate_lz77(self, data: str) -> int:
        """
        Simulate LZ77 compression (simplified for speed).
        """
        n = len(data)
        if n < 10:
            return n
        
        # Simple dictionary-based compression simulation
        dictionary = {}
        output = []
        i = 0
        
        while i < n:
            # Find longest match in dictionary
            best_len = 0
            best_pos = -1
            
            # Check dictionary for matches
            for j in range(max(0, i - self.window_size), i):
                k = 0
                while (j + k < i and i + k < n and 
                       data[j + k] == data[i + k] and k < self.lookahead_size):
                    k += 1
                if k > best_len:
                    best_len = k
                    best_pos = i - j
            
            if best_len > 2:
                # Output as (distance, length)
                output.append((best_pos, best_len))
                i += best_len
            else:
                # Output as literal
                output.append(('lit', data[i]))
                i += 1
        
        return len(str(output))
    
    def normalized_compression_distance(self, x: str, y: str) -> float:
        """
        Compute Normalized Compression Distance (NCD) between two strings.
        
        NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
        
        Returns:
            float: Distance between 0 (identical) and 1 (unrelated)
        """
        if not x or not y:
            return 1.0
        
        # Compute compressed lengths
        c_x = self.compress(x)
        c_y = self.compress(y)
        c_xy = self.compress(x + y)
        c_yx = self.compress(y + x)
        
        # Use symmetric NCD
        c_xy_min = min(c_xy, c_yx)
        
        if max(c_x, c_y) == 0:
            return 1.0
        
        ncd = (c_xy_min - min(c_x, c_y)) / max(c_x, c_y)
        
        return max(0.0, min(1.0, ncd))
    
    def algorithmic_mutual_information(self, x: str, y: str) -> float:
        """
        Estimate algorithmic mutual information.
        
        AMI(x,y) = C(x) + C(y) - C(xy)
        Higher values = more information shared
        """
        if not x or not y:
            return 0.0
        
        c_x = self.compress(x)
        c_y = self.compress(y)
        c_xy = self.compress(x + y)
        
        return max(0.0, c_x + c_y - c_xy)


class AICGrangerEngine:
    """
    Algorithmic Information Flow Granger Causality.
    
    Uses Kolmogorov complexity to test non-parametric causality.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.compressor = LempelZivCompressor(config.get("compression", {}))
        self.lag = config.get("lag", 5)
        self.threshold = config.get("threshold", 0.1)
        self.min_samples = config.get("min_samples", 50)
        self.shuffle_permutations = config.get("shuffle_permutations", 100)
        self.significance_level = config.get("significance_level", 0.05)
        
    def encode_time_series(self, data: np.ndarray, precision: int = 3) -> str:
        """
        Encode time series as string for compression.
        
        Uses finite-precision encoding to maintain causality structure.
        """
        if len(data) == 0:
            return ""
        
        # Round to reduce noise
        rounded = np.round(data, precision)
        
        # Encode as string with separators
        return ','.join([f"{x:.{precision}f}" for x in rounded])
    
    def compute_causality(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Test if Y Granger-causes X using algorithmic information flow.
        
        Causality is measured as:
        If the presence of Y's past makes X's present harder to compress
        (conditional on X's past), then Y is causally informative for X.
        """
        n = min(len(x), len(y))
        if n < self.min_samples + self.lag:
            return {
                "causality_score": 0,
                "ncd_value": 1.0,
                "p_value": 1.0,
                "significant": False,
                "direction": "none",
                "error": "Insufficient data"
            }
        
        # Use only the last n samples
        x = x[-n:]
        y = y[-n:]
        
        # Get past and present
        x_past = x[:-1]
        x_present = x[1:]
        y_past = y[:-1]
        
        # Encode as strings
        x_past_str = self.encode_time_series(x_past)
        x_present_str = self.encode_time_series(x_present)
        y_past_str = self.encode_time_series(y_past)
        x_y_combined = self.encode_time_series(np.concatenate([x_past, y_past]))
        
        # Compute NCDs
        # NCD(x_present | x_past) - compression with only X's past
        ncd_x_past = self.compressor.normalized_compression_distance(
            x_present_str, x_past_str
        )
        
        # NCD(x_present | x_past + y_past) - compression with X and Y past
        ncd_xy = self.compressor.normalized_compression_distance(
            x_present_str, x_y_combined
        )
        
        # Causality score: reduction in NCD when including Y's past
        # Lower NCD = better compression = more information shared
        causality_score = ncd_x_past - ncd_xy
        
        # If Y's past helps compress X's present, causality_score > 0
        # This means Y is causally informative for X
        
        # Perform permutation test for significance
        p_value = self._permutation_test(x_present, x_past, y_past)
        
        significant = p_value < self.significance_level and causality_score > 0
        
        return {
            "causality_score": max(0, causality_score),
            "ncd_with_x": ncd_x_past,
            "ncd_with_xy": ncd_xy,
            "p_value": p_value,
            "significant": significant,
            "direction": "y_causes_x" if significant and causality_score > 0 else "none",
            "information_flow": causality_score,
        }
    
    def _permutation_test(self, x_present: np.ndarray, 
                          x_past: np.ndarray, 
                          y_past: np.ndarray) -> float:
        """
        Perform permutation test for significance.
        
        Shuffle y_past and recompute causality score.
        p-value = proportion of shuffled scores > original score.
        """
        original_score = self._compute_causality_score(x_present, x_past, y_past)
        
        shuffled_scores = []
        for _ in range(self.shuffle_permutations):
            y_shuffled = np.random.permutation(y_past)
            score = self._compute_causality_score(x_present, x_past, y_shuffled)
            shuffled_scores.append(score)
        
        shuffled_scores = np.array(shuffled_scores)
        p_value = np.mean(shuffled_scores > original_score)
        
        return float(p_value)
    
    def _compute_causality_score(self, x_present: np.ndarray, 
                                  x_past: np.ndarray, 
                                  y_past: np.ndarray) -> float:
        """
        Compute causality score for a specific permutation.
        """
        x_past_str = self.encode_time_series(x_past)
        x_present_str = self.encode_time_series(x_present)
        y_past_str = self.encode_time_series(y_past)
        x_y_combined = self.encode_time_series(np.concatenate([x_past, y_past]))
        
        ncd_x_past = self.compressor.normalized_compression_distance(
            x_present_str, x_past_str
        )
        ncd_xy = self.compressor.normalized_compression_distance(
            x_present_str, x_y_combined
        )
        
        return ncd_x_past - ncd_xy
    
    def compute_pairwise_causality(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Compute causality in both directions.
        
        Returns:
            x_to_y: Does X Granger-cause Y?
            y_to_x: Does Y Granger-cause X?
        """
        y_to_x = self.compute_causality(x, y)
        x_to_y = self.compute_causality(y, x)
        
        return {
            "y_to_x": y_to_x,
            "x_to_y": x_to_y,
            "causality_strength": max(
                y_to_x.get("causality_score", 0),
                x_to_y.get("causality_score", 0)
            ),
            "direction": "y_to_x" if y_to_x.get("significant", False) else (
                        "x_to_y" if x_to_y.get("significant", False) else "none")
        }


def compute_aic_granger(
    prices_x: pd.Series,
    prices_y: pd.Series,
    config: Dict,
    window: int = 252
) -> Dict:
    """
    Compute AIC-Granger causality between two time series.
    """
    # Compute returns
    returns_x = np.log(prices_x / prices_x.shift(1)).dropna().values
    returns_y = np.log(prices_y / prices_y.shift(1)).dropna().values
    
    if len(returns_x) < window or len(returns_y) < window:
        return {
            "causality_score": 0,
            "causality_strength": 0,
            "direction": "insufficient_data",
            "y_to_x": {"significant": False, "causality_score": 0},
            "x_to_y": {"significant": False, "causality_score": 0}
        }
    
    # Use last 'window' days
    x = returns_x[-window:]
    y = returns_y[-window:]
    
    engine = AICGrangerEngine(config)
    result = engine.compute_pairwise_causality(x, y)
    
    # Compute z-score relative to random (for ranking)
    # Higher causality strength = better
    result["z_score"] = result["causality_strength"]
    
    return result


def compute_universe_causality(
    prices_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """
    Compute AIC-Granger causality for all ETFs in a universe.
    Uses each ETF as both cause and effect.
    """
    results = {}
    
    # Compute returns for all
    returns_dict = {}
    for ticker in prices_df.columns:
        returns = np.log(prices_df[ticker] / prices_df[ticker].shift(1)).dropna().values
        if len(returns) >= window:
            returns_dict[ticker] = returns[-window:]
    
    if len(returns_dict) < 2:
        return results
    
    tickers = list(returns_dict.keys())
    n_tickers = len(tickers)
    
    # Compute causality for each pair
    causality_scores = {}
    for i, ticker_i in enumerate(tickers):
        for j, ticker_j in enumerate(tickers):
            if i == j:
                continue
            
            key = f"{ticker_i}→{ticker_j}"
            result = compute_aic_granger(
                prices_df[ticker_i],
                prices_df[ticker_j],
                config,
                window
            )
            causality_scores[key] = result
    
    # Aggregate scores per ticker (average causality strength)
    for ticker in tickers:
        incoming = []
        outgoing = []
        
        for key, result in causality_scores.items():
            if key.endswith(f"→{ticker}"):
                incoming.append(result.get("causality_strength", 0))
            elif key.startswith(f"{ticker}→"):
                outgoing.append(result.get("causality_strength", 0))
        
        results[ticker] = {
            "incoming_causality": np.mean(incoming) if incoming else 0,
            "outgoing_causality": np.mean(outgoing) if outgoing else 0,
            "net_causality": np.mean(incoming) - np.mean(outgoing) if incoming and outgoing else 0,
            "n_incoming": len(incoming),
            "n_outgoing": len(outgoing),
            "z_score": 0  # Will be normalized
        }
    
    # Normalize z-scores
    net_scores = np.array([r["net_causality"] for r in results.values()])
    if len(net_scores) > 0 and np.std(net_scores) > 0:
        mean_n = np.mean(net_scores)
        std_n = np.std(net_scores)
        for ticker, r in results.items():
            r["z_score"] = (r["net_causality"] - mean_n) / std_n
    else:
        for r in results.values():
            r["z_score"] = 0
    
    return results
