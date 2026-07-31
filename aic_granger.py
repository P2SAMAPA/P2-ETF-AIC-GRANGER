"""
aic_granger.py  —  AIC-Granger Engine (Optimized)
===================================================

Optimizations:
- Caching of compressed strings
- Faster LZ77 simulation
- Reduced compression attempts
- Vectorized operations where possible
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
import zlib
import lzma
import gzip
import io
from functools import lru_cache
import hashlib

warnings.filterwarnings("ignore")


class LempelZivCompressor:
    """
    LZ77-style compressor for Algorithmic Information Theory.
    Optimized with caching and faster algorithms.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.window_size = config.get("window_size", 256)
        self.lookahead_size = config.get("lookahead_size", 32)
        self.method = config.get("method", "lz77")
        self.normalize = config.get("normalize", True)
        
        # Cache for compressed strings (LRU)
        self._cache = {}
        self._cache_max = 1000
        
    def _get_cache_key(self, data: str) -> str:
        """Generate cache key for a string."""
        return hashlib.md5(data.encode('utf-8')).hexdigest()
    
    def compress(self, data: str, use_cache: bool = True) -> int:
        """
        Compress data and return compressed length.
        Uses caching for repeated strings.
        """
        if not data:
            return 0
        
        # Check cache
        if use_cache:
            key = self._get_cache_key(data)
            if key in self._cache:
                return self._cache[key]
        
        data_bytes = data.encode('utf-8')
        
        # Try only the fastest compressors first (zlib, then gzip)
        compressions = []
        
        # zlib (fast)
        try:
            zlib_comp = zlib.compress(data_bytes, level=6)  # Reduced level for speed
            compressions.append(len(zlib_comp))
        except:
            pass
        
        # gzip (fast)
        try:
            gzip_buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=gzip_buffer, mode='wb', compresslevel=6) as f:
                f.write(data_bytes)
            compressions.append(len(gzip_buffer.getvalue()))
        except:
            pass
        
        # LZMA (slower, but better compression - only for longer strings)
        if len(data_bytes) > 500:
            try:
                lzma_comp = lzma.compress(data_bytes, preset=3)  # Reduced preset
                compressions.append(len(lzma_comp))
            except:
                pass
        
        # LZ77 simulation (fast approximation)
        lz77_len = self._simulate_lz77(data)
        compressions.append(lz77_len)
        
        # Take minimum compression length
        result = min(compressions) if compressions else len(data_bytes)
        
        # Cache result
        if use_cache and len(self._cache) < self._cache_max:
            self._cache[key] = result
        
        return result
    
    def _simulate_lz77(self, data: str) -> int:
        """
        Fast LZ77 simulation using a simple sliding window.
        Optimized for speed.
        """
        n = len(data)
        if n < 10:
            return n
        
        # Simple dictionary-based compression simulation
        output = []
        i = 0
        
        # Use a fixed window size
        window = min(self.window_size, 128)  # Reduced for speed
        
        while i < n:
            best_len = 0
            best_pos = -1
            
            # Only check a subset of positions for speed
            start = max(0, i - window)
            step = 1 if n < 1000 else 2  # Skip positions for large strings
            
            for j in range(start, i, step):
                k = 0
                max_k = min(self.lookahead_size, n - i, i - j)
                while k < max_k and data[j + k] == data[i + k]:
                    k += 1
                if k > best_len:
                    best_len = k
                    best_pos = i - j
                    if best_len >= self.lookahead_size:
                        break
            
            if best_len > 2:
                output.append((best_pos, best_len))
                i += best_len
            else:
                output.append(('lit', data[i]))
                i += 1
        
        return len(str(output))
    
    def normalized_compression_distance(self, x: str, y: str) -> float:
        """Compute Normalized Compression Distance (NCD)."""
        if not x or not y:
            return 1.0
        
        # Compute compressed lengths (with caching)
        c_x = self.compress(x)
        c_y = self.compress(y)
        c_xy = self.compress(x + y)
        
        if max(c_x, c_y) == 0:
            return 1.0
        
        ncd = (c_xy - min(c_x, c_y)) / max(c_x, c_y)
        return max(0.0, min(1.0, ncd))


class AICGrangerEngine:
    """
    Algorithmic Information Flow Granger Causality.
    Optimized for speed.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.compressor = LempelZivCompressor(config)
        self.lag = config.get("lag", 5)
        self.threshold = config.get("threshold", 0.1)
        self.min_samples = config.get("min_samples", 50)
        self.shuffle_permutations = min(config.get("shuffle_permutations", 100), 50)  # Reduced for speed
        
    def encode_time_series(self, data: np.ndarray, precision: int = 2) -> str:
        """Encode time series as string with reduced precision for speed."""
        if len(data) == 0:
            return ""
        
        # Reduce precision for faster compression
        rounded = np.round(data, precision)
        
        # Use compact encoding with fewer separators
        return ','.join([f"{x:.{precision}f}" for x in rounded])
    
    def compute_causality(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Test if Y Granger-causes X using algorithmic information flow."""
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
        
        # Use only the last n samples (truncate for speed)
        x = x[-n:]
        y = y[-n:]
        
        # Get past and present
        x_past = x[:-1]
        x_present = x[1:]
        y_past = y[:-1]
        
        # Encode as strings (with reduced precision for speed)
        x_past_str = self.encode_time_series(x_past, precision=2)
        x_present_str = self.encode_time_series(x_present, precision=2)
        y_past_str = self.encode_time_series(y_past, precision=2)
        x_y_combined = self.encode_time_series(np.concatenate([x_past, y_past]), precision=2)
        
        # Compute NCDs
        ncd_x_past = self.compressor.normalized_compression_distance(
            x_present_str, x_past_str
        )
        ncd_xy = self.compressor.normalized_compression_distance(
            x_present_str, x_y_combined
        )
        
        # Causality score
        causality_score = ncd_x_past - ncd_xy
        
        # Quick significance test (reduced permutations)
        p_value = self._permutation_test_fast(x_present, x_past, y_past)
        
        significant = p_value < 0.05 and causality_score > 0
        
        return {
            "causality_score": max(0, causality_score),
            "ncd_with_x": ncd_x_past,
            "ncd_with_xy": ncd_xy,
            "p_value": p_value,
            "significant": significant,
            "direction": "y_causes_x" if significant and causality_score > 0 else "none",
            "information_flow": causality_score,
        }
    
    def _permutation_test_fast(self, x_present: np.ndarray, 
                                x_past: np.ndarray, 
                                y_past: np.ndarray) -> float:
        """Fast permutation test with fewer shuffles."""
        original_score = self._compute_causality_score(x_present, x_past, y_past)
        
        # Use fewer permutations for speed
        n_perm = min(50, self.shuffle_permutations)
        shuffled_scores = np.zeros(n_perm)
        
        for i in range(n_perm):
            y_shuffled = np.random.permutation(y_past)
            shuffled_scores[i] = self._compute_causality_score(x_present, x_past, y_shuffled)
        
        p_value = np.mean(shuffled_scores > original_score)
        return float(p_value)
    
    def _compute_causality_score(self, x_present: np.ndarray, 
                                  x_past: np.ndarray, 
                                  y_past: np.ndarray) -> float:
        """Compute causality score for a specific permutation."""
        x_past_str = self.encode_time_series(x_past, precision=2)
        x_present_str = self.encode_time_series(x_present, precision=2)
        y_past_str = self.encode_time_series(y_past, precision=2)
        x_y_combined = self.encode_time_series(np.concatenate([x_past, y_past]), precision=2)
        
        ncd_x_past = self.compressor.normalized_compression_distance(
            x_present_str, x_past_str
        )
        ncd_xy = self.compressor.normalized_compression_distance(
            x_present_str, x_y_combined
        )
        
        return ncd_x_past - ncd_xy
    
    def compute_pairwise_causality(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Compute causality in both directions."""
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
    """Compute AIC-Granger causality between two time series."""
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
    
    result["z_score"] = result["causality_strength"]
    
    return result


def compute_universe_causality(
    prices_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """Compute AIC-Granger causality for all ETFs in a universe."""
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
    
    # Only compute causality for a subset of pairs to reduce time
    # For large universes, limit to top 10 by correlation
    if n_tickers > 15:
        # Compute correlation matrix to find most connected assets
        corr_matrix = np.corrcoef([returns_dict[t] for t in tickers])
        avg_corr = np.mean(np.abs(corr_matrix - np.eye(n_tickers)), axis=1)
        top_indices = np.argsort(avg_corr)[-10:]
        tickers_subset = [tickers[i] for i in top_indices]
    else:
        tickers_subset = tickers
    
    causality_scores = {}
    
    # Compute causality for each pair (skip redundant pairs for speed)
    for i, ticker_i in enumerate(tickers_subset):
        for j, ticker_j in enumerate(tickers_subset):
            if i <= j:  # Only compute each pair once (and skip self)
                continue
            
            key = f"{ticker_i}→{ticker_j}"
            result = compute_aic_granger(
                prices_df[ticker_i],
                prices_df[ticker_j],
                config,
                window
            )
            causality_scores[key] = result
            
            # Also store reverse
            reverse_key = f"{ticker_j}→{ticker_i}"
            reverse_result = compute_aic_granger(
                prices_df[ticker_j],
                prices_df[ticker_i],
                config,
                window
            )
            causality_scores[reverse_key] = reverse_result
    
    # Aggregate scores per ticker
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
            "z_score": 0
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
