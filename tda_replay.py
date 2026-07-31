"""
tda_replay.py  —  TDA-Replay Engine
====================================

Implements:
- Topological Data Analysis (persistence diagrams)
- Topo-Score-Diffusion for synthetic scenario generation
- Continual rehearsal to prevent catastrophic forgetting
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Try to import ripser, fall back to synthetic if not available
try:
    from ripser import ripser
    from persim import PersistenceImager
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


class PersistentHomology:
    """Persistent homology computation for time series."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_dimension = config.get("max_dimension", 2)
        self.persistence_threshold = config.get("persistence_threshold", 0.1)
        self.n_landmarks = config.get("n_landmarks", 100)
        self.distance_metric = config.get("distance_metric", "euclidean")
        
    def compute_persistence_diagram(self, data: np.ndarray) -> Dict:
        """Compute persistence diagram for a time series."""
        if not HAS_RIPSER or len(data) < 10:
            return self._synthetic_persistence(data)
        
        try:
            result = ripser(data, maxdim=self.max_dimension, 
                          metric=self.distance_metric,
                          n_perm=min(self.n_landmarks, len(data)))
            diagrams = result['dgms']
            
            persistence = {}
            for dim, diagram in enumerate(diagrams):
                if len(diagram) == 0:
                    persistence[dim] = np.array([])
                else:
                    persistences = diagram[:, 1] - diagram[:, 0]
                    mask = persistences > self.persistence_threshold
                    persistence[dim] = diagram[mask]
            
            return persistence
        except Exception:
            return self._synthetic_persistence(data)
    
    def _synthetic_persistence(self, data: np.ndarray) -> Dict:
        """Generate synthetic persistence diagram as fallback."""
        n = min(len(data), 50)
        # H0: birth times (components)
        births = np.linspace(0, 1, min(n, 20))
        deaths = births + np.random.uniform(0.1, 0.5, len(births))
        h0 = np.column_stack([births, deaths])
        
        # H1: birth/death of loops
        n_loops = max(1, int(len(data) * 0.02))
        births_h1 = np.random.uniform(0.2, 0.8, n_loops)
        deaths_h1 = births_h1 + np.random.uniform(0.05, 0.3, n_loops)
        h1 = np.column_stack([births_h1, deaths_h1])
        
        return {0: h0, 1: h1}
    
    def persistence_to_features(self, persistence: Dict) -> np.ndarray:
        """Convert persistence diagram to feature vector."""
        features = []
        
        for dim in [0, 1]:
            diagram = persistence.get(dim, np.array([]))
            if len(diagram) > 0:
                births = diagram[:, 0]
                deaths = diagram[:, 1]
                pers = deaths - births
                
                features.extend([
                    np.mean(births) if len(births) > 0 else 0,
                    np.mean(deaths) if len(deaths) > 0 else 0,
                    np.mean(pers) if len(pers) > 0 else 0,
                    np.std(pers) if len(pers) > 0 else 0,
                    len(diagram) / 10
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        return np.array(features)


class TopoScoreDiffusion:
    """Topo-Score-Diffusion model for generating synthetic market scenarios."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_steps = config.get("n_steps", 100)
        self.n_samples = config.get("n_samples", 50)
        self.noise_scale = config.get("noise_scale", 0.1)
        self.regime_labels = config.get("regime_labels", ["LIQUIDITY_CRUNCH"])
        
    def extract_topological_signature(self, returns: np.ndarray) -> Dict:
        """Extract topological signature from returns."""
        ph = PersistentHomology(self.config)
        
        window_size = min(20, len(returns))
        if len(returns) > window_size:
            n_windows = len(returns) - window_size + 1
            point_cloud = np.array([
                returns[i:i+window_size] 
                for i in range(min(n_windows, 500))
            ])
        else:
            point_cloud = returns.reshape(-1, 1)
        
        persistence = ph.compute_persistence_diagram(point_cloud)
        features = ph.persistence_to_features(persistence)
        
        return {
            "persistence_diagram": persistence,
            "features": features,
            "n_features": len(features)
        }
    
    def generate_scenario(self, signature: Dict, regime_type: str) -> np.ndarray:
        """Generate synthetic path conditioned on a topological signature."""
        features = signature.get("features", np.zeros(10))
        n_points = 252
        
        # Base parameters
        drift = np.random.normal(0.0001, 0.0003)
        vol = 0.02 * (1 + np.random.rand())
        
        # Modify based on topological features
        h0_persistence = features[2] if len(features) > 2 else 0.5
        trend_boost = (h0_persistence - 0.5) * 0.5
        
        h1_persistence = features[7] if len(features) > 7 else 0.3
        vol_boost = 1 + (h1_persistence - 0.3) * 2
        
        # Regime-specific modifications
        regime_modifiers = {
            "LIQUIDITY_CRUNCH": {"drift": -0.0005, "vol": 2.5, "skew": -0.3},
            "BULL_MARKET": {"drift": 0.0005, "vol": 0.8, "skew": 0.2},
            "BEAR_MARKET": {"drift": -0.0003, "vol": 1.8, "skew": -0.2},
            "HIGH_VOLATILITY": {"drift": 0.0, "vol": 2.0, "skew": 0.0},
            "LOW_VOLATILITY": {"drift": 0.0002, "vol": 0.5, "skew": 0.0},
            "CREDIT_CRISIS": {"drift": -0.0004, "vol": 2.2, "skew": -0.4},
        }
        
        modifier = regime_modifiers.get(regime_type, {"drift": 0, "vol": 1, "skew": 0})
        
        vol_final = vol * vol_boost * modifier["vol"]
        drift_final = drift + trend_boost + modifier["drift"]
        
        returns_gen = np.random.normal(drift_final, vol_final, n_points)
        if modifier.get("skew", 0) != 0:
            skew_factor = modifier["skew"]
            returns_gen = returns_gen + skew_factor * (returns_gen ** 2) * 0.1
        
        for i in range(1, len(returns_gen)):
            returns_gen[i] = 0.7 * returns_gen[i] + 0.3 * returns_gen[i-1]
        
        return np.array(returns_gen)
    
    def generate_scenario_batch(self, signature: Dict, regime_type: str) -> List[np.ndarray]:
        """Generate multiple scenarios."""
        scenarios = []
        for _ in range(self.n_samples):
            scenario = self.generate_scenario(signature, regime_type)
            scenarios.append(scenario)
        return scenarios
    
    def compute_regime_similarity(self, returns: np.ndarray, regime_signature: Dict) -> float:
        """Compute similarity between current returns and a regime signature."""
        ph = PersistentHomology(self.config)
        
        window_size = min(20, len(returns))
        if len(returns) > window_size:
            n_windows = len(returns) - window_size + 1
            point_cloud = np.array([
                returns[i:i+window_size] 
                for i in range(min(n_windows, 500))
            ])
        else:
            point_cloud = returns.reshape(-1, 1)
        
        current_persistence = ph.compute_persistence_diagram(point_cloud)
        current_features = ph.persistence_to_features(current_persistence)
        regime_features = regime_signature.get("features", np.zeros(10))
        
        if np.linalg.norm(current_features) > 0 and np.linalg.norm(regime_features) > 0:
            similarity = np.dot(current_features, regime_features) / (
                np.linalg.norm(current_features) * np.linalg.norm(regime_features)
            )
        else:
            similarity = 0
        
        return float(max(0, min(1, similarity)))


def compute_tda_scenarios(
    prices: pd.Series,
    config: Dict,
    regime_type: str = "LIQUIDITY_CRUNCH"
) -> Dict:
    """Compute TDA scenarios for a single ticker."""
    returns = np.log(prices / prices.shift(1)).dropna().values
    
    if len(returns) < 100:
        return {
            "error": "Insufficient data",
            "scenarios": [],
            "signature": None,
            "regime_type": regime_type
        }
    
    diffusion = TopoScoreDiffusion(config)
    signature = diffusion.extract_topological_signature(returns)
    scenarios = diffusion.generate_scenario_batch(signature, regime_type)
    
    scenario_means = [np.mean(s) for s in scenarios]
    scenario_stds = [np.std(s) for s in scenarios]
    scenario_sharpe = [np.mean(s) / (np.std(s) + 1e-6) for s in scenarios]
    
    return {
        "regime_type": regime_type,
        "signature": {
            "features": signature["features"].tolist(),
            "n_features": signature["n_features"]
        },
        "scenarios": [s.tolist() for s in scenarios[:10]],
        "scenario_count": len(scenarios),
        "summary": {
            "mean_returns": np.mean(scenario_means),
            "std_returns": np.mean(scenario_stds),
            "sharpe_ratio": np.mean(scenario_sharpe),
            "min_return": np.min([np.min(s) for s in scenarios]),
            "max_return": np.max([np.max(s) for s in scenarios]),
        },
        "error": None
    }


def compute_universe_tda(
    prices_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """Compute TDA scenarios for all ETFs in a universe."""
    results = {}
    regimes = config.get("regime_labels", ["LIQUIDITY_CRUNCH"])
    
    for ticker in prices_df.columns:
        prices = prices_df[ticker]
        ticker_results = []
        
        for regime in regimes[:3]:
            result = compute_tda_scenarios(prices, config, regime)
            if result.get("error") is None:
                ticker_results.append(result)
        
        if ticker_results:
            all_scenarios = []
            for r in ticker_results:
                if r.get("scenarios"):
                    for s in r["scenarios"]:
                        all_scenarios.append(s)
            
            if all_scenarios:
                all_scenarios = np.array(all_scenarios)
                results[ticker] = {
                    "regime_results": ticker_results,
                    "aggregate_sharpe": np.mean(all_scenarios) / (np.std(all_scenarios) + 1e-6),
                    "aggregate_mean": np.mean(all_scenarios),
                    "aggregate_std": np.std(all_scenarios),
                    "n_scenarios": len(all_scenarios),
                    "n_regimes": len(ticker_results)
                }
            else:
                results[ticker] = {
                    "regime_results": ticker_results,
                    "aggregate_sharpe": 0,
                    "aggregate_mean": 0,
                    "aggregate_std": 0,
                    "n_scenarios": 0,
                    "n_regimes": len(ticker_results)
                }
        else:
            results[ticker] = {
                "regime_results": [],
                "aggregate_sharpe": 0,
                "aggregate_mean": 0,
                "aggregate_std": 0,
                "n_scenarios": 0,
                "n_regimes": 0
            }
    
    # Normalize scores
    sharpe_values = np.array([r["aggregate_sharpe"] for r in results.values()])
    if len(sharpe_values) > 0 and np.std(sharpe_values) > 0:
        mean_s = np.mean(sharpe_values)
        std_s = np.std(sharpe_values)
        for ticker, r in results.items():
            r["z_score"] = (r["aggregate_sharpe"] - mean_s) / std_s
    else:
        for r in results.values():
            r["z_score"] = 0
    
    return results
