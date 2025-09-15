# strategy.py
import pandas as pd
import numpy as np

def strategy(df, config_dict):
    # --- Input validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(config_dict, dict):
        raise TypeError("config_dict must be a dictionary.")

    # --- Load strategy-specific config ---
    strategy_specific_config = config_dict.get("strategy_config", {})
    lookback = strategy_specific_config.get("lookback", 20)
    long_allocation_pct = strategy_specific_config.get("long_allocation_pct", 0.7)
    short_allocation_pct = strategy_specific_config.get("short_allocation_pct", 0.3)

    if long_allocation_pct + short_allocation_pct > 1.0:
        raise ValueError("Sum of long and short allocation percentages cannot exceed 1.0")

    # --- Momentum calculation ---
    momentum_scores = {}
    for symbol in df.columns:
        series = df[symbol].dropna()
        if len(series) < lookback + 1:
            continue
        # momentum = current price / price lookback periods ago - 1
        momentum = series.iloc[-1] / series.iloc[-lookback] - 1
        momentum_scores[symbol] = momentum

    if not momentum_scores:
        return {}
    
    # --- Split into long and short candidates ---
    longs = {s: m for s, m in momentum_scores.items() if m > 0}
    shorts = {s: abs(m) for s, m in momentum_scores.items() if m < 0}

    weights = {}

    # --- Allocate longs proportionally ---
    if longs:
        long_total = sum(longs.values())
        for s, m in longs.items():
            weights[s] = (m / long_total) * long_allocation_pct

    # --- Allocate shorts proportionally ---
    if shorts:
        short_total = sum(shorts.values())
        for s, m in shorts.items():
            weights[s] = -(m / short_total) * short_allocation_pct

    # --- Final check: ensure sum(|weights|) ≤ 1.0 ---
    total_abs = sum(abs(w) for w in weights.values())
    
    if total_abs > 1.0:
        # Normalize to fit within 1.0
        for s in weights:
            weights[s] = weights[s] / total_abs

    return weights

# allowed_symbols = config_dict.get("symbols", df.columns) 
# df = df[ [s for s in df.columns if s in allowed_symbols] ]