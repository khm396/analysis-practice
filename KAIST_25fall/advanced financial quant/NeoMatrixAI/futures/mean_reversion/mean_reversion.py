# ============================
# futures/mean_reversion/mean_reversion.py
# (Simple Bollinger Band Mean Reversion)
# ============================

import pandas as pd
import numpy as np
from typing import Dict

def strategy(df: pd.DataFrame, config_dict: Dict) -> Dict[str, float]:
    """
    Bollinger Band mean reversion:
      - Compute MA(window) and rolling std for each symbol
      - If Price < MA - k*Std → long candidate
      - If Price > MA + k*Std → short candidate
      - Otherwise flat
      - Allocate equally within long/short buckets using target allocation shares
      - Enforce ∑|w| ≤ 1.0
    Returns: weights dict {symbol: weight}, longs > 0, shorts < 0
    """
    # --- Input validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(config_dict, dict):
        raise TypeError("config_dict must be a dictionary.")

    cfg = config_dict.get("strategy_config", {})

    # Parameters
    window   = int(cfg.get("window", 20))         # rolling window
    num_std  = float(cfg.get("num_std", 2.0))     # band width multiplier
    long_pct = float(cfg.get("long_allocation_pct", 0.5))
    short_pct= float(cfg.get("short_allocation_pct", 0.5))

    if window <= 1:
        raise ValueError("window must be > 1")
    if long_pct < 0 or short_pct < 0:
        raise ValueError("allocation percentages must be non-negative")
    if long_pct + short_pct > 1.0:
        total = long_pct + short_pct
        long_pct  /= total
        short_pct /= total

    # Allowed symbols
    allowed = config_dict.get("symbols", df.columns.tolist())
    cols = [c for c in df.columns if c in allowed]
    if not cols:
        return {}
    if len(df) < window + 5:
        return {}

    X = df[cols].replace(0, np.nan).ffill().bfill()
    ma  = X.rolling(window=window, min_periods=window).mean()
    std = X.rolling(window=window, min_periods=window).std(ddof=0)

    last_price = X.iloc[-1]
    last_ma    = ma.iloc[-1]
    last_std   = std.iloc[-1]

    longs, shorts = [], []
    for s in cols:
        p = last_price[s]
        m = last_ma[s]
        sd= last_std[s]
        if not np.isfinite(p) or not np.isfinite(m) or not np.isfinite(sd):
            continue
        if p < m - num_std*sd:
            longs.append(s)
        elif p > m + num_std*sd:
            shorts.append(s)

    weights: Dict[str, float] = {s: 0.0 for s in cols}

    # Equal allocation
    if longs and long_pct > 0:
        w_each = long_pct / len(longs)
        for s in longs:
            weights[s] = w_each

    if shorts and short_pct > 0:
        w_each = short_pct / len(shorts)
        for s in shorts:
            weights[s] = -w_each

    # Final cap ∑|w| ≤ 1.0
    gross = sum(abs(w) for w in weights.values())
    if gross > 1.0 and gross > 0:
        scale = 1.0 / gross
        for s in list(weights.keys()):
            weights[s] *= scale

    return {k: v for k, v in weights.items() if abs(v) > 1e-12}