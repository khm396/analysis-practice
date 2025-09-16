# ============================
# futures/mean_reversion/mean_reversion.py
# (Bollinger Band MR + Dynamic Weighting + Fallback)
# ============================

import pandas as pd
import numpy as np
from typing import Dict

def strategy(df: pd.DataFrame, config_dict: Dict) -> Dict[str, float]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(config_dict, dict):
        raise TypeError("config_dict must be a dictionary.")

    cfg = config_dict.get("strategy_config", {})
    window    = int(cfg.get("window", 20))
    num_std   = float(cfg.get("num_std", 2.0))
    long_pct  = float(cfg.get("long_allocation_pct", 0.5))
    short_pct = float(cfg.get("short_allocation_pct", 0.5))

    # --- Dynamic weighting params ---
    strength_power = float(cfg.get("strength_power", 1.0))
    strength_cap   = float(cfg.get("strength_cap", np.inf))

    # --- Fallback params: only used when a bucket is empty ---
    fallback_enabled   = bool(cfg.get("fallback_enabled", True))
    fallback_top_n     = int(cfg.get("fallback_top_n", 2))     # pick up to N per side
    fallback_frac_cap  = float(cfg.get("fallback_frac_cap", 0.5))  # use up to 50% of each bucket

    if window <= 1:
        raise ValueError("window must be > 1")
    if long_pct < 0 or short_pct < 0:
        raise ValueError("allocation percentages must be non-negative")
    if long_pct + short_pct > 1.0:
        total = long_pct + short_pct
        long_pct  /= total
        short_pct /= total

    allowed = config_dict.get("symbols", df.columns.tolist())
    cols = [c for c in df.columns if c in allowed]
    if not cols:
        return {}
    if len(df) < window + 5:
        return {}

    X   = df[cols].replace(0, np.nan).ffill().bfill()
    ma  = X.rolling(window=window, min_periods=window).mean()
    std = X.rolling(window=window, min_periods=window).std(ddof=0)

    last_price = X.iloc[-1]
    last_ma    = ma.iloc[-1]
    last_std   = std.iloc[-1].replace(0, np.nan)

    # z = (MA - Price)/Std
    z = (last_ma - last_price) / last_std

    longs, shorts = [], []
    long_strengths, short_strengths = {}, {}

    for s in cols:
        z_s = z.get(s, np.nan)
        if not np.isfinite(z_s):
            continue

        # entry by strict band break
        if z_s > num_std:  # long
            strength = max(abs(z_s) - num_std, 0.0)
            if np.isfinite(strength_cap):
                strength = min(strength, strength_cap)
            long_strengths[s] = strength ** strength_power
            longs.append(s)
        elif z_s < -num_std:  # short
            strength = max(abs(z_s) - num_std, 0.0)
            if np.isfinite(strength_cap):
                strength = min(strength, strength_cap)
            short_strengths[s] = strength ** strength_power
            shorts.append(s)

    weights: Dict[str, float] = {s: 0.0 for s in cols}

    # --- Dynamic allocation (strict entries) ---
    if longs and long_pct > 0:
        ssum = sum(long_strengths.values())
        if ssum > 0:
            for s in longs:
                weights[s] = long_pct * (long_strengths[s] / ssum)
        else:
            w_each = long_pct / len(longs)
            for s in longs:
                weights[s] = w_each

    if shorts and short_pct > 0:
        ssum = sum(short_strengths.values())
        if ssum > 0:
            for s in shorts:
                weights[s] = -short_pct * (short_strengths[s] / ssum)
        else:
            w_each = short_pct / len(shorts)
            for s in shorts:
                weights[s] = -w_each

    # --- Fallback when empty bucket(s) ---
    if fallback_enabled:
        # LONG fallback: if no strict longs, pick largest positive z
        if long_pct > 0 and not longs:
            z_pos = z.dropna()[z > 0.0].sort_values(ascending=False)
            if len(z_pos) > 0:
                picks = z_pos.index[:fallback_top_n]
                strengths = z_pos.loc[picks].abs().to_dict()
                ssum = sum(strengths.values())
                if ssum > 0:
                    for s in picks:
                        # only a fraction of long_pct is used in fallback
                        weights[s] += (long_pct * fallback_frac_cap) * (strengths[s] / ssum)

        # SHORT fallback: if no strict shorts, pick most negative z
        if short_pct > 0 and not shorts:
            z_neg = z.dropna()[z < 0.0].sort_values(ascending=True)
            if len(z_neg) > 0:
                picks = z_neg.index[:fallback_top_n]
                strengths = z_neg.loc[picks].abs().to_dict()
                ssum = sum(strengths.values())
                if ssum > 0:
                    for s in picks:
                        weights[s] -= (short_pct * fallback_frac_cap) * (strengths[s] / ssum)

    # Final cap ∑|w| ≤ 1.0
    gross = sum(abs(w) for w in weights.values())
    if gross > 1.0 and gross > 0:
        scale = 1.0 / gross
        for s in list(weights.keys()):
            weights[s] *= scale

    return {k: v for k, v in weights.items() if abs(v) > 1e-12}
