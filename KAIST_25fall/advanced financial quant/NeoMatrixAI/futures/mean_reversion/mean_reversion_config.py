
# ============================
# futures/mean_reversion/mean_reversion_config.py
# ============================

strategy_config = {
    "window": 20,                # rolling window size
    "num_std": 2.0,              # Bollinger band multiplier
    "long_allocation_pct": 0.5,  # share allocated across longs
    "short_allocation_pct": 0.5  # share allocated across shorts
}