# ==========================
# Required System Settings
# ==========================
system_config = {
    "data_apikey": "a71eaf04-802f-40be-93c2-5bee2548f4db",
    "tz_str": "Asia/Seoul", 
    "timeframe": "1min",
    "orderType": "market",
    "base_symbol": "BTCUSDT",
    "productType": "usdt-futures",
    "posMode": "hedge_mode",
    "holdSide": "long",                   
    "marginMode": "crossed",
    "marginCoin": "usdt",
    "symbols": ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'TRXUSDT', 'DOTUSDT', 'DOGEUSDT', 'XRPUSDT', 'BCHUSDT', 'ADAUSDT'],
    "leverage": 5,
    "trading_hours": 6,
    "total_allocation": 0.4, 
}


# ==========================
# Rebalancing Trade Parameters
# ==========================
rebalancing_config = {
    "rebalancing_interval_hours": 1, # Rebalancing cycle (hours)
}
