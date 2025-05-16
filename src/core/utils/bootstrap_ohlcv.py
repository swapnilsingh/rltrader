# utils/bootstrap_ohlcv.py

import requests

def fetch_initial_ohlcv(symbol="BTCUSDT", interval="5s", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": "1m",  # or use your actual interval
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()

    ohlcv = []
    for candle in data:
        ohlcv.append({
            "timestamp": candle[0],
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5])
        })

    return ohlcv

def convert_ohlcv_to_tick_format(candle):
    """
    Convert OHLCV candle to tick-compatible format for TickProcessor.
    """
    return {
        "timestamp": candle["timestamp"],
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "volume": candle["volume"],
        "price": candle["close"],  # ensure 'price' is set for compatibility
    }
