import numpy as np
import pandas_ta as ta
from core.decorators.decorators import inject_logger

def validate_dataframe(df, required_columns):
    if df is None or df.empty:
        return False
    if not all(col in df.columns for col in required_columns):
        return False
    if df.isna().any().any():
        return False
    return True

@inject_logger()
class TickFeatureBuilder:
    log_level = "INFO"

    def __init__(self, feature_order):
        self.feature_order = feature_order
        self.prev_timestamp = None
        self.prev_price = None

    def build(self, tick_buffer, wallet_state):
        self.logger.debug("📥 Starting feature extraction...")

        if not tick_buffer.is_ready():
            self.logger.warning("⚠️ Tick buffer not ready.")
            raise ValueError("Tick buffer is not ready.")

        df = tick_buffer.to_dataframe()
        if df is None or df.empty:
            self.logger.warning("⚠️ Tick buffer DataFrame is empty.")
            raise ValueError("Tick buffer DataFrame is empty.")

        df["close"] = df["high"] = df["low"] = df["price"]
        df["volume"] = df["quantity"]
        df.ta.adx(append=True)
        df.ta.atr(append=True)
        df.ta.bbands(length=20, std=2.0, append=True)
        df.ta.rsi(append=True)
        df.ta.macd(append=True)

        df = df.ffill().dropna()
        latest = df.iloc[-1]

        now = df.index[-1]
        price = float(latest["price"])
        tick_price_change = df["price"].diff().iloc[-1] if len(df["price"]) > 1 else 0.0
        momentum_pct = price / df["price"].iloc[0] - 1.0
        atr_pct = float(latest["ATRr_14"]) / price if price != 0 else 0.0
        adx_scaled = float(latest["ADX_14"]) / 100.0
        rsi_scaled = float(latest["RSI_14"]) / 100.0
        band_position = (price - float(latest["BBL_20_2.0"])) / (float(latest["BBU_20_2.0"]) - float(latest["BBL_20_2.0"]) + 1e-6)

        # 💼 Wallet-aware features
        inventory = wallet_state.get("inventory", 0.0)
        balance = wallet_state.get("balance", 0.0)
        entry_price = wallet_state.get("entry_price", price)
        has_position = 1.0 if inventory > 0 else 0.0
        normalized_cash = balance / (balance + inventory * price + 1e-6)
        inventory_ratio = inventory * price / (balance + inventory * price + 1e-6)
        entry_price_diff_pct = (price - entry_price) / entry_price if entry_price != 0 else 0.0
        unrealized_pnl_pct = wallet_state.get("unrealized_pnl", 0.0) / (inventory * entry_price + 1e-6)
        drawdown_pct = max(0.0, -unrealized_pnl_pct)

        # ⏱️ Tick timing
        tick_gap = 0.0
        if self.prev_timestamp:
            tick_gap = (now - self.prev_timestamp).total_seconds()
        self.prev_timestamp = now

        # 🕒 Time features
        hour_sin = np.sin(2 * np.pi * now.hour / 24)
        hour_cos = np.cos(2 * np.pi * now.hour / 24)
        day_sin = np.sin(2 * np.pi * now.weekday() / 7)
        day_cos = np.cos(2 * np.pi * now.weekday() / 7)

        # 📈 Advanced
        volatility_pct = df["price"].rolling(20).std().iloc[-1] / df["price"].rolling(20).mean().iloc[-1]
        spread_volatility = df["price"].diff().rolling(20).std().iloc[-1]
        slippage_pct = 0.0
        if self.prev_price is not None:
            slippage_pct = abs(price - self.prev_price) / self.prev_price
        self.prev_price = price

        features = {
            "adx_scaled": adx_scaled,
            "atr_pct": atr_pct,
            "band_position": band_position,
            "drawdown_pct": drawdown_pct,
            "entry_price_diff_pct": entry_price_diff_pct,
            "has_position": has_position,
            "ind_rsi": float(latest["RSI_14"]),
            "ind_macd": float(latest["MACD_12_26_9"]),
            "inventory_ratio": inventory_ratio,
            "momentum_pct": momentum_pct,
            "normalized_cash": normalized_cash,
            "rsi_scaled": rsi_scaled,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "tick_arrival_gap": tick_gap,
            "tick_price_change": tick_price_change,
            "bid_ask_spread_pct": 0.001,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week_sin": day_sin,
            "day_of_week_cos": day_cos,
            "regime_volatility_level": 0.0,  # placeholder
            "volatility_pct": volatility_pct,
            "spread_volatility": spread_volatility,
            "slippage_pct": slippage_pct,
            "orderbook_imbalance": 0.0,  # placeholder
        }

        # 🩹 Patch: Fill in missing keys from feature_order with 0.0
        missing_keys = [k for k in self.feature_order if k not in features]
        if missing_keys:
            self.logger.debug(f"🔧 Filling missing keys: {missing_keys}")
            for key in missing_keys:
                features[key] = 0.0

        vector = np.array([features[k] for k in self.feature_order], dtype=np.float32)
        self.logger.debug("✅ Feature vector constructed successfully.")
        return vector, features
