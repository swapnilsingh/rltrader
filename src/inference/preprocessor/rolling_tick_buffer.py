import pandas as pd
from collections import deque
from datetime import datetime
import numpy as np
from core.decorators.decorators import inject_logger

@inject_logger()
class RollingTickBuffer:
    log_level = "INFO"
    def __init__(self, maxlen=300):
        self.buffer = deque(maxlen=maxlen)
        self.min_required = 60  # Minimum required for stable indicators (MACD, ADX, etc.)

    def add_tick(self, tick):
        self.buffer.append({
            "price": float(tick["p"]),
            "quantity": float(tick["q"]),
            "timestamp": int(tick["T"])
        })

    def is_ready(self):
        if len(self.buffer) < self.min_required:
            return False

        df = pd.DataFrame(list(self.buffer))
        unique_timestamps = df["timestamp"].nunique()
        unique_prices = df["price"].nunique()

        self.logger.debug(f"[TickBuffer] total={len(df)}, unique_ts={unique_timestamps}, unique_prices={unique_prices}")
        return unique_timestamps >= self.min_required and unique_prices > 5

    def to_dataframe(self):
        if not self.is_ready():
            return None

        df = pd.DataFrame(list(self.buffer))
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["timestamp"] += pd.to_timedelta(np.arange(len(df)), unit="ns")  # avoid duplicate indices
        df.set_index("timestamp", inplace=True)

        self.logger.debug(f"[TickBuffer] DataFrame ready: {len(df)} rows from buffer.")
        return df
