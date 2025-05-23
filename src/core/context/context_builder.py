import json
import numpy as np
from datetime import datetime

class ContextBuilder:
    def __init__(self, redis_conn, symbol, metric_key=None, window=200):
        self.redis_conn = redis_conn
        self.symbol = symbol.lower()
        self.metric_key = metric_key or f"reward_metrics:{self.symbol}"
        self.window = window

    def pull_metrics(self):
        raw = self.redis_conn.lrange(self.metric_key, 0, self.window - 1)
        parsed = [json.loads(item) for item in raw if item]
        return parsed

    def summarize_metrics(self, records):
        if not records:
            return "No recent data available."

        metrics = {
            "reward": [],
            "confidence": [],
            "stability": [],
            "drawdown_component": [],
            "volatility_component": [],
        }

        for rec in records:
            metrics["reward"].append(rec.get("final_reward", 0))
            metrics["confidence"].append(rec.get("confidence_component", 0))
            metrics["stability"].append(rec.get("stability_component", 0))
            metrics["drawdown_component"].append(rec.get("drawdown_component", 0))
            metrics["volatility_component"].append(rec.get("volatility_component", 0))

        def fmt(val):
            return f"{val:.2f}"

        return (
            f"In the past {len(records)} trades:\n"
            f"- Avg reward: {fmt(np.mean(metrics['reward']))}\n"
            f"- Confidence: {fmt(np.mean(metrics['confidence']))}\n"
            f"- Stability: {fmt(np.mean(metrics['stability']))}\n"
            f"- Drawdown: {fmt(np.mean(metrics['drawdown_component']))}\n"
            f"- Volatility: {fmt(np.mean(metrics['volatility_component']))}\n"
            f"Timestamp: {datetime.utcnow().isoformat()}"
        )

    def build_context(self):
        metrics = self.pull_metrics()
        return self.summarize_metrics(metrics)
