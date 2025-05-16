import json
from datetime import datetime
import numpy as np
import torch
from core.decorators.decorators import inject_logger

@inject_logger()
class SignalPublisher:
    def __init__(self, redis_conn, symbol="btcusdt", max_len=100):
        self.redis_conn = redis_conn
        self.symbol = symbol.lower()
        self.max_len = max_len
        self.key = f"signals:{self.symbol}"

    def _sanitize(self, value):
        """Recursively convert values to JSON-serializable types."""
        if isinstance(value, dict):
            return {k: self._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist() if value.numel() > 1 else value.item()
        return value

    def publish(self, action, confidence, reason=None, extra=None):
        payload = {
            "action": self._sanitize(action),
            "confidence": round(self._sanitize(confidence), 4),
            "reason": self._sanitize(reason),
            "timestamp": datetime.utcnow().isoformat()
        }

        if extra:
            payload.update(self._sanitize(extra))

        self.redis_conn.lpush(self.key, json.dumps(payload))
        self.redis_conn.ltrim(self.key, 0, self.max_len - 1)
        self.logger.info(f"📤 Published signal: {payload}")

    def publish_metrics(self, key, data, max_len=200):
        safe_data = self._sanitize(data)
        self.redis_conn.lpush(key, json.dumps(safe_data))
        self.redis_conn.ltrim(key, 0, max_len - 1)

    def push(self, payload: dict):
        self.publish(
            action=payload.get("action"),
            confidence=payload.get("confidence"),
            reason=payload.get("reason"),
            extra={k: v for k, v in payload.items() if k not in ["action", "confidence", "reason"]}
        )
