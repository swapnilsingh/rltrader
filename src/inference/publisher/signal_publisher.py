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

        # Regular keys
        self.signal_key = f"signals:{self.symbol}"
        self.ui_signal_key = f"ui:signals:{self.symbol}"
        self.ui_signal_latest_key = f"ui:latest_signal:{self.symbol}"

    def _sanitize(self, value):
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

        # Push to primary queue
        self.redis_conn.lpush(self.signal_key, json.dumps(payload))
        self.redis_conn.ltrim(self.signal_key, 0, self.max_len - 1)

        # UI-friendly keys
        self.redis_conn.lpush(self.ui_signal_key, json.dumps(payload))
        self.redis_conn.ltrim(self.ui_signal_key, 0, self.max_len - 1)
        self.redis_conn.set(self.ui_signal_latest_key, json.dumps(payload))  # snapshot for quick UI access

        self.logger.debug(f"📤 Published signal to both queues: {payload}")

    def publish_metrics(self, key, data, max_len=200):
        safe_data = self._sanitize(data)

        # Push to original key
        self.redis_conn.lpush(key, json.dumps(safe_data))
        self.redis_conn.ltrim(key, 0, max_len - 1)

        # UI-mirrored key (if metrics:xyz → ui:metrics:xyz)
        if ":" in key:
            prefix, rest = key.split(":", 1)
            ui_key = f"ui:{prefix}:{rest}"
            self.redis_conn.lpush(ui_key, json.dumps(safe_data))
            self.redis_conn.ltrim(ui_key, 0, max_len - 1)

        self.logger.debug(f"📈 Published metrics to: {key} and ui:* mirror")

    def push(self, payload: dict):
        self.publish(
            action=payload.get("action"),
            confidence=payload.get("confidence"),
            reason=payload.get("reason"),
            extra={k: v for k, v in payload.items() if k not in ["action", "confidence", "reason"]}
        )
