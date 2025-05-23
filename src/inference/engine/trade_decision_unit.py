
import numpy as np

class TradeDecisionUnit:
    def __init__(self, feature_order=None):
        self.feature_order = feature_order or []

    def decode(self, model_output):
        return {
            "action": self._decode_action(model_output.get("signal_logits")),
            "quantity": self._decode_quantity(model_output.get("quantity")),
            "exec_mode": self._decode_exec_mode(model_output.get("execution_mode")),
            "raw": model_output
        }

    def _decode_action(self, logits):
        if logits is None:
            return "HOLD"
        return ["BUY", "SELL", "HOLD"][int(np.argmax(logits))]

    def _decode_quantity(self, logits):
        if not isinstance(logits, list):
            return 0.1
        options = [0.05, 0.1, 0.2, 0.5]
        idx = int(np.argmax(logits))
        return options[idx] if idx < len(options) else 0.1

    def _decode_exec_mode(self, logits):
        if not isinstance(logits, list):
            return "MARKET"
        options = ["MARKET", "LIMIT", "CANCEL"]
        idx = int(np.argmax(logits))
        return options[idx] if idx < len(options) else "MARKET"
