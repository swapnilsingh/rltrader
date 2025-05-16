import time
import json
from core.decorators.decorators import inject_logger

@inject_logger()
class ExperienceWriter:
    def __init__(self, redis_conn, symbol="btcusdt", config=None):
        self.redis_conn = redis_conn
        self.symbol = symbol.lower()
        self.config = config or {}

        self.model_version = self.config.get("model_version", "v1")
        self.max_experience_length = self.config.get("max_length", 1000)

        keys = self.config.get("keys", {})
        self.experience_key = keys.get("experience", "experience_queue")
        self.trade_log_key = keys.get("trade_log", "trade_log")

    def write_experience(self, state_dict, action, reward, logits=None, meta=None):
        self.logger.debug("🧪 Raw experience before sanitization:")
        self.logger.debug(f"  Action: {action}")
        self.logger.debug(f"  Reward: {reward}")
        self.logger.debug(f"  State Dict Keys: {list(state_dict.keys()) if isinstance(state_dict, dict) else 'Invalid type'}")
        self.logger.debug(f"  Logits: {logits.tolist() if hasattr(logits, 'tolist') else logits}")
        self.logger.debug(f"  Meta: {meta}")

        if action is None or not isinstance(state_dict, dict) or not isinstance(reward, (int, float)):
            self.logger.warning("⚠️ Invalid experience input. Skipping.")
            return

        clean_state = {k: float(v) for k, v in state_dict.items() if isinstance(v, (float, int))}

        meta = meta or {}
        meta.setdefault("timestamp", time.time())

        next_state = meta.get("next_state", {})
        if not isinstance(next_state, dict):
            self.logger.warning("⚠️ next_state is not a dict. Defaulting to empty.")
            next_state = {}

        if logits is not None:
            try:
                meta["logits"] = logits.tolist() if hasattr(logits, "tolist") else logits
            except Exception:
                self.logger.warning("⚠️ Failed to convert logits to list. Skipping.")

        # ✅ Dict format for trainer compatibility
        trainer_exp = {
            "state": clean_state,
            "action": action,
            "reward": float(reward),
            "next_state": next_state,
            "meta": meta
        }

        try:
            self.redis_conn.lpush(self.experience_key, json.dumps(trainer_exp))
            self.redis_conn.ltrim(self.experience_key, 0, self.max_experience_length - 1)
            self.logger.debug(f"📨 Experience pushed to Redis: {self.experience_key}")
        except Exception as e:
            self.logger.warning(f"❌ Failed to push experience to Redis: {e}")

