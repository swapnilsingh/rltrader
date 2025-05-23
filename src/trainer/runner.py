import os
import sys
import time
import argparse
import redis
import json
from datetime import datetime

# Setup import path
sys.path.append(os.path.abspath("src"))

from trainer.agent.dqn_trainer import DQNTrainer
from core.utils.config_loader import load_config
from core.utils.yaml_loader import load_yaml
from core.decorators.decorators import inject_logger
from core.utils.model_manager import ModelManager
import torch


@inject_logger()
class UnifiedTrainer:
    log_level = "INFO"

    def __init__(self, mode: str, batches: int = 1):
        self.mode = mode
        self.batches = batches
        self.config = self._load_config()
        self.redis = self._init_redis() if self.mode == "live" else None
        self.trainer = DQNTrainer(config=self.config, redis_conn=self.redis)
        self.logger.info(f"🎛️ ReplayBuffer Strategy: {self.trainer.buffer.strategy}")

        if self.mode == "live":
            redis_cfg = self.config.get("redis", {})
            self.experience_key = redis_cfg.get("experience_key", "experience_queue")
            self.TRAIN_INTERVAL = self.config.get("train_interval", 5)
            self.SAVE_INTERVAL = self.config.get("save_interval", 60)
            self.last_train_time = time.time()
            self.last_save_time = time.time()
            self.bootstrapped = self._perform_bootstrap()

    def _load_config(self):
        config_path = "configs/trainer/config.yaml"
        config = load_config(env=self.mode, path=config_path)

        model_config_path = config.get("model_config_path")
        if not model_config_path or not os.path.exists(model_config_path):
            alt_path = os.path.join("configs", "trainer", "model_config.yaml")
            if os.path.exists(alt_path):
                self.logger.warning(f"⚠️ Overriding invalid path: {model_config_path} → {alt_path}")
                model_config_path = alt_path

        config["model_config"] = load_yaml(model_config_path)
        return config

    def _init_redis(self):
        redis_cfg = self.config.get("redis", {})
        return redis.Redis(
            host=redis_cfg.get("host", "redis"),
            port=redis_cfg.get("port", 6379),
            decode_responses=True,
        )

    def _perform_bootstrap(self):
        model_path = os.path.join(self.trainer.model_manager.model_dir, "model.pt")
        if not self.trainer.model_manager.model_exists():
            self.logger.info(f"🧊 No model found at {model_path}. Running bootstrap...")
            self.trainer.bootstrap()
            self._safe_model_save()
            self.logger.info("✅ Bootstrapping completed and model saved.")
            return True
        else:
            self.logger.info(f"✅ Model already exists at {model_path}. Skipping bootstrap.")
        return False

    def _safe_model_save(self):
        model = self.trainer.model
        model_config = self.config.get("model_config", {})
        self.trainer.model_manager.save_model(model, metadata=model_config)

    def _convert_to_tuple(self, experience: dict):
        try:
            state = experience.get("state", {})
            action = experience.get("action")
            reward = experience.get("reward")
            next_state = experience.get("next_state") or experience.get("metadata", {}).get("next_state", {})

            if not isinstance(state, dict) or not isinstance(next_state, dict):
                self.logger.debug(f"⚠️ Invalid state/next_state type: {type(state)}, {type(next_state)}")
                return None

            if not isinstance(reward, (int, float)) or action not in self.trainer.action_index_map:
                self.logger.debug(f"⚠️ Invalid action/reward: {action} | {reward}")
                return None

            q_class = self.trainer._derive_quantity_class(state, action)
            return (state, action, reward, next_state, q_class)
        except Exception as e:
            self.logger.warning(f"⚠️ Malformed experience tuple: {e}")
            return None

    def run(self):
        return self.run_dry() if self.mode == "dry" else self.run_live()

    def run_dry(self):
        self.logger.info("🚀 Starting dry trainer run...")
        if not self.trainer.model_manager.model_exists():
            self.logger.info("🧪 No model found. Running bootstrap...")
            self.trainer.bootstrap()
            self._safe_model_save()

        for i in range(self.batches):
            self.logger.info(f"🧠 Training batch {i+1}/{self.batches}")
            self.trainer.train_on_buffer()

        self.logger.info("💾 Final model save...")
        self._safe_model_save()

    def run_live(self):
        self.logger.info("🚀 Starting live trainer...")
        while True:
            try:
                raw = self.redis.lpop(self.experience_key)
                if raw:
                    experience = json.loads(raw)
                    tupled = self._convert_to_tuple(experience)
                    if tupled:
                        self.trainer.buffer.add(tupled)
                        self.bootstrapped = False
                    else:
                        self.logger.debug("⚠️ Skipped experience due to invalid format.")
                now = time.time()
                if now - self.last_train_time >= self.TRAIN_INTERVAL:
                    self._train_step(now)
                if now - self.last_save_time >= self.SAVE_INTERVAL:
                    self._save_step(now)
                time.sleep(0.5)
            except KeyboardInterrupt:
                self.logger.info("🛑 Graceful shutdown.")
                break
            except Exception as e:
                self.logger.exception(f"💥 Error in trainer loop: {e}")
                time.sleep(2)

    def _train_step(self, now):
        if not self.bootstrapped and len(self.trainer.buffer) > 0:
            self.logger.info(f"🧠 Training on {len(self.trainer.buffer)} experiences...")
            losses = self.trainer.train_on_buffer()
            if losses:
                avg = sum(losses) / len(losses)
                self.logger.info(f"📉 Avg loss: {avg:.6f}")
            else:
                self.logger.warning("⚠️ No valid experiences for training.")
        else:
            self.logger.warning("⚠️ No valid experiences for training.")
        self.last_train_time = now

    def _save_step(self, now):
        if not self.bootstrapped and len(self.trainer.buffer) > 0:
            self._safe_model_save()
            self.logger.info(f"💾 Model saved at {datetime.now().isoformat()}")
        else:
            self.logger.debug("🛑 Skipping model save — no real experience yet.")
        self.last_save_time = now


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry", "live"], required=True, help="Mode: dry (offline test) or live (train from Redis)")
    parser.add_argument("--batches", type=int, default=1, help="Number of training batches (only used in dry mode)")
    args = parser.parse_args()

    UnifiedTrainer(mode=args.mode, batches=args.batches).run()
