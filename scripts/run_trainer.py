import os
import sys
import time
import argparse
import redis
import json
from datetime import datetime

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["dry", "live"], required=True, help="Mode: dry (offline test) or live (train from Redis)")
parser.add_argument("--batches", type=int, default=1, help="Number of training batches (only used in dry mode)")
args = parser.parse_args()

# Setup import path
sys.path.append(os.path.abspath("src"))

from trainer.agent.dqn_trainer import DQNTrainer
from core.utils.config_loader import load_config
from core.utils.yaml_loader import load_yaml
from core.decorators.decorators import inject_logger


@inject_logger()
class UnifiedTrainer:
    def __init__(self):
        config_path = "configs/trainer/config.yaml"
        self.config = load_config(env=args.mode, path=config_path)

        # Load model config and inject into self.config
        model_config_path = self.config.get("model_config_path")

        if not os.path.exists(model_config_path):
            alt_path = os.path.join("configs", "trainer", "model_config.yaml")
            if os.path.exists(alt_path):
                self.logger.warning(f"⚠️ Overriding invalid path: {model_config_path} → {alt_path}")
                model_config_path = alt_path

        self.config["model_config"] = load_yaml(model_config_path)

        self.redis = None
        if args.mode == "live":
            redis_cfg = self.config.get("redis", {})
            self.redis = redis.Redis(
                host=redis_cfg.get("host", "redis"),
                port=redis_cfg.get("port", 6379),
                decode_responses=True,
            )
            self.experience_key = redis_cfg.get("experience_key", "experience_queue")
            self.TRAIN_INTERVAL = self.config.get("train_interval", 5)
            self.SAVE_INTERVAL = self.config.get("save_interval", 60)
            self.last_train_time = time.time()
            self.last_save_time = time.time()

        redis_conn = self.redis if args.mode == "live" else None
        self.trainer = DQNTrainer(config=self.config, redis_conn=redis_conn)
        self.logger.info(f"🎛️ ReplayBuffer Strategy: {self.trainer.buffer.strategy}")

    def run(self):
        if args.mode == "dry":
            self.run_dry()
        else:
            self.run_live()

    def run_dry(self):
        print("🚀 Starting dry trainer run...")
        print(f"📦 Experience buffer size: {len(self.trainer.buffer)}")

        if not self.trainer.model_manager.model_exists():
            print("🧪 No model found. Running bootstrap...")
            self.trainer.bootstrap()
            print(f"📦 Post-bootstrap buffer: {len(self.trainer.buffer)}")
        else:
            print("✅ Model already exists. Skipping bootstrap.")

        for i in range(args.batches):
            print(f"🧠 Training batch {i+1}/{args.batches}")
            self.trainer.train_on_buffer()

        model_path = os.path.join(self.trainer.model_manager.model_dir, "model.pt")
        if os.path.exists(model_path):
            self.trainer.model_manager.save_model(
                self.trainer.model,
                metadata=self.config.get("model_config")
            )
            print(f"💾 Model saved at: {model_path}")
        else:
            print("❌ Model file not found.")

    def run_live(self):
        self.logger.info("🚀 Starting live trainer...")

        if not self.trainer.model_manager.model_exists():
            self.logger.warning("🚨 No model found! Bootstrapping before live training.")
            self.trainer.bootstrap()
            self.trainer.model_manager.save_model(
                self.trainer.model,
                metadata=self.config.get("model_config")
            )
            self.logger.info("✅ Bootstrapping completed and model saved.")

        self.bootstrapped = True

        while True:
            try:
                new_experience_received = False

                raw = self.redis.lpop(self.experience_key)
                if raw:
                    try:
                        experience = json.loads(raw)
                        self.trainer.buffer.add(tuple(experience))
                        new_experience_received = True
                        self.bootstrapped = False
                    except Exception as e:
                        self.logger.warning(f"⚠️ Malformed experience: {e}")

                now = time.time()

                if now - self.last_train_time >= self.TRAIN_INTERVAL:
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

                if now - self.last_save_time >= self.SAVE_INTERVAL:
                    if not self.bootstrapped and len(self.trainer.buffer) > 0:
                        self.trainer.model_manager.save_model(
                            self.trainer.model,
                            metadata=self.config.get("model_config")
                        )
                        self.logger.info(f"💾 Model saved at {datetime.now().isoformat()}")
                    else:
                        self.logger.debug("🛑 Skipping model save — no real experience yet.")
                    self.last_save_time = now

                time.sleep(0.5)

            except KeyboardInterrupt:
                self.logger.info("🛑 Graceful shutdown.")
                break
            except Exception as e:
                self.logger.exception(f"💥 Error in trainer loop: {e}")
                time.sleep(2)


if __name__ == "__main__":
    UnifiedTrainer().run()
