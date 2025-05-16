import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np

from trainer.agent.replay_buffer import ReplayBuffer
from trainer.agent.reward_agent import RewardAgent
from trainer.bootstrap.feature_state_builder import FeatureStateBuilder
from core.agents.agent_factory import load_indicator_agents
from core.utils.model_manager import ModelManager
from core.utils.bootstrap_ohlcv import fetch_initial_ohlcv
from core.utils.state_utils import sanitize_state
from core.models.dynamic_qnetwork import DynamicQNetwork
from core.decorators.decorators import inject_logger
from core.utils.metrics import push_metric_to_redis


@inject_logger()
class DQNTrainer:
    def __init__(self, config, redis_conn=None):
        self.config = config
        self.symbol = self.config.get("symbol")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.redis_conn = redis_conn
        self.training_step = 0

        # Load config values
        self.lr = self.config.get("lr", 0.001)
        self.batch_size = self.config.get("batch_size", 32)
        self.gamma = self.config.get("gamma", 0.99)
        self.bootstrap_candles = self.config.get("bootstrap_candles", 100)
        self.action_index_map = self.config.get("action_index_map", {"SELL": 0, "HOLD": 1, "BUY": 2})
        self.action_value_map = self.config.get("action_value_map", {"SELL": -1, "HOLD": 0, "BUY": 1})

        # Components
        strategy = self.config.get("buffer_strategy", "static")
        buffer_size = self.config.get("buffer_size", 1000)
        self.buffer = ReplayBuffer(max_size=buffer_size, strategy=strategy)

        self.reward_agent = RewardAgent(evaluator_agent=None, state_tracker=None)
        self.state_builder = FeatureStateBuilder()
        self.indicator_agents = load_indicator_agents()
        self.model_dir = self.config.get("model_dir", "/tmp")
        self.model_manager = ModelManager(model_dir=self.model_dir)

        # Model setup
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        model_config = self.config.get("model_config")
        selected_profile = self.config.get("model_profile") or model_config.get("architecture", {}).get("selected_profile")
        if not model_config:
            raise ValueError("❌ model_config not found in config. Ensure it is preloaded before passing.")

        self.logger.info(f"📦 Building model using in-memory config and profile: {selected_profile}")
        model = DynamicQNetwork(model_config, profile=selected_profile).to(self.device)
        model.output_heads_config = model_config.get("output_heads", {})  # 🔁 Preserve output head config for saving
        return model

    def bootstrap(self):
        self.logger.info("🪄 Starting bootstrap phase...")
        candles = fetch_initial_ohlcv(symbol=self.symbol, limit=self.bootstrap_candles)
        states = self.state_builder.build_state(candles)

        if len(states) < 2:
            self.logger.warning("⚠️ Not enough states for bootstrapping")
            return

        for i in range(len(states) - 1):
            s = sanitize_state(states[i])
            s_next = sanitize_state(states[i + 1])
            votes = [agent.vote(s) for agent in self.indicator_agents]
            action = self.aggregate_votes(votes)

            try:
                reward = self.reward_agent.compute_reward_from_ticks(s, action, s_next)
                self.buffer.add((s, action, reward, s_next))
            except Exception as e:
                self.logger.warning(f"⚠️ Skipping experience due to reward error: {e}")

        self.logger.info(f"✅ Bootstrapping completed with {len(self.buffer)} experiences.")
        self.train_on_buffer(source="synthetic")

    def aggregate_votes(self, votes):
        return max(set(votes), key=votes.count)

    def extract_ordered_vector(self, state_dict, feature_order):
        return [state_dict.get(key, 0.0) for key in feature_order]

    def train_on_buffer(self, source="buffer"):
        self.logger.info(f"🧠 Training on {source} experience...")

        if len(self.buffer) == 0:
            self.logger.warning("⚠️ No experience to train on.")
            return

        if not hasattr(self.model, "feature_order"):
            raise ValueError("❌ Model is missing feature_order metadata.")
        self.feature_order = self.model.feature_order

        batch = self.buffer.sample(min(self.batch_size, len(self.buffer)))

        states = torch.tensor(
            [self.extract_ordered_vector(e[0], self.feature_order) for e in batch],
            dtype=torch.float32,
        ).to(self.device)

        next_states = torch.tensor(
            [self.extract_ordered_vector(e[3], self.feature_order) for e in batch],
            dtype=torch.float32,
        ).to(self.device)

        if states.shape[1] != self.model.input_dim:
            raise ValueError(f"❌ Invalid input dim. Got {states.shape[1]}, expected {self.model.input_dim}")

        actions = torch.tensor([self.action_index_map[e[1]] for e in batch], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(self.device)

        model_output = self.model(states)
        target_output = self.target_model(next_states)

        signal_logits = model_output["signal_logits"]
        target_logits = target_output["signal_logits"]

        q_values = signal_logits.gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = target_logits.max(1)[0].detach()
        target_q = rewards + self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.info(f"✅ Training step complete. Loss: {loss.item():.4f}")

        # 🔁 Redis metric push
        self.training_step += 1
        if self.redis_conn:
            push_metric_to_redis(
                redis_conn=self.redis_conn,
                step=self.training_step,
                loss=loss.item(),
                symbol=self.symbol,
                max_len=100
            )

        # 🔄 Buffer strategy adjustment
        if hasattr(self.buffer, "strategy") and self.buffer.strategy == "dynamic":
            if loss.item() > 0.3:
                new_size = 500
                self.logger.warning(f"📉 High loss detected. Shrinking buffer to {new_size}")
                self.buffer.resize(new_size)
            elif loss.item() < 0.05:
                new_size = 2000
                self.logger.info(f"📈 Low loss detected. Expanding buffer to {new_size}")
                self.buffer.resize(new_size)

    def run_training_loop(self):
        import redis
        import json
        import time

        self.logger.info("🔁 Starting online training loop...")

        r_cfg = self.config.get("redis", {})
        redis_client = redis.Redis(
            host=r_cfg.get("host", "localhost"),
            port=r_cfg.get("port", 6379),
            decode_responses=True,
        )
        experience_key = r_cfg.get("experience_key", "experience_queue")
        train_interval = self.config.get("train_interval", 5)
        stale_threshold_sec = self.config.get("stale_experience_threshold", 30)

        last_experience_time = None

        while True:
            try:
                raw = redis_client.lpop(experience_key)
                if raw:
                    try:
                        exp = json.loads(raw)
                        state = exp.get("state", {})
                        action = exp.get("action")
                        reward = exp.get("reward")
                        next_state = exp.get("meta", {}).get("next_state", {})
                        timestamp = exp.get("meta", {}).get("timestamp")

                        if not isinstance(state, dict) or action not in self.action_index_map or not isinstance(reward, (int, float)):
                            self.logger.debug(f"❌ Invalid experience skipped: {exp}")
                            continue

                        if timestamp:
                            ts = int(float(timestamp))
                            now = int(time.time() * 1000)
                            if now - ts > stale_threshold_sec * 1000:
                                self.logger.warning(f"🕒 Stale experience skipped (age={now - ts}ms): {exp}")
                                continue
                            last_experience_time = time.time()

                        self.buffer.add((state, action, reward, next_state))

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to parse experience: {e}")

                if last_experience_time and (time.time() - last_experience_time > stale_threshold_sec):
                    self.logger.warning(f"⏰ No fresh experience received in the last {stale_threshold_sec} seconds.")

                if len(self.buffer) >= self.batch_size:
                    self.logger.info(f"🧠 Training on {len(self.buffer)} experiences...")
                    self.train_on_buffer(source="live")
                else:
                    self.logger.warning("⚠️ Not enough experience to train.")

                time.sleep(train_interval)

            except KeyboardInterrupt:
                self.logger.info("🛑 Stopping training loop.")
                break
            except Exception as e:
                self.logger.exception(f"💥 Error during training loop: {e}")
                time.sleep(2)
