import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import redis

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

        self.lr = self.config.get("lr", 0.001)
        self.batch_size = self.config.get("batch_size", 32)
        self.gamma = self.config.get("gamma", 0.99)
        self.bootstrap_candles = self.config.get("bootstrap_candles", 100)
        self.action_index_map = self.config.get("action_index_map", {"SELL": 0, "HOLD": 1, "BUY": 2})

        buffer_size = self.config.get("buffer_size", 1000)
        strategy = self.config.get("buffer_strategy", "static")
        self.buffer = ReplayBuffer(max_size=buffer_size, strategy=strategy)

        self.reward_agent = RewardAgent(evaluator_agent=None, state_tracker=None)
        self.state_builder = FeatureStateBuilder()
        self.indicator_agents = load_indicator_agents()
        self.model_dir = self.config.get("model_dir", "/tmp")
        self.model_manager = ModelManager(model_dir=self.model_dir)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _build_model(self):
        model_config = self.config.get("model_config")
        selected_profile = self.config.get("model_profile") or model_config.get("architecture", {}).get("selected_profile")
        if not model_config:
            raise ValueError("❌ model_config not found in config.")
        self.logger.info(f"📦 Building model with profile: {selected_profile}")
        model = DynamicQNetwork(model_config, profile=selected_profile).to(self.device)
        model.output_heads_config = model_config.get("output_heads", {})
        self.logger.info(f"✅ Model initialized with input dim: {model.input_dim}")
        return model

    def _derive_quantity_class(self, state, action):
        try:
            price = state.get("current_price", 1.0)
            balance = state.get("balance", 0.0)
            inventory = state.get("inventory", 0.0)
            drawdown = state.get("drawdown_pct", 0.0)
            risk_factor = 0.5 if drawdown > 0.25 else 1.0

            if price <= 0:
                self.logger.warning(f"⚠️ Invalid price ({price}) detected. Using default quantity class 0.")
                return 0

            if action == "BUY":
                affordable = balance / price
                raw_quantity = 0.9 * affordable * risk_factor
            elif action == "SELL":
                affordable = inventory
                raw_quantity = 0.9 * affordable * risk_factor
            else:
                return 0  # HOLD

            if affordable <= 0:
                self.logger.warning(f"⚠️ Affordable quantity is zero or negative for action {action}. Returning class 0.")
                return 0

            ratio = raw_quantity / (affordable + 1e-8)

            q_class = (
                0 if ratio <= 0.1 else
                1 if ratio <= 0.3 else
                2 if ratio <= 0.6 else
                3
            )

            self.logger.debug(f"📦 Quantity Derivation | Action: {action} | Price: {price:.2f} | "
                            f"Affordable: {affordable:.4f} | Raw Qty: {raw_quantity:.4f} | "
                            f"Class: {q_class}")
            return q_class

        except Exception as ex:
            self.logger.exception(f"💥 Failed to derive quantity class: {ex}")
            return 0



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

            if not isinstance(s, dict) or not isinstance(s_next, dict):
                self.logger.warning(f"⚠️ Skipping invalid state during bootstrap. s={type(s)}, s_next={type(s_next)}")
                continue

            try:
                votes = [agent.vote(s) for agent in self.indicator_agents]
                action = self.aggregate_votes(votes)
                reward = self.reward_agent.compute_reward_from_ticks(s, action, s_next)
                q_class = self._derive_quantity_class(s, action)
                self.buffer.add((s, action, reward, s_next, q_class))
            except Exception as e:
                self.logger.warning(f"⚠️ Skipping experience due to error: {e}")

        self.logger.info(f"✅ Bootstrapping completed with {len(self.buffer)} experiences.")
        self.train_on_buffer(source="synthetic")

    def aggregate_votes(self, votes):
        return max(set(votes), key=votes.count)

    def extract_ordered_vector(self, state_dict, feature_order):
        return [state_dict.get(key, 0.0) for key in feature_order]

    def train_on_buffer(self, source="buffer"):
        if len(self.buffer) == 0:
            self.logger.warning("⚠️ No experiences to train on.")
            return

        self.feature_order = getattr(self.model, "feature_order", [])
        raw_batch = self.buffer.sample(min(self.batch_size, len(self.buffer)))

        valid_batch = []
        for i, e in enumerate(raw_batch):
            if not isinstance(e, tuple):
                self.logger.warning(f"❌ Skipping item at index {i}: Not a tuple → {type(e)}")
                continue
            if len(e) != 5:
                self.logger.warning(f"❌ Skipping tuple at index {i}: Expected length=5, got {len(e)} → {e}")
                continue
            if not isinstance(e[0], dict) or not isinstance(e[3], dict):
                self.logger.warning(f"❌ Skipping tuple at index {i}: state or next_state not dict → {type(e[0])}, {type(e[3])}")
                continue
            if not isinstance(e[1], str) or not isinstance(e[2], (int, float)) or not isinstance(e[4], int):
                self.logger.warning(f"❌ Skipping tuple at index {i}: action/reward/quantity class type mismatch → {e}")
                continue
            valid_batch.append(e)

        batch = valid_batch
        if len(batch) == 0:
            self.logger.warning("⚠️ All sampled experiences were malformed.")
            return

        try:
            # Convert to tensors
            states = torch.stack([
                torch.tensor(self.extract_ordered_vector(e[0], self.feature_order), dtype=torch.float32)
                for e in batch
            ]).to(self.device)

            next_states = torch.stack([
                torch.tensor(self.extract_ordered_vector(e[3], self.feature_order), dtype=torch.float32)
                for e in batch
            ]).to(self.device)

            actions = torch.tensor([self.action_index_map.get(e[1], -1) for e in batch], dtype=torch.int64).to(self.device)
            rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(self.device)
            quantity_classes = torch.tensor([e[4] for e in batch], dtype=torch.long).to(self.device)

            # Check if any invalid actions
            if (actions < 0).any():
                self.logger.warning("❌ Found invalid action index in batch. Dumping actions:")
                self.logger.warning(f"{[e[1] for e in batch]}")
                return

            # Forward pass
            model_output = self.model(states)
            target_output = self.target_model(next_states)

            # Q-value loss
            q_values = model_output["signal_logits"].gather(1, actions.unsqueeze(1)).squeeze()
            next_q = target_output["signal_logits"].max(1)[0].detach()
            target_q = rewards + self.gamma * next_q
            signal_loss = nn.MSELoss()(q_values, target_q)

            # Quantity class loss
            quantity_loss = nn.CrossEntropyLoss()(model_output["quantity"], quantity_classes)
            total_loss = signal_loss + self.config.get("quantity_loss_weight", 0.1) * quantity_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.training_step += 1
            self.logger.info(
                f"🧠 Step {self.training_step}: Total={total_loss.item():.4f}, "
                f"Signal={signal_loss.item():.4f}, Qty={quantity_loss.item():.4f}, "
                f"Batch Size={len(batch)}"
            )

            # Metrics
            if self.redis_conn:
                push_metric_to_redis(self.redis_conn, self.training_step, total_loss.item(), self.symbol)

            # Optional buffer resizing
            if self.buffer.strategy == "dynamic":
                if total_loss.item() > 0.3:
                    self.buffer.resize(500)
                elif total_loss.item() < 0.05:
                    self.buffer.resize(2000)

        except Exception as ex:
            self.logger.exception(f"💥 Training failed on buffer batch: {ex}")
    
