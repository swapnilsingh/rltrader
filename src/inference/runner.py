# File: src/inference/runner.py

import sys
import asyncio
import redis
import argparse
import threading
from core.decorators.decorators import inject_logger
from core.utils.config_loader import load_config
from core.portfolio.wallet import Wallet
from inference.agent.inference_agent import InferenceAgent
from inference.preprocessor.rolling_tick_buffer import RollingTickBuffer
from inference.preprocessor.tick_feature_builder import TickFeatureBuilder
from inference.publisher.signal_publisher import SignalPublisher
from inference.publisher.experience_writer import ExperienceWriter
from inference.connectors.websocket_connector import BinanceWebSocketClient
from inference.engine.inference_engine import EnhancedInferenceEngine
from core.shared.reward_agent import RewardAgent
from core.evaluator.evaluator_agent import EvaluatorAgent
from core.reward.llm_reward_controller import LLMRewardController
from core.utils.state_utils import sanitize_state

@inject_logger()
class InferenceRunner:
    log_level = "INFO"

    def __init__(self, env="local"):
        self.config = load_config(env=env, path="configs/inference/config.yaml")
        self.symbol = self.config.get("symbol", "btcusdt").lower()
        self.model_path = self.config.get("model", {}).get("path", "models/model.pt")
        self.device = self.config.get("device", "cpu")

        # Redis connection
        self.redis_conn = redis.Redis(
            host=self.config["redis"]["host"],
            port=self.config["redis"]["port"],
            db=self.config["redis"].get("db", 0),
            decode_responses=True
        )

        # Core components
        self.wallet = Wallet(self.config.get("starting_balance", 1000))
        self.experience_writer = ExperienceWriter(self.redis_conn, self.symbol, self.config.get("experience_writer", {}))
        self.signal_publisher = SignalPublisher(self.redis_conn, symbol=self.symbol)
        self.ws_client = BinanceWebSocketClient(symbol=self.symbol)

        # Model + feature pipeline
        self.agent = InferenceAgent(self.model_path, self.device)
        self.tick_buffer = RollingTickBuffer(maxlen=300)
        self.feature_builder = TickFeatureBuilder(feature_order=self.agent.feature_order)

        # Reward and evaluation
        self.reward_agent = RewardAgent(config=self.config.get("reward", {}))
        self.evaluator = EvaluatorAgent(reward_agent=self.reward_agent)

        # Refactored Inference Engine
        self.engine = EnhancedInferenceEngine(
            model_agent=self.agent,
            config=self.config,
            wallet=self.wallet,
            evaluator=self.evaluator,
            experience_writer=self.experience_writer,
            signal_publisher=self.signal_publisher,
            redis_client=self.redis_conn,
            symbol=self.symbol
        )

        # 🔁 LLM Reward Controller Background Thread
        interval_secs = self.config.get("llm_interval_secs", 120)
        self.llm_controller = LLMRewardController(
            reward_agent=self.reward_agent,
            redis_conn=self.redis_conn,
            symbol=self.symbol,
            interval_secs=interval_secs
        )
        threading.Thread(target=self.llm_controller.run_loop, daemon=True).start()
        self.logger.info("🧠 LLMRewardController started in background")

        self._ready_logged = False
        self.logger.info(f"✅ InferenceRunner initialized for {self.symbol} on {self.device}")

    async def handle_tick(self, tick_msg):
        self.logger.debug(f"📥 Received tick: {tick_msg}")
        try:
            price = float(tick_msg["p"])
            timestamp = int(tick_msg["T"])
        except Exception as e:
            self.logger.warning(f"❌ Tick parsing failed: {e}")
            return

        self.tick_buffer.add_tick(tick_msg)

        if not self.tick_buffer.is_ready():
            if not self._ready_logged:
                self.logger.debug("⏳ Waiting for tick buffer to fill...")
            return

        if not self._ready_logged:
            self.logger.info("📈 Buffer ready. Starting inference.")
            self._ready_logged = True

        try:
            wallet_state = self.wallet.get_state_dict(price)
            input_vector, feature_dict = self.feature_builder.build(self.tick_buffer, wallet_state)

            state = {
                **feature_dict,
                **wallet_state,
                "current_price": price,
                "timestamp": timestamp
            }

            if not sanitize_state(state):
                self.logger.warning("⚠️ Invalid state (NaN detected), skipping.")
                return

            self.engine.run(state, current_price=price, timestamp=timestamp)

        except Exception as e:
            self.logger.exception(f"❌ Tick processing failed: {e}")

    def run(self):
        self.logger.info("🚀 Starting real-time inference loop...")
        asyncio.run(self.ws_client.listen(self.handle_tick))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run InferenceRunner")
    parser.add_argument("--env", type=str, default="local", help="Environment name (e.g., local, prod)")
    args = parser.parse_args()

    InferenceRunner(env=args.env).run()
