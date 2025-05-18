import sys
import asyncio
import redis
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

# 🧠 NEW: Import evaluator + reward agent
from core.shared.reward_agent import RewardAgent
from core.evaluator.evaluator_agent import EvaluatorAgent


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

        # Components
        self.experience_writer = ExperienceWriter(self.redis_conn, self.symbol, self.config.get("experience_writer", {}))
        self.agent = InferenceAgent(self.model_path, self.device)
        self.tick_buffer = RollingTickBuffer(maxlen=300)
        self.feature_builder = TickFeatureBuilder(feature_order=self.agent.feature_order)
        self.wallet = Wallet(self.config.get("starting_balance", 1000))
        self.signal_publisher = SignalPublisher(self.redis_conn, symbol=self.symbol)
        self.ws_client = BinanceWebSocketClient(symbol=self.symbol)

        self.metric_key = f"metrics:inference:{self.symbol}"
        self.prev_wallet_snapshot = self.wallet.get_state_dict(current_price=1.0)
        self.prev_price = None
        self._ready_logged = False

        # ✅ NEW: Reward + Evaluation Pipeline
        reward_agent = RewardAgent(config=self.config.get("reward", {}))
        self.evaluator = EvaluatorAgent(reward_agent=reward_agent, logger=self.logger)

        # 🧠 Engine with full modern pipeline
        self.engine = EnhancedInferenceEngine(
            model=self.agent,
            config=self.config,
            wallet=self.wallet,
            experience_writer=self.experience_writer,
            signal_publisher=self.signal_publisher,
            evaluator=self.evaluator,
            prev_wallet_snapshot=self.prev_wallet_snapshot,
            prev_price=self.prev_price,
            symbol=self.symbol,
            metric_key=self.metric_key,
            logger=self.logger
        )

        self.logger.info(f"✅ InferenceRunner initialized for {self.symbol} on {self.device}")

    async def handle_tick(self, tick_msg):
        self.logger.debug(f"📥 Received tick: {tick_msg}")
        try:
            price = float(tick_msg["p"])
            ts = int(tick_msg["T"])
        except Exception as e:
            self.logger.warning(f"❌ Tick parsing failed: {e}")
            return

        self.tick_buffer.add_tick(tick_msg)

        if not self.tick_buffer.is_ready():
            if not self._ready_logged:
                self.logger.debug("⏳ Waiting for tick buffer to fill...")
            return

        if not self._ready_logged:
            self.logger.info("📈 Buffer is now ready. Inference will begin.")
            self._ready_logged = True

        try:
            # 🧠 Build the enriched feature state
            wallet_state = self.wallet.get_state_dict(current_price=price)
            state_vector, feature_dict = self.feature_builder.build(self.tick_buffer, wallet_state)
            
            # 🚀 Run inference and trading logic
            self.engine.run_inference(
                state=state_vector,
                feature_dict=feature_dict,
                current_price=price,
                timestamp=ts
            )

            # 🔁 Update snapshot references for next tick
            self.prev_wallet_snapshot = self.wallet.get_state_dict(price)
            self.prev_price = price
            self.engine.prev_wallet_snapshot = self.prev_wallet_snapshot
            self.engine.prev_price = self.prev_price

        except Exception as e:
            self.logger.exception(f"❌ Tick processing failed: {e}")

    def run(self):
        self.logger.info("🚀 Starting real-time inference loop...")
        asyncio.run(self.ws_client.listen(self.handle_tick))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run InferenceRunner")
    parser.add_argument("--env", type=str, default="local", help="Environment name (e.g., local, prod)")
    args = parser.parse_args()

    InferenceRunner(env=args.env).run()
