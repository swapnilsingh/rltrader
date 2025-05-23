import numpy as np
import time
from datetime import datetime
from core.decorators.decorators import inject_logger
from engine.trade_decision_unit import TradeDecisionUnit
from core.shared.trade_outcome import TradeOutcome
from core.models.metrics import RewardMetrics

@inject_logger()
class EnhancedInferenceEngine:
    log_level = "INFO"

    def __init__(self, model_agent, config, wallet, evaluator,
                 experience_writer, signal_publisher, symbol,
                 redis_client, logger=None):  # ✅ Redis client injected
        self.agent = model_agent
        self.config = config
        self.wallet = wallet
        self.evaluator = evaluator
        self.writer = experience_writer
        self.publisher = signal_publisher
        self.redis_client = redis_client  # ✅ Store redis client
        self.symbol = symbol
        self.metric_key = f"metrics:inference:{symbol}"
        self.logger = logger or self.logger

        self.decoder = TradeDecisionUnit()
        self.prev_feature_dict = {}
        self.prev_wallet_snapshot = wallet.get_state_dict(1.0)
        self.prev_action = None

    def run(self, state_dict, current_price, timestamp):
        model_output = self.run_inference(state_dict)
        self.logger.debug(f"📊 Model output: {model_output}")
        decision = self.decode_trade(model_output)
        self.logger.debug(f"🧠 Decoded decision: {decision}")
        return self.process_trade(decision, state_dict, model_output, current_price, timestamp)

    def run_inference(self, state_dict):
        input_vector = self.agent.preprocess(state_dict)
        self.logger.debug(f"📥 Input vector shape: {input_vector.shape}")
        return self.agent.predict(input_vector)

    def decode_trade(self, model_output):
        return self.decoder.decode(model_output)

    def process_trade(self, decision, state_dict, model_output, current_price, timestamp):
        self.logger.debug(f"🚀 Processing trade: {decision}")
        if decision["action"] == "HOLD":
            self.logger.debug("🛑 HOLD action. Skipping trade execution.")
            return "HOLD"

        executed = self._simulate_trade(decision["action"], decision["quantity"], current_price)
        self.logger.debug(f"✅ Trade execution result: {executed}")

        reward, breakdown, metadata = self._evaluate_and_log_reward(model_output, executed, current_price, timestamp)
        if reward is None or np.isnan(reward):
            self.logger.warning("⚠️ Invalid reward. Skipping experience write.")
            return decision["action"]

        self.logger.debug(f"🎯 Reward: {reward}, Breakdown: {breakdown}")

        self.writer.write_experience(
            state_dict=self.prev_feature_dict,
            action=decision["action"],
            reward=reward,
            logits=model_output.get("signal_logits"),
            meta={**model_output, **breakdown, **metadata, "next_state": state_dict}
        )
        self.prev_feature_dict = state_dict
        self.prev_wallet_snapshot = self.wallet.get_state_dict(current_price)
        self.prev_action = decision["action"]

        trade_outcome = TradeOutcome.from_wallets(
            prev_wallet=self.prev_wallet_snapshot,
            current_wallet=self.wallet.get_state_dict(current_price),
            price=current_price,
            action=decision["action"],
            timestamp=timestamp,
            executed=executed
        )
        self.logger.debug(f"📈 Trade outcome: {trade_outcome.__dict__}")

        self._publish(model_output, reward, breakdown, current_price, timestamp, metadata, trade_outcome)

        # ✅ Publish reward metrics to Redis
        self._publish_reward_metrics(
            signal_metadata={
                "action": decision["action"],
                "confidence": model_output.get("confidence", 0.0),
                "stability": model_output.get("signal_stability_score", 0.0),
                "quantity": model_output.get("quantity", 0.0),
                "current_price": current_price
            },
            trade_outcome=trade_outcome,
            wallet_snapshot=self.wallet.get_state_dict(current_price),
            timestamp=timestamp
        )

        return decision["action"]

    def _evaluate_and_log_reward(self, parsed, executed, current_price, timestamp):
        result = self.evaluator.evaluate_trade(
            prev_wallet=self.prev_wallet_snapshot,
            current_wallet=self.wallet.get_state_dict(current_price),
            current_price=current_price,
            timestamp=timestamp,
            model_outputs=parsed,
            executed=executed
        )
        return result["reward"], result["reward_breakdown"], result["experience_metadata"]

    def _simulate_trade(self, action, quantity_fraction, price):
        if action == "BUY":
            cost = quantity_fraction * price
            if self.wallet.balance < cost:
                self.logger.debug(f"❌ Insufficient balance to BUY {quantity_fraction} @ {price} (required: {cost}, available: {self.wallet.balance})")
                return False
            self.logger.info(f"📦 Executing BUY @ {price} x {quantity_fraction}")
            self.wallet.buy(price, quantity_fraction)
            return True

        elif action == "SELL":
            if self.wallet.inventory < quantity_fraction:
                self.logger.debug(f"❌ Insufficient inventory to SELL {quantity_fraction} (available: {self.wallet.inventory})")
                return False
            self.logger.info(f"📤 Executing SELL @ {price} x {quantity_fraction}")
            self.wallet.sell(price, quantity_fraction)
            return True

        self.logger.warning(f"⚠️ Unknown trade action: {action}")
        return False

    def _publish(self, parsed, reward, breakdown, price, timestamp, metadata, trade_outcome):
        if self.publisher:
            self.logger.debug("📡 Publishing signal and metrics to Redis.")
            self.publisher.publish_metrics(self.metric_key, {
                "timestamp": datetime.utcnow().isoformat(),
                "price": price,
                "action": self.prev_action,
                "wallet": self.wallet.get_state_dict(price),
                "reward": reward,
                "metrics": breakdown,
                "force_exit": metadata.get("force_exit", False),
                "trade_outcome": trade_outcome.__dict__
            })
            self.publisher.push({
                "timestamp": datetime.utcnow().isoformat(),
                "action": self.prev_action,
                "confidence": parsed.get("confidence"),
                "cooldown": parsed.get("cooldown_timer", 0),
                "quantity": parsed.get("quantity"),
                "reason": parsed.get("reason", "inference"),
                "signal_logits": parsed.get("signal_logits"),
                "force_exit": metadata.get("force_exit", False),
                "pnl": trade_outcome.realized_pnl,
                "drawdown": trade_outcome.drawdown_pct
            })

    def _publish_reward_metrics(self, signal_metadata, trade_outcome, wallet_snapshot, timestamp):
        try:
            raw_qty = signal_metadata.get("quantity", 0.0)
            quantity_scalar = float(raw_qty[0]) if isinstance(raw_qty, list) else float(raw_qty)

            metrics = RewardMetrics(
                timestamp=timestamp,
                symbol=self.symbol,
                action=signal_metadata.get("action", ""),
                confidence=signal_metadata.get("confidence", 0.0),
                stability=signal_metadata.get("stability", 0.0),
                quantity=quantity_scalar,  # ✅ FIXED: must be float, not list
                current_price=signal_metadata.get("current_price", 0.0),
                equity=wallet_snapshot["equity"],
                drawdown_pct=wallet_snapshot["drawdown_pct"],
                realized_pnl=wallet_snapshot["realized_pnl"],
                unrealized_pnl=wallet_snapshot["unrealized_pnl"],
                holding_time=trade_outcome.holding_time if trade_outcome else None,
                reason=trade_outcome.reason if trade_outcome else None,
            )
            self.redis_client.lpush(f"reward_metrics:{self.symbol}", metrics.json())
            self.redis_client.ltrim(f"reward_metrics:{self.symbol}", 0, 199)

            self.logger.debug(f"📊 Published reward metrics: {metrics.dict()}")

        except Exception as e:
            self.logger.warning(f"❌ Failed to publish reward metrics: {e}")

