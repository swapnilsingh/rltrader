from core.shared.trade_outcome import TradeOutcome

class EvaluatorAgent:
    def __init__(self, reward_agent, logger=None):
        self.reward_agent = reward_agent
        self.logger = logger

    def evaluate_trade(
        self,
        prev_wallet,
        current_wallet,
        current_price,
        timestamp,
        model_outputs,
        executed=True
    ):
        # Build outcome from wallet states
        outcome = TradeOutcome.from_wallets(
            prev_wallet=prev_wallet,
            current_wallet=current_wallet,
            price=current_price,
            action=model_outputs.get("action", "HOLD"),
            timestamp=timestamp,
            executed=executed
        )

        if not executed and self.logger:
            self.logger.debug("⚠️ Trade was rejected. Evaluating with penalty.")

        # Fill missing model outputs with safe defaults
        enriched_outputs = {
            **model_outputs,
            "volatility_pct": model_outputs.get("volatility_pct", 0.0),
            "spread_volatility": model_outputs.get("spread_volatility", 0.0),
            "slippage_pct": model_outputs.get("slippage_pct", 0.0),
            "orderbook_imbalance": model_outputs.get("orderbook_imbalance", 0.0),
        }

        # Compute reward and metadata
        reward, breakdown, metadata = self.reward_agent.compute_reward(outcome, enriched_outputs)

        if self.logger:
            self.logger.debug(f"🎯 Reward breakdown: {breakdown}")

        return {
            "reward": reward,
            "reward_breakdown": breakdown,
            "experience_metadata": metadata
        }
