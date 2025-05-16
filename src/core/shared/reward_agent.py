from core.utils.type_safe import safe_float

class RewardAgent:
    def __init__(self, config=None):
        self.config = config or {}
        self.weights = self.config.get("reward_weights", {})
        self.default_penalty = self.config.get("failed_trade_penalty", -1.0)
        self.timeout_penalty = self.config.get("timeout_penalty", -0.5)
        self.reversal_bonus = self.config.get("reversal_bonus", 0.3)

    def compute_reward(self, trade_outcome, model_output):
        if not getattr(trade_outcome, "was_executed", True):
            return (
                self.default_penalty,
                {"penalty": self.default_penalty, "reason": "TradeFailed"},
                {"executed": False, "penalty_reason": "InsufficientBalance"}
            )

        # ✅ Safe values from trade outcome
        pnl = safe_float(getattr(trade_outcome, "realized_pnl", 0.0))
        hold_reward = safe_float(getattr(trade_outcome, "unrealized_pnl", 0.0))
        drawdown = safe_float(getattr(trade_outcome, "drawdown", 0.0))

        # ✅ Safe values from model output
        confidence = safe_float(model_output.get("confidence", 0.0))
        stability = safe_float(model_output.get("signal_stability_score", 0.0))
        volatility_pct = safe_float(model_output.get("volatility_pct", 0.0))
        spread_volatility = safe_float(model_output.get("spread_volatility", 0.0))
        slippage_pct = safe_float(model_output.get("slippage_pct", 0.0))
        orderbook_imbalance = safe_float(model_output.get("orderbook_imbalance", 0.0))
        exit_reason = model_output.get("exit_reason", "inference")

        # 🎯 Weighted Reward
        reward = (
            self.weights.get("pnl", 1.0) * pnl +
            self.weights.get("hold", 0.5) * hold_reward +
            self.weights.get("drawdown", -0.3) * drawdown +
            self.weights.get("confidence", 0.2) * confidence +
            self.weights.get("stability", -0.2) * stability +
            self.weights.get("volatility", 0.1) * volatility_pct +
            self.weights.get("spread_volatility", -0.1) * spread_volatility +
            self.weights.get("slippage", -0.15) * slippage_pct +
            self.weights.get("orderbook_imbalance", 0.05) * orderbook_imbalance
        )

        # 🚫 Exit condition modifiers
        if exit_reason == "TIMEOUT":
            reward += self.timeout_penalty
        elif exit_reason == "REVERSE_EXIT":
            reward += self.reversal_bonus

        breakdown = {
            "pnl_component": pnl,
            "hold_component": hold_reward,
            "drawdown_component": drawdown,
            "confidence_component": confidence,
            "stability_component": stability,
            "volatility_component": volatility_pct,
            "spread_volatility_component": spread_volatility,
            "slippage_component": slippage_pct,
            "orderbook_imbalance_component": orderbook_imbalance,
            "exit_reason": exit_reason,
        }

        metadata = {
            "executed": True,
            "reason_weights": model_output.get("reason_weights"),
            "reward_weights": model_output.get("reward_weights"),
            "exit_reason": exit_reason
        }

        return reward, breakdown, metadata
