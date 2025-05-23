# File: core/reward/reward_agent.py

import numpy as np
from core.utils.type_safe import safe_float

class RewardAgent:
    def __init__(self, config=None):
        self.config = config or {}
        self.weights = self.config.get("reward_weights", {})

        reward_cfg = self.config.get("reward", {})
        self.default_penalty = reward_cfg.get("failed_trade_penalty", -1.0)
        self.timeout_penalty = reward_cfg.get("timeout_penalty", -0.5)
        self.reversal_bonus = reward_cfg.get("reversal_bonus", 0.3)
        self.cancel_penalty = reward_cfg.get("cancel_penalty", -0.1)

        self.dynamic_drawdown_enabled = True
        self.drawdown_threshold = reward_cfg.get("drawdown_threshold", 0.05)
        self.high_drawdown_penalty = reward_cfg.get("high_drawdown_penalty", -1.5)
        self.drawdown_execution_limit = reward_cfg.get("drawdown_limit", 0.10)
        self.low_drawdown_penalty = self.weights.get("drawdown_pct", -0.3)

        self.equity_breakout_bonus = reward_cfg.get("equity_breakout_bonus", 2.0)
        self.min_hold_secs = reward_cfg.get("min_hold_secs", 10)
        self.early_exit_penalty = reward_cfg.get("early_exit_penalty", -0.5)

    def update_weights(self, new_weights: dict):
        if isinstance(new_weights, dict):
            # Clamp all weights between -1.0 and 2.0
            for k, v in new_weights.items():
                if isinstance(v, (int, float)):
                    self.weights[k] = float(np.clip(v, -1.0, 2.0))

    def compute_reward(self, trade_outcome, model_output):
        if hasattr(trade_outcome, "was_executed") and not getattr(trade_outcome, "was_executed"):
            return self._reward_failed_execution(model_output)

        action = getattr(trade_outcome, "action", "HOLD")
        drawdown_pct = safe_float(getattr(trade_outcome, "drawdown_pct", 0.0))
        if self._is_execution_blocked(action, drawdown_pct):
            return self._build_execution_blocked_response()

        c = self._extract_components(trade_outcome, model_output)
        drawdown_penalty, force_exit = self._compute_drawdown_penalty(c["equity"], c["initial_balance"])
        exec_mode_penalty = self._compute_cancel_penalty(model_output, action)

        pnl = c["pnl"]
        profit_multiplier = self.config.get("reward", {}).get("profit_multiplier", 5.0)
        profit_boost = pnl * profit_multiplier if action == "SELL" and pnl > 0 else 0.0

        early_exit_penalty = -1.0 if model_output.get("exit_reason") == "REVERSE_EXIT" else 0.0

        reward = (
            self.weights.get("pnl", 1.0) * pnl +
            self.weights.get("hold", 0.5) * c["hold_reward"] +
            drawdown_penalty * c["drawdown_pct"] +
            self.weights.get("confidence", 0.2) * c["confidence"] +
            self.weights.get("stability", -0.2) * c["stability"] +
            self.weights.get("volatility", 0.1) * c["volatility_pct"] +
            self.weights.get("spread_volatility", -0.1) * c["spread_volatility"] +
            self.weights.get("slippage", -0.15) * c["slippage_pct"] +
            self.weights.get("orderbook_imbalance", 0.05) * c["orderbook_imbalance"] +
            exec_mode_penalty +
            early_exit_penalty +
            profit_boost
        )

        reward += self._compute_equity_bonus(c["equity_peak"], c["prev_equity_peak"])
        reward += self._apply_exit_adjustments(c["exit_reason"])
        reward = np.clip(reward, -10, 10)

        breakdown = self._build_breakdown(c, drawdown_penalty, exec_mode_penalty, reward)
        metadata = self._build_metadata(model_output, c["exit_reason"], force_exit)
        return reward, breakdown, metadata

    def _reward_failed_execution(self, model_output):
        penalty = self.default_penalty

        # ⛔ Quantity-aware penalty (discourage large quantity if failed)
        quantity = safe_float(model_output.get("quantity", 1.0))
        quantity_penalty = min(1.0, quantity)  # penalize larger orders
        penalty *= (1.0 + quantity_penalty)    # scale penalty

        breakdown = {
            "pnl_component": 0.0,
            "hold_component": 0.0,
            "drawdown_component": 0.0,
            "drawdown_penalty_used": 0.0,
            "confidence_component": model_output.get("confidence", 0.0),
            "stability_component": model_output.get("signal_stability_score", 0.0),
            "volatility_component": model_output.get("volatility_pct", 0.0),
            "spread_volatility_component": model_output.get("spread_volatility", 0.0),
            "slippage_component": model_output.get("slippage_pct", 0.0),
            "orderbook_imbalance_component": model_output.get("orderbook_imbalance", 0.0),
            "exec_mode_penalty": 0.0,
            "equity_breakout_bonus": 0.0,
            "exit_reason": "FAILED_EXECUTION",
            "final_reward": penalty
        }
        metadata = {
            "executed": False,
            "exit_reason": "FAILED_EXECUTION",
            "force_exit": True
        }
        return penalty, breakdown, metadata

    def _is_execution_blocked(self, action, drawdown_pct):
        return self.dynamic_drawdown_enabled and drawdown_pct >= self.drawdown_execution_limit and action != "HOLD"

    def _build_execution_blocked_response(self):
        return (
            self.default_penalty,
            {"penalty": self.default_penalty, "reason": "ExecutionBlockedDueToDrawdown"},
            {
                "executed": False,
                "penalty_reason": "DrawdownLimitExceeded",
                "force_exit": True
            }
        )

    def _extract_components(self, trade_outcome, model_output):
        return {
            "pnl": safe_float(getattr(trade_outcome, "realized_pnl", 0.0)),
            "hold_reward": safe_float(getattr(trade_outcome, "unrealized_pnl", 0.0)),
            "equity": safe_float(getattr(trade_outcome, "equity", 0.0)),
            "initial_balance": safe_float(getattr(trade_outcome, "initial_balance", 1000.0)),
            "equity_peak": safe_float(getattr(trade_outcome, "equity_peak", 0.0)),
            "prev_equity_peak": safe_float(getattr(trade_outcome, "prev_equity_peak", 0.0)),
            "drawdown_pct": safe_float(getattr(trade_outcome, "drawdown_pct", 0.0)),
            "confidence": safe_float(model_output.get("confidence", 0.0)),
            "stability": safe_float(model_output.get("signal_stability_score", 0.0)),
            "volatility_pct": safe_float(model_output.get("volatility_pct", 0.0)),
            "spread_volatility": safe_float(model_output.get("spread_volatility", 0.0)),
            "slippage_pct": safe_float(model_output.get("slippage_pct", 0.0)),
            "orderbook_imbalance": safe_float(model_output.get("orderbook_imbalance", 0.0)),
            "exit_reason": model_output.get("exit_reason", "inference"),
            "exec_mode": model_output.get("exec_mode", "CANCEL"),
        }

    def _compute_drawdown_penalty(self, equity, initial):
        pct_remaining = equity / initial if initial > 0 else 1.0
        penalty = 0.0
        override_required = False
        if pct_remaining <= 0.10:
            penalty = -10.0
            override_required = True
        elif pct_remaining <= 0.20:
            penalty = -6.0
        elif pct_remaining <= 0.30:
            penalty = -5.0
        elif pct_remaining <= 0.50:
            penalty = -4.0
        elif pct_remaining <= 0.70:
            penalty = -3.0
        elif pct_remaining <= 0.90:
            penalty = -2.0
        elif pct_remaining <= 0.95:
            penalty = -1.0
        return penalty, override_required

    def _compute_cancel_penalty(self, model_output, action):
        exec_mode = model_output.get("exec_mode", "CANCEL")
        if exec_mode != "CANCEL" or action == "HOLD":
            return 0.0
        confidence = safe_float(model_output.get("confidence", 0.0))
        volatility = safe_float(model_output.get("volatility_pct", 0.0))
        return self.cancel_penalty * (1.0 + confidence + max(1.0 - volatility, 0.0))

    def _compute_equity_bonus(self, equity_peak, prev_equity_peak):
        return self.equity_breakout_bonus if equity_peak > prev_equity_peak else 0.0

    def _apply_exit_adjustments(self, exit_reason):
        if exit_reason == "TIMEOUT":
            return self.timeout_penalty
        elif exit_reason == "REVERSE_EXIT":
            return self.reversal_bonus
        return 0.0

    def _build_breakdown(self, c, drawdown_penalty, exec_mode_penalty, reward):
        return {
            "pnl_component": c["pnl"],
            "hold_component": c["hold_reward"],
            "drawdown_component": c["drawdown_pct"],
            "drawdown_penalty_used": drawdown_penalty,
            "confidence_component": c["confidence"],
            "stability_component": c["stability"],
            "volatility_component": c["volatility_pct"],
            "spread_volatility_component": c["spread_volatility"],
            "slippage_component": c["slippage_pct"],
            "orderbook_imbalance_component": c["orderbook_imbalance"],
            "exec_mode_penalty": exec_mode_penalty,
            "equity_breakout_bonus": self._compute_equity_bonus(c["equity_peak"], c["prev_equity_peak"]),
            "exit_reason": c["exit_reason"],
            "final_reward": reward,
        }

    def _build_metadata(self, model_output, exit_reason, force_exit=False):
        return {
            "executed": True,
            "reason_weights": model_output.get("reason_weights"),
            "reward_weights": model_output.get("reward_weights"),
            "exit_reason": exit_reason,
            "exec_mode": model_output.get("exec_mode", "CANCEL"),
            "force_exit": force_exit
        }
