from core.utils.type_safe import safe_float

class RewardTracker:
    def __init__(self, config=None):
        self.config = config or {}
        self.weights = self.config.get("reward_weights", {
            "pnl": 1.0,
            "hold": 0.5,
            "drawdown": -0.3,
            "confidence": 0.2,
            "stability": -0.2,
        })

    def compute_pnl_component(self, prev_wallet, curr_wallet):
        return safe_float(curr_wallet.realized_pnl - prev_wallet.realized_pnl)

    def compute_hold_component(self, prev_price, curr_price, entry_price, inventory):
        if inventory > 0 and entry_price > 0:
            return safe_float((curr_price - prev_price) * inventory)
        return 0.0

    def compute_drawdown_penalty(self, wallet, curr_price):
        equity = safe_float(wallet.compute_equity(curr_price))
        max_equity = safe_float(wallet.max_equity)
        drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
        return drawdown

    def compute_confidence_bonus(self, model_output):
        return safe_float(model_output.get("confidence"))

    def compute_stability_penalty(self, model_output):
        score = safe_float(model_output.get("signal_stability_score"))
        return 1.0 - score if score is not None else 0.0

    def compute_total_reward(self, prev_wallet, curr_wallet, prev_price, curr_price,
                             model_output=None, track_metrics=False):
        model_output = model_output or {}

        pnl_r = self.compute_pnl_component(prev_wallet, curr_wallet)
        hold_r = self.compute_hold_component(prev_price, curr_price, curr_wallet.entry_price, curr_wallet.inventory)
        drawdown_r = self.compute_drawdown_penalty(curr_wallet, curr_price)
        conf_r = self.compute_confidence_bonus(model_output)
        stab_r = self.compute_stability_penalty(model_output)

        model_weights = model_output.get("reward_weights")
        if model_weights:
            weights = [safe_float(w) for w in model_weights]
        else:
            weights = [
                self.weights["pnl"],
                self.weights["hold"],
                self.weights["drawdown"],
                self.weights["confidence"],
                self.weights["stability"],
            ]

        reward = (
            weights[0] * pnl_r +
            weights[1] * hold_r +
            weights[2] * drawdown_r +
            weights[3] * conf_r +
            weights[4] * stab_r
        )

        metrics = {
            "reward": reward,
            "pnl_component": pnl_r,
            "hold_component": hold_r,
            "drawdown_penalty": drawdown_r,
            "confidence_bonus": conf_r,
            "stability_penalty": stab_r,
            "weights_used": weights
        } if track_metrics else {}

        return (reward, metrics) if track_metrics else reward
