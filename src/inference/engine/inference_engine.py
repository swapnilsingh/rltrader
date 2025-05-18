import numpy as np
from datetime import datetime
from core.utils.metrics import push_generic_metric_to_redis
from core.decorators.decorators import inject_logger

@inject_logger()
class EnhancedInferenceEngine:
    log_level="INFO"
    def __init__(self, model, config, wallet=None, exchange=None,
                 experience_writer=None, signal_publisher=None,
                 evaluator=None, prev_wallet_snapshot=None,
                 prev_price=None, symbol=None, metric_key=None, logger=None):
        self.model = model
        self.config = config
        self.wallet = wallet
        self.exchange = exchange
        self.experience_writer = experience_writer
        self.signal_publisher = signal_publisher
        self.evaluator = evaluator
        self.prev_wallet_snapshot = prev_wallet_snapshot
        self.prev_price = prev_price
        self.symbol = symbol or config.get('symbol', 'UNKNOWN')
        self.metric_key = metric_key or f"metrics:inference:{self.symbol}"
        self.env_mode = config.get('environment', {}).get('mode', 'local')
        self.trading_mode = config.get('trading', {}).get('mode', 'paper')

        self.prev_state = None
        self.prev_action = None
        self.last_trade_time = None
        self.prev_feature_dict = {}

        self.active_position = False
        self.last_entry_time = None
        self.holding_horizon = None
        self.entry_price = None
        self.last_buy_time = None
        self.is_cold_start = True

    def run_inference(self, state, feature_dict, current_price=None, timestamp=None):
        self.logger.debug(f"🔍 Starting inference at {timestamp}, price={current_price}")
        self.check_exit_conditions(current_price, timestamp)

        outputs = self._get_model_output(state)
        if outputs is None:
            self.logger.warning("❌ No model outputs received. Skipping inference.")
            return None

        parsed = self._parse_outputs(outputs)
        action = self._determine_action(parsed['signal_logits'], outputs)

        # Cold Start handling
        if self.is_cold_start and not self.wallet.has_position() and self.wallet.balance > 0:
            fixed_fraction = self.config.get("cold_start_quantity_fraction", 0.1)
            available_equity = self.wallet.get_available_equity()
            quantity_fraction = (available_equity * fixed_fraction) / current_price

            if quantity_fraction * current_price > available_equity or quantity_fraction <= 0:
                self.logger.warning("❌ Not enough funds for cold start BUY. Skipping.")
                return "COLD_START_FAILED"

            self.logger.warning(f"🧪 Cold start — forcing BUY with qty={quantity_fraction:.6f}")
            action = "BUY"
            parsed["exec_mode"] = "MARKET"
            self.is_cold_start = False
        else:
            quantity_fraction = self._determine_quantity(parsed.get('quantity'), current_price)

        # Enforce Hold Time: Minimum seconds before selling
        min_hold_secs = self.config.get("trading", {}).get("min_hold_secs", 10)
        if action == "SELL" and self.wallet.has_position():
            held_secs = (timestamp - self.last_buy_time) if self.last_buy_time else float("inf")
            if held_secs < min_hold_secs:
                self.logger.debug(f"⏳ Minimum hold time not met: {held_secs:.2f}s < {min_hold_secs}s. Forcing HOLD.")
                action = "HOLD"

        # Early exit check
        if self._check_early_exit(action, parsed, current_price):
            return "EXIT_EARLY"

        exec_mode = self._determine_exec_mode(parsed['execution_mode'])
        parsed["exec_mode"] = exec_mode

        conf = parsed.get("confidence", 0.0)
        stab = parsed.get("signal_stability_score", 0.0)
        
        # Override CANCEL mode with MARKET if confidence and stability thresholds are met
        if exec_mode == "CANCEL" and action in {"BUY", "SELL"} and conf > 0.25 and stab > 0.25:
            self.logger.debug(f"⚠️ Overriding CANCEL mode → MARKET due to high conf={conf:.2f}, stab={stab:.2f}")
            exec_mode = "MARKET"

        cooldown_period = self._determine_cooldown(parsed['cooldown_timer'])

        if action in ["BUY", "SELL"]:
            self.logger.info(f"🧠 Action={action} | Inventory={self.wallet.inventory:.6f}, Balance={self.wallet.balance:.2f} | Executing {action} with qty={quantity_fraction:.6f}, exec_mode={exec_mode}")

        # Execute trade
        executed = self.execute_trade(action, quantity_fraction, exec_mode,
                                    parsed['stop_loss_pct'], parsed['take_profit_pct'],
                                    conf, stab, cooldown_period, current_price)

        if executed and action == "BUY":
            self.last_buy_time = timestamp

        # Track buy metadata
        self._track_buy_metadata(executed, action, parsed, current_price, timestamp)

        # Evaluate and log reward
        reward, breakdown, metadata = self._evaluate_and_log_reward(parsed, executed, current_price, timestamp)

        # Handle forced exit due to thresholds
        if metadata.get("force_exit", False) and not metadata.get("executed", False):
            self._reset_holding_state()
            return "FORCED_EXIT"

        # Skip if reward is invalid or NaN
        if reward is None or np.isnan(reward):
            self.logger.warning("⚠️ Invalid reward value. Skipping experience logging.")
            return action

        # Log experience and state
        self._log_experience_and_state(state, feature_dict, action, reward, parsed, breakdown, metadata, current_price)

        # Publish metrics and signals to Redis
        self._publish_signals_and_metrics(parsed, reward, breakdown, current_price, timestamp, metadata)

        return action


    def check_exit_conditions(self, current_price, timestamp=None):
        if not self.wallet or not self.wallet.has_position():
            return
        if not self.active_position or not self.last_entry_time or not self.holding_horizon:
            return
        now = timestamp or datetime.utcnow().timestamp()
        elapsed = now - self.last_entry_time
        if elapsed >= self.holding_horizon:
            try:
                self.wallet.sell(current_price, self.wallet.inventory)
                self._reset_holding_state()
            except Exception as e:
                self.logger.warning(f"Timeout exit failed: {e}")

    def _check_early_exit(self, action, parsed, current_price):
        if self.wallet.has_position() and self.active_position and self.prev_action == "BUY" and action == "SELL":
            confidence = parsed.get("confidence", 0)
            stability = parsed.get("signal_stability_score", 0)
            if confidence > 0.9 and stability > 0.8:
                try:
                    self.wallet.sell(current_price, self.wallet.inventory)
                    self._reset_holding_state()
                    return True
                except Exception as e:
                    self.logger.warning(f"Early exit failed: {e}")
        return False

    def _track_buy_metadata(self, executed, action, parsed, current_price, timestamp):
        if executed and action == "BUY":
            self.last_entry_time = timestamp or datetime.utcnow().timestamp()
            self.holding_horizon = parsed.get("expected_holding_time", 20)
            self.entry_price = current_price
            self.active_position = True

    def _evaluate_and_log_reward(self, parsed, executed, current_price, timestamp):
        new_wallet_state = self.wallet.get_state_dict(current_price)
        result = self.evaluator.evaluate_trade(
            prev_wallet=self.prev_wallet_snapshot,
            current_wallet=new_wallet_state,
            current_price=current_price,
            timestamp=timestamp,
            model_outputs=parsed,
            executed=executed
        )
        return result["reward"], result["reward_breakdown"], result["experience_metadata"]

    def _log_experience_and_state(self, state, feature_dict, action, reward, parsed, breakdown, metadata, current_price):
        self.experience_writer.write_experience(
            state_dict=self.prev_feature_dict,
            action=action,
            reward=reward,
            logits=parsed.get("signal_logits"),
            meta={**parsed, **breakdown, **metadata, "next_state": feature_dict}
        )
        self.prev_state = state
        self.prev_feature_dict = feature_dict
        self.prev_action = action
        self.prev_wallet_snapshot = self.wallet.get_state_dict(current_price)

    def _publish_signals_and_metrics(self, parsed, reward, breakdown, current_price, timestamp, metadata):
        if self.signal_publisher and current_price is not None:
            self.signal_publisher.publish_metrics(self.metric_key, {
                "timestamp": datetime.utcnow().isoformat(),
                "price": current_price,
                "action": self.prev_action,
                "wallet": self.wallet.get_state_dict(current_price),
                "reward": reward,
                "metrics": breakdown,
                "force_exit": metadata.get("force_exit", False)
            })
        if self.signal_publisher and parsed.get("signal_logits"):
            self.signal_publisher.push({
                "timestamp": datetime.utcnow().isoformat(),
                "action": self.prev_action,
                "confidence": parsed.get("confidence"),
                "cooldown": parsed.get("cooldown_timer", 0),
                "quantity": parsed.get("quantity"),
                "reason": parsed.get("reason", "inference"),
                "signal_logits": parsed.get("signal_logits"),
                "force_exit": metadata.get("force_exit", False)
            })
        try:
            push_generic_metric_to_redis(
                redis_conn=self.signal_publisher.redis_conn,
                step=timestamp or datetime.utcnow().timestamp(),
                data={**breakdown, **parsed},
                key=f"reward_metrics:{self.symbol}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to push reward weights: {e}")

    def _reset_holding_state(self):
        self.active_position = False
        self.last_entry_time = None
        self.holding_horizon = None
        self.entry_price = None

    def _get_model_output(self, state):
        try:
            return self.model.predict(state) if hasattr(self.model, 'predict') else self.model(state)
        except Exception as e:
            self.logger.error(f"Model inference failed: {e}")
            return None

    def _parse_outputs(self, outputs):
        def to_value(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
            except ImportError:
                pass
            if isinstance(x, np.ndarray):
                return x.item() if x.shape == () else x.tolist()
            if isinstance(x, (list, tuple)):
                return list(x)
            return x

        parsed = {k: to_value(outputs.get(k)) for k in [
            'signal_logits', 'confidence', 'quantity', 'reward_weights',
            'reason_weights', 'execution_mode', 'cooldown_timer',
            'stop_loss_pct', 'take_profit_pct', 'expected_holding_time', 'signal_stability_score'
        ]}

        for key in parsed:
            val = parsed[key]
            if isinstance(val, list) and len(val) == 1:
                parsed[key] = val[0]
        return parsed

    def _determine_action(self, signal_logits, outputs):
        if signal_logits is not None:
            probs = signal_logits if isinstance(signal_logits, list) else signal_logits.tolist()
            idx = int(np.argmax(probs))
            return ["BUY", "SELL", "HOLD"][idx] if idx in [0, 1, 2] else None
        return outputs.get('action') or outputs.get('signal')

    def _determine_exec_mode(self, execution_mode_logits):
        if not isinstance(execution_mode_logits, list):
            return "MARKET"
        idx = int(np.argmax(execution_mode_logits))
        label = ["MARKET", "LIMIT", "CANCEL"][idx] if idx in [0, 1, 2] else "MARKET"
        return label

    def _determine_quantity(self, quantity_logits, price):
        if not isinstance(quantity_logits, list) or not self.wallet or price is None:
            return 1.0
        fraction_map = [0.05, 0.10, 0.20, 0.50]
        idx = int(np.argmax(quantity_logits))
        pct_to_use = fraction_map[idx] if 0 <= idx < len(fraction_map) else fraction_map[0]
        available_equity = self.wallet.get_available_equity()
        return (available_equity * pct_to_use) / price

    def _determine_cooldown(self, cooldown_timer):
        if cooldown_timer is not None:
            try:
                return float(cooldown_timer)
            except Exception:
                return 0.0
        return float(self.config.get('trading', {}).get('cooldown', 0))

    def _should_execute(self, action, confidence, stability, cooldown_period):
        if action is None or action.upper() == "HOLD":
            return False
        return True

    def _handle_cancel_mode(self):
        return False

    def _compute_sl_tp(self, price, sl_pct, tp_pct):
        sl_price = float(price) * (1 - float(sl_pct)) if sl_pct and price else None
        tp_price = float(price) * (1 + float(tp_pct)) if tp_pct and price else None
        return sl_price, tp_price

    def _simulate_trade(self, action, quantity_fraction, price, sl_price, tp_price):
        if not self.wallet:
            self.logger.error("No wallet for paper trading.")
            return False
        if action == "BUY":
            try:
                self.wallet.buy(price, quantity_fraction)
                return True
            except Exception as e:
                self.logger.warning(f"BUY failed: {e}")
                return False
        elif action == "SELL":
            if self.wallet.inventory <= 0:
                return False
            try:
                self.wallet.sell(price, self.wallet.inventory)
                return True
            except Exception as e:
                self.logger.warning(f"SELL failed: {e}")
                return False
        return False

    def _execute_live_trade(self, action, quantity_fraction, price, exec_mode):
        return False

    def execute_trade(self, action, quantity_fraction, exec_mode,
                      stop_loss_pct, take_profit_pct, confidence, stability,
                      cooldown_period, current_price):
        if exec_mode == "CANCEL":
            return self._handle_cancel_mode()
        if not self._should_execute(action, confidence, stability, cooldown_period):
            return False
        sl_price, tp_price = self._compute_sl_tp(current_price, stop_loss_pct, take_profit_pct)
        if self.trading_mode == "live":
            return self._execute_live_trade(action, quantity_fraction, current_price, exec_mode)
        else:
            return self._simulate_trade(action, quantity_fraction, current_price, sl_price, tp_price)