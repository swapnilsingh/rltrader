
## 🧠 Neurotrade Model Schema — `DynamicQNetwork`

### ✅ Overview

This schema defines the input and output structure of the `DynamicQNetwork`, a configurable multi-headed neural network used for autonomous cryptocurrency trading in Neurotrade. The model handles signal classification, confidence estimation, trade sizing, reasoning, reward learning, and exposes a rich metrics interface for visual inspection and strategy benchmarking.

---

## 📅 Input Schema

### ➤ Type: `torch.Tensor`
### ➤ Shape: `[batch_size, input_dim]`

### ➤ Description:

A flattened and normalized vector composed of real-time market information (tick + OHLCV), technical indicators, portfolio statistics, time-based signals, and engineered features. All features are aligned by name and order.

### 🔑 Feature Keys (Examples)

| Feature Name               | Description                                      | Data Type | Example     |
|----------------------------|--------------------------------------------------|-----------|-------------|
| `adx_scaled`               | Scaled ADX value to measure trend strength       | float     | 0.14        |
| `atr_pct`                  | ATR as a % of price                              | float     | 0.00026     |
| `band_position`            | Position within Bollinger bands                  | float     | 0.50        |
| `drawdown_pct`             | Max drawdown as % of peak equity                 | float     | 0.02        |
| `entry_price_diff_pct`     | % diff from entry to current price               | float     | -0.001      |
| `has_position`             | Boolean: 1 if position open                      | float     | 1.0         |
| `ind_rsi`, `ind_macd`      | Raw indicators                                   | float     | varies      |
| `inventory_ratio`          | % of portfolio allocated                         | float     | 0.3         |
| `momentum_pct`             | Recent price momentum                            | float     | 0.0012      |
| `normalized_cash`          | Normalized available cash                        | float     | 0.85        |
| `rsi_scaled`               | RSI rescaled to [-1, 1]                          | float     | -0.12       |
| `reward_profit`, `reward_risk` | Components of reward function              | float     | varies      |
| `unrealized_pnl_pct`       | Current PnL in %                                 | float     | 0.004       |
| `tick_arrival_gap`         | Time (ms) between latest ticks                   | float     | 60          |
| `tick_price_change`        | Tick-to-tick price difference                    | float     | 0.0004      |
| `bid_ask_spread_pct`       | Spread between top bid/ask as % of price         | float     | 0.02        |
| `hour_sin` / `hour_cos`    | Sin/cos encoding of time of day                  | float     | 0.866 / 0.5 |
| `day_of_week_sin` / `day_of_week_cos`| Cyclical encoding of weekday         | float     | varies      |
| `regime_volatility_level`  | Regime flag for high/low volatility              | float     | 1.0         |

---

## 📄 Output Heads

| Head Name               | Shape              | Activation | Description                                                  |
|-------------------------|--------------------|------------|--------------------------------------------------------------|
| `signal_logits`         | `[batch, 3]`        | softmax    | Probabilities for `BUY`, `SELL`, `HOLD`                     |
| `confidence`            | `[batch, 1]`        | sigmoid    | Confidence in the chosen action                            |
| `quantity`              | `[batch, 4]`        | softmax    | Trade size bucket (e.g., [0.1x, 0.25x, 0.5x, 1x])           |
| `reward_weights`        | `[batch, 4]`        | softmax    | Distribution over reward components                        |
| `reason_weights`        | `[batch, input_dim]`| tanh       | Feature-level attribution weights                          |
| `execution_mode`        | `[batch, 3]`        | softmax    | Execution route suggestion (MARKET, LIMIT, CANCEL)         |
| `cooldown_timer`        | `[batch, 1]`        | relu       | Wait time before next action (seconds)                     |
| `stop_loss_pct`         | `[batch, 1]`        | sigmoid    | Recommended stop-loss distance                             |
| `take_profit_pct`       | `[batch, 1]`        | sigmoid    | Recommended take-profit distance                           |
| `expected_holding_time` | `[batch, 1]`        | relu       | How long to hold the position                              |
| `signal_stability_score`| `[batch, 1]`        | sigmoid    | Indicates how stable/persistent the current signal is      |

---

## 📊 Metrics Block

### Trading Metrics
- `realized_pnl_pct`, `unrealized_pnl_pct`
- `trade_duration_sec`, `slippage_pct`, `volume_used`
- `take_profit_triggered`, `stop_loss_triggered`
- `entry_price`, `exit_price`

### Model Metrics
- `signal_entropy`, `signal_distribution`
- `reason_topk_features`, `reward_weights_raw`
- `model_version`, `strategy_id`, `decision_latency_ms`

### Risk Metrics
- `max_drawdown_pct`, `inventory_ratio`, `risk_to_reward_ratio`
- `volatility_regime`, `spread_pct`, `tick_velocity`

### Diagnostic Flags
- `reward_alignment_score`, `bootstrap_mode_active`, `fallback_confidence`
- `confidence_std_over_time`, `exploration_flag`, `invalid_state_skipped`
- `replay_buffer_size`, `target_sync_counter`

---

## 🔄 Behavior Summary

| Head                  | Purpose                                 |
|-----------------------|-----------------------------------------|
| `signal_logits`       | Core trading decision (BUY/SELL/HOLD)   |
| `confidence`          | Strength of the action decision         |
| `quantity`            | Trade size suggestion                   |
| `reward_weights`      | Attribution of reward signal            |
| `reason_weights`      | Feature explainability                  |
| `execution_mode`      | Order placement type                    |
| `cooldown_timer`      | Helps avoid overtrading                 |
| `stop_loss_pct`       | Risk management suggestion              |
| `take_profit_pct`     | Profit capture suggestion               |
| `expected_holding_time`| Duration of trade expectation          |
| `signal_stability_score`| Filters noisy/volatile decisions       |

---

## 📌 Design Rationale

- Compact and cyclical encodings are used for time and day.
- Model heads allow both action and execution logic to be learned.
- Metrics enable real-time and batch evaluation.

### Why We *Exclude* Yearly or Monthly Seasonality

| Reason | Explanation |
|--------|-------------|
| ❌ Too slow for scalping | Year/month features are macro and non-predictive at micro timescales |
| ❌ Weak crypto seasonality | Crypto lacks strong calendar-based behavior |
| ❌ No generalization | Time-bound features degrade model robustness |
| ✅ RL learns timing implicitly | If weekends or nights underperform, model learns it from reward |

---

## 📀 Model Metadata Stored

- `input_dim`
- `feature_order`
- `quantity_buckets`
- `output_heads`
- `model_version`
- `strategy_id`
- `model_uid`

---

## 📋 Trading Scenario Checklist

| Scenario                             | Covered? | Explanation |
|--------------------------------------|----------|-------------|
| Intra-day cycles                     | ✅       | via `hour_sin`, `hour_cos` |
| Weekly behavior differences          | ✅       | via `day_of_week_sin/cos` |
| Breakout detection                   | ✅       | via `momentum_pct`, `band_position` |
| Mean reversion                       | ✅       | via `rsi_scaled`, `bollinger` |
| Trade size calibration               | ✅       | via `quantity` head |
| Execution routing                    | ✅       | via `execution_mode` |
| Risk-based exit (TP/SL)              | ✅       | via `stop_loss_pct`, `take_profit_pct` |
| Overtrading mitigation               | ✅       | via `cooldown_timer` |
| Signal filtering                     | ✅       | via `signal_stability_score` |
| Trade duration learning              | ✅       | via `expected_holding_time` |

> ✅ Finalized and streamlined for fast, explainable, and highly reactive crypto scalping.
