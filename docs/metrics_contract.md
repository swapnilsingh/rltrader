# Reward Metrics Contract for LLM Agents

Each record in `reward_metrics:<symbol>` must include:

- `timestamp`: Unix timestamp
- `symbol`: e.g., "btcusdt"
- `action`: "BUY" / "SELL" / "HOLD"
- `confidence`: Model confidence score
- `stability`: Signal volatility
- `quantity`: Executed or proposed quantity
- `current_price`: Market price at decision time
- `equity`: Portfolio value at time
- `drawdown_pct`: % from peak equity
- `realized_pnl`: Net gain/loss from closed trades
- `unrealized_pnl`: PnL from open position
- `holding_time`: Duration trade was held
- `reason`: Optional exit reason

This contract is read by the `ContextBuilder` → `LLMRewardController` → `RewardAgent`.
