from dataclasses import dataclass

@dataclass
class TradeOutcome:
    realized_pnl_pct: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    holding_time: float = 0.0
    drawdown_pct: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    action: str = ""
    timestamp: int = 0
    slippage_pct: float = 0.0
    volatility_pct: float = 0.0
    spread_volatility: float = 0.0
    orderbook_imbalance: float = 0.0
    reason: str = ""
    was_executed: bool = True

    @classmethod
    def from_wallets(cls, prev_wallet: dict, current_wallet: dict, price: float, action: str = "HOLD", timestamp: int = 0, executed: bool = True):
        try:
            entry_price = prev_wallet.get("entry_price", 0.0)
            exit_price = price
            realized_pnl = current_wallet.get("realized_pnl", 0.0) - prev_wallet.get("realized_pnl", 0.0)
            unrealized_pnl = current_wallet.get("unrealized_pnl", 0.0)
            equity = current_wallet.get("equity", 0.0)
            prev_equity = prev_wallet.get("equity", 0.0)
            max_equity = max(prev_equity, equity)

            realized_pnl_pct = (realized_pnl / prev_equity) if prev_equity > 0 else 0.0
            drawdown_pct = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
            holding_time = current_wallet.get("holding_time", 0.0)
            quantity = current_wallet.get("inventory", 0.0)

            # New volatility-related metrics
            slippage_pct = abs(price - entry_price) / price if price > 0 else 0.0
            volatility_pct = abs(equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            spread_volatility = current_wallet.get("spread_volatility", 0.0)
            orderbook_imbalance = current_wallet.get("orderbook_imbalance", 0.0)

            return cls(
                realized_pnl_pct=realized_pnl_pct,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                drawdown_pct=drawdown_pct,
                holding_time=holding_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                action=action,
                timestamp=timestamp,
                slippage_pct=slippage_pct,
                volatility_pct=volatility_pct,
                spread_volatility=spread_volatility,
                orderbook_imbalance=orderbook_imbalance,
                was_executed=executed
            )

        except Exception as e:
            return cls(
                realized_pnl_pct=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                drawdown_pct=0.0,
                holding_time=0.0,
                entry_price=0.0,
                exit_price=price,
                quantity=0.0,
                action=action,
                timestamp=timestamp,
                slippage_pct=0.0,
                volatility_pct=0.0,
                spread_volatility=0.0,
                orderbook_imbalance=0.0,
                was_executed=False,
                reason=f"from_wallets_error: {e}"
            )
