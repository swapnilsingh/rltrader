from datetime import datetime
from core.decorators.decorators import inject_logger

@inject_logger()
class Wallet:
    def __init__(self, starting_balance=1000.0):
        self.initial_balance = starting_balance
        self.balance = starting_balance
        self.inventory = 0.0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.trade_history = []
        self.equity_curve = []
        self.max_equity = starting_balance
        self.min_equity = starting_balance
        self.equity_peak = starting_balance

    def buy(self, price: float, quantity: float):
        cost = price * quantity
        if self.balance >= cost:
            if cost < 1.0:
                self.logger.warning(f"⚠️ Low notional BUY: ${cost:.4f}")
            self.balance -= cost
            self.inventory += quantity
            self.entry_price = price  # Model re-enters at new price
            self._log_trade("BUY", price, quantity)
        else:
            raise ValueError("❌ Insufficient balance to buy.")

    def sell(self, price: float, quantity: float):
        if self.inventory >= quantity:
            proceeds = price * quantity
            pnl = (price - self.entry_price) * quantity
            self.balance += proceeds
            self.realized_pnl += pnl
            self.inventory -= quantity
            if self.inventory == 0:
                self.entry_price = 0.0
            self._log_trade("SELL", price, quantity, pnl)
        else:
            raise ValueError("❌ Insufficient inventory to sell.")

    def has_position(self) -> bool:
        return self.inventory > 0

    def get_available_equity(self) -> float:
        return self.balance

    def compute_unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.inventory if self.has_position() else 0.0

    def compute_equity(self, price: float) -> float:
        equity = self.balance + self.inventory * price
        self.equity_peak = max(self.equity_peak, equity)
        self.max_equity = max(self.max_equity, equity)
        self.min_equity = min(self.min_equity, equity)
        self.equity_curve.append({
            "timestamp": datetime.utcnow().isoformat(),
            "equity": equity,
            "balance": self.balance,
            "inventory": self.inventory
        })
        return equity

    def get_state_dict(self, price: float):
        equity = self.compute_equity(price)
        drawdown_pct = (self.equity_peak - equity) / self.equity_peak if self.equity_peak > 0 else 0.0
        return {
            "balance": self.balance,
            "inventory": self.inventory,
            "entry_price": self.entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.compute_unrealized_pnl(price),
            "has_position": int(self.has_position()),
            "equity": equity,
            "drawdown_pct": drawdown_pct
        }

    def get_performance_metrics(self, price: float):
        equity = self.compute_equity(price)
        drawdown_pct = (self.equity_peak - equity) / self.equity_peak if self.equity_peak > 0 else 0.0
        sell_trades = [t for t in self.trade_history if t["action"] == "SELL"]
        win_trades = [t for t in sell_trades if t.get("pnl", 0.0) > 0]
        win_rate = len(win_trades) / len(sell_trades) if sell_trades else 0.0
        avg_pnl = sum(t.get("pnl", 0.0) for t in sell_trades) / len(sell_trades) if sell_trades else 0.0
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.compute_unrealized_pnl(price),
            "drawdown_pct": drawdown_pct,
            "win_rate": win_rate,
            "avg_trade_pnl": avg_pnl,
            "equity": equity,
            "max_equity": self.max_equity,
            "min_equity": self.min_equity,
            "equity_peak": self.equity_peak,
            "trade_count": len(self.trade_history)
        }

    def _log_trade(self, action, price, quantity, pnl=None):
        trade = {
            "action": action,
            "price": price,
            "quantity": quantity,
            "timestamp": datetime.utcnow().isoformat()
        }
        if pnl is not None:
            trade["pnl"] = pnl
        self.logger.info(f"🧾 {action} | Qty: {quantity:.6f} | Price: {price:.2f} | Balance: {self.balance:.2f}")
        self.trade_history.append(trade)
