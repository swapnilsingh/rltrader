# src/core/models/metrics.py

from pydantic import BaseModel
from typing import Optional

class RewardMetrics(BaseModel):
    timestamp: int
    symbol: str
    action: str
    confidence: float
    stability: float
    quantity: float
    current_price: float
    equity: float
    drawdown_pct: float
    realized_pnl: float
    unrealized_pnl: float
    holding_time: Optional[float] = None
    reason: Optional[str] = None
