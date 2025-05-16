
import pytest
import numpy as np
from unittest.mock import MagicMock

from inference.agent.inference_agent import InferenceAgent
from inference.preprocessor.ta_tick_state_builder import TATickStateBuilder
from core.portfolio.wallet import Wallet
from inference.publisher.signal_publisher import SignalPublisher

@pytest.fixture
def mock_agent():
    agent = InferenceAgent(model_path='models/model.pt', device='cpu')
    agent.infer_signal = lambda state: {
        "signal_logits": np.array([0.1, 0.2, 0.7]),
        "volume": np.array([0.5]),
        "reason_weights": np.array([0.3, 0.3, 0.4]),
        "reward_weights": np.array([1.0, 0.0, 0.0]),
        "bootstrap_mode": False,
        "metadata": {"mock": True}
    }
    return agent

def test_end_to_end_inference_pipeline(mock_agent):
    print("🧪 TEST: End-to-end inference pipeline")
    print("➡️ EXPECTED: Build state → infer → publish without errors")

    builder = TATickStateBuilder(feature_order=[
        "tick_price_change", "adx_scaled", "atr_pct", "band_position", "drawdown_pct",
        "ind_rsi", "ind_macd", "momentum_pct", "tick_arrival_gap", "hour_sin", "hour_cos",
        "day_of_week_sin", "day_of_week_cos", "regime_volatility_level",
        "has_position", "entry_price_diff_pct", "inventory_ratio", "normalized_cash",
        "reward_profit", "reward_risk", "unrealized_pnl_pct"
    ])

    wallet = Wallet(starting_balance=1000.0)
    for _ in range(10):
        builder.update_tick({'p': '100', 'q': '0.01', 'T': 1715678123456})

    state = builder.build_state(wallet.get_state_dict(100.0))
    print(f"✅ ACTUAL: Generated state with {len(state)} features")

    signal = mock_agent.infer_signal(state)
    assert "signal_logits" in signal
    print(f"✅ ACTUAL: Inferred signal with logits = {signal['signal_logits']}")

    mock_redis = MagicMock()
    publisher = SignalPublisher(redis_conn=mock_redis)
    publisher.publish(
        action="BUY",
        confidence=0.95,
        reason="mocked test",
        extra=signal  # ✅ pass the full signal dict as 'extra', which will be sanitized
    )
    # will be a no-op with mock
    print("✅ ACTUAL: Signal published (mocked Redis)")
