
import pytest
from unittest.mock import MagicMock
import numpy as np

@pytest.fixture
def dummy_tick():
    return {
        "p": "30000.5",
        "q": "0.01",
        "T": 1715678123456
    }

@pytest.fixture
def dummy_model_output():
    return {
        "signal_logits": np.array([0.1, 0.8, 0.1]),
        "volume": np.array([0.5]),
        "reason_weights": np.array([0.2, 0.5, 0.3]),
        "reward_weights": np.array([1.0, 0.0, 0.0])
    }

@pytest.fixture
def mock_redis():
    return MagicMock()

@pytest.fixture
def mock_config():
    return {
        "symbol": "btcusdt",
        "device": "cpu",
        "model": {"path": "models/model.pt"},
        "redis": {"host": "localhost", "port": 6379}
    }
