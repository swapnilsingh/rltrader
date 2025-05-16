import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from inference.publisher.signal_publisher import SignalPublisher
import json


@pytest.fixture
def mock_publisher():
    redis_mock = MagicMock()
    publisher = SignalPublisher(redis_conn=redis_mock, symbol="btcusdt")
    return publisher, redis_mock


def extract_payload(mock_redis):
    args, _ = mock_redis.lpush.call_args
    return json.loads(args[1])


def test_publish_sanitizes_torch_tensor_scalar(mock_publisher):
    print("🧪 TEST: Sanitizing scalar torch.Tensor (confidence)")
    print("➡️ EXPECTED: confidence is float after serialization")
    publisher, redis_mock = mock_publisher
    publisher.publish(action="BUY", confidence=torch.tensor(0.95), reason="scalar tensor test", extra={})
    payload = extract_payload(redis_mock)
    print(f"✅ ACTUAL: confidence = {payload['confidence']}")
    assert isinstance(payload["confidence"], float)


def test_publish_sanitizes_torch_tensor_array(mock_publisher):
    print("🧪 TEST: Sanitizing torch.Tensor array (signal_logits)")
    print("➡️ EXPECTED: signal_logits is list after serialization")
    publisher, redis_mock = mock_publisher
    publisher.publish(action="BUY", confidence=0.9, reason="array tensor test", extra={
        "signal_logits": torch.tensor([0.1, 0.2, 0.7])
    })
    payload = extract_payload(redis_mock)
    print(f"✅ ACTUAL: signal_logits = {payload['signal_logits']}")
    assert isinstance(payload["signal_logits"], list)


def test_publish_sanitizes_numpy_array(mock_publisher):
    print("🧪 TEST: Sanitizing np.ndarray (volume)")
    print("➡️ EXPECTED: volume is list after serialization")
    publisher, redis_mock = mock_publisher
    publisher.publish(action="BUY", confidence=0.9, reason="numpy array test", extra={
        "volume": np.array([0.5])
    })
    payload = extract_payload(redis_mock)
    print(f"✅ ACTUAL: volume = {payload['volume']}")
    assert isinstance(payload["volume"], list)


def test_publish_sanitizes_numpy_scalar(mock_publisher):
    print("🧪 TEST: Sanitizing np.generic scalar (reward_score)")
    print("➡️ EXPECTED: reward_score is float after serialization")
    publisher, redis_mock = mock_publisher
    publisher.publish(action="BUY", confidence=0.9, reason="numpy scalar test", extra={
        "reward_score": np.float32(0.8)
    })
    payload = extract_payload(redis_mock)
    print(f"✅ ACTUAL: reward_score = {payload['reward_score']}")
    assert isinstance(payload["reward_score"], float)


def test_publish_sanitizes_nested_tensor_structure(mock_publisher):
    print("🧪 TEST: Sanitizing nested dict with tensor and ndarray")
    print("➡️ EXPECTED: meta.tensor and meta.array are float")
    publisher, redis_mock = mock_publisher
    publisher.publish(action="BUY", confidence=0.9, reason="nested tensor", extra={
        "meta": {
            "tensor": torch.tensor(1.23),
            "array": np.array(1.1)
        }
    })
    payload = extract_payload(redis_mock)
    print(f"✅ ACTUAL: tensor = {payload['meta']['tensor']}, array = {payload['meta']['array']}")
    assert isinstance(payload["meta"]["tensor"], float)
    assert isinstance(payload["meta"]["array"], float)
