import pytest
from trainer.agent.replay_buffer import ReplayBuffer

def create_sample(i):
    return ({"f0": i}, "BUY", 1.0, {"f0": i + 1})

def test_static_buffer_behavior():
    buffer = ReplayBuffer(max_size=3, strategy="static")
    for i in range(5):
        buffer.add(create_sample(i))
    assert len(buffer) == 3
    samples = buffer.sample(2)
    assert len(samples) == 2
    assert all(isinstance(e, tuple) for e in samples)

def test_train_once_behavior():
    buffer = ReplayBuffer(strategy="train_once")
    for i in range(5):
        buffer.add(create_sample(i))
    assert len(buffer) == 5
    samples = buffer.sample(3)
    assert len(samples) == 3
    assert len(buffer) == 2  # Remaining untrained
    samples2 = buffer.sample(5)
    assert len(samples2) == 2
    assert len(buffer) == 0

def test_dynamic_buffer_resize():
    buffer = ReplayBuffer(max_size=5, strategy="dynamic")
    for i in range(5):
        buffer.add(create_sample(i))
    assert len(buffer) == 5
    buffer.resize(3)
    assert len(buffer) == 3
    # Check that only the last 3 items are retained
    retained = list(buffer.buffer)
    assert retained[0][0]["f0"] == 2
    assert retained[-1][0]["f0"] == 4

def test_buffer_clear_method():
    buffer = ReplayBuffer(max_size=5, strategy="static")
    for i in range(5):
        buffer.add(create_sample(i))
    assert len(buffer) == 5
    buffer.clear()
    assert len(buffer) == 0

def test_dynamic_resize_behavior_expand():
    buffer = ReplayBuffer(max_size=5, strategy="dynamic")
    for i in range(5):
        buffer.add(({"f0": i}, "BUY", 1.0, {"f0": i+1}))

    assert len(buffer) == 5
    assert buffer.max_size == 5

    buffer.resize(10)
    assert len(buffer) == 5
    assert buffer.max_size == 10
    for i in range(5, 10):
        buffer.add(({"f0": i}, "SELL", -1.0, {"f0": i+1}))
    assert len(buffer) == 10

def test_dynamic_resize_behavior_shrink():
    buffer = ReplayBuffer(max_size=10, strategy="dynamic")
    for i in range(10):
        buffer.add(({"f0": i}, "BUY", 1.0, {"f0": i+1}))

    assert len(buffer) == 10

    buffer.resize(4)
    assert len(buffer) == 4
    assert buffer.max_size == 4
    retained = list(buffer.buffer)
    assert retained[0][0]["f0"] == 6
    assert retained[-1][0]["f0"] == 9

