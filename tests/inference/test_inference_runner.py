
import pytest
from inference.inference_runner import InferenceRunner

def test_inference_runner_init():
    print("🧪 TEST: InferenceRunner initialization")
    print("➡️ EXPECTED: Should load config and instantiate dependencies")
    runner = InferenceRunner()
    print(f"✅ ACTUAL: Initialized for symbol = {runner.symbol} on device = {runner.device}")
    assert runner is not None
