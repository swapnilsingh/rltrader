
import pytest
from inference.agent.inference_agent import InferenceAgent

def test_inference_agent_init():
    print("🧪 TEST: InferenceAgent initialization")
    print("➡️ EXPECTED: Should initialize with model path and device")
    agent = InferenceAgent(model_path='models/model.pt', device='cpu')
    print(f"✅ ACTUAL: Initialized with device = {agent.device}")
    assert agent is not None
