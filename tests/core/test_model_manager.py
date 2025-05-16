import torch
from core.models.dynamic_qnetwork import DynamicQNetwork
from core.utils.model_manager import ModelManager
from core.utils.yaml_loader import load_yaml
from pathlib import Path

def test_model_save_and_load(tmp_path):
    config_path = Path("configs/trainer/model_config.yaml")
    assert config_path.exists(), f"❌ model_config.yaml not found at {config_path}"

    config = load_yaml(config_path)
    config["input_dim"] = 10
    config["feature_order"] = [f"f{i}" for i in range(10)]
    config["output_heads"] = {
        "signal_logits": {"shape": (3,), "activation": "linear"}
    }

    model = DynamicQNetwork(config=config)
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)

    loaded = DynamicQNetwork(config=config)
    loaded.load_state_dict(torch.load(path)["model_state_dict"])
