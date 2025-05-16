import pytest
import shutil
from pathlib import Path
from trainer.agent.dqn_trainer import DQNTrainer
from core.utils.yaml_loader import load_yaml

@pytest.fixture(scope="session")
def model_config_path(tmp_path_factory):
    src = Path("configs/trainer/model_config.yaml")  # ✅ correct relative path
    dst = tmp_path_factory.mktemp("configs") / "model_config.yaml"
    shutil.copy(src, dst)
    return str(dst)

@pytest.fixture
def trainer(tmp_path, model_config_path):
    config = {
        "symbol": "BTCUSDT",
        "gamma": 0.99,
        "batch_size": 32,
        "lr": 0.001,
        "bootstrap_candles": 100,
        "action_index_map": {"SELL": 0, "HOLD": 1, "BUY": 2},
        "action_value_map": {"SELL": -1, "HOLD": 0, "BUY": 1},
        "model_dir": str(tmp_path),
        "model_profile": "balanced",
        "model_config_path": model_config_path,
        "buffer_strategy": "static",
        "buffer_size": 500
    }

    # ✅ Inject pre-parsed model config
    config["model_config"] = load_yaml(model_config_path)

    return DQNTrainer(config=config)

@pytest.fixture
def mock_config(tmp_path, model_config_path):
    config = {
        "symbol": "BTCUSDT",
        "batch_size": 2,
        "buffer_strategy": "dynamic",
        "buffer_size": 10,
        "model_dir": str(tmp_path),
        "model_profile": "balanced",
        "model_config_path": model_config_path,
        "redis": {
            "host": "localhost",
            "port": 6379,
            "experience_key": "experience_queue"
        },
        "train_interval": 0.1
    }

    # ✅ Inject pre-parsed model config
    config["model_config"] = load_yaml(model_config_path)

    return config
