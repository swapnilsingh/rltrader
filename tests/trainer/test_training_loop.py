import pytest
import json
from unittest.mock import patch, MagicMock
from trainer.agent.dqn_trainer import DQNTrainer
import core.decorators.decorators as decorators_module

@patch("redis.Redis")
def test_run_training_loop_with_mock_redis(mock_redis_class, mock_config):
    mock_redis = MagicMock()
    mock_redis_class.return_value = mock_redis

    exp = ({"f0": 1}, "BUY", 1.0, {"f0": 2})
    mock_redis.lpop.side_effect = [json.dumps(exp), json.dumps(exp), None, KeyboardInterrupt]

    # 🚫 Remove decorator behavior during test
    with patch.object(decorators_module, "load_config", return_value=mock_config):
        with patch("core.utils.config_loader.load_config", return_value=mock_config):
            trainer = DQNTrainer(config=mock_config)
            trainer.train_on_buffer = MagicMock()
            trainer.run_training_loop()

    assert trainer.train_on_buffer.call_count >= 1
    assert len(trainer.buffer) <= trainer.buffer.max_size
