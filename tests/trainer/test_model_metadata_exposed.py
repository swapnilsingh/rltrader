import pytest
import torch
from unittest.mock import MagicMock, patch



@pytest.mark.parametrize("description, setup_fn, assert_fn", [
    (
        "Buffer sampling should be invoked",
        lambda t: True,
        lambda t: t.buffer.sample.assert_called_once()
    ),
    (
        "Q-values and target Q-values should be computed",
        lambda t: True,
        lambda t: (
            t.model.assert_called(),
            t.target_model.assert_called()
        )
    ),
    (
        "Loss and optimizer should be used",
        lambda t: True,
        lambda t: (
            t.optimizer.zero_grad.assert_called_once(),
            t.optimizer.step.assert_called_once()
        )
    ),
    (
        "Model should be saved after training",
        lambda t: True,
        lambda t: t.model_manager.save_model.assert_called_once()
    ),
], ids=lambda val: val if isinstance(val, str) else None)
def test_train_on_buffer_logic(trainer, description, setup_fn, assert_fn):
    print(f"🧪 TEST: {description}")

    trainer.buffer = MagicMock()
    trainer.buffer.__len__.return_value = 5
    trainer.buffer.sample = MagicMock(return_value=[
        ({"feature1": 1.0}, "BUY", 1.0, {"feature1": 1.5}) for _ in range(5)
    ])

    logits = torch.tensor([[0.1, 0.2, 0.7]] * 5)

    # Use side_effect to simulate callable mock returning dict
    trainer.model = MagicMock(side_effect=lambda x: {"signal_logits": logits})
    trainer.model.input_dim = 1
    trainer.model.feature_order = ["feature1"]

    trainer.target_model = MagicMock(side_effect=lambda x: {"signal_logits": logits})
    trainer.target_model.input_dim = 1
    trainer.target_model.feature_order = ["feature1"]

    trainer.loss_fn = MagicMock(return_value=MagicMock(item=lambda: 0.05))
    trainer.optimizer = MagicMock()
    trainer.optimizer.zero_grad = MagicMock()
    trainer.optimizer.step = MagicMock()
    trainer.model_manager = MagicMock()
    trainer.model_manager.save_model = MagicMock()

    setup_fn(trainer)
    trainer.train_on_buffer()
    assert_fn(trainer)
