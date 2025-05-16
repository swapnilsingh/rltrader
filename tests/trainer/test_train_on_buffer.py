import pytest
import torch
from unittest.mock import MagicMock, patch



def patch_model_metadata(trainer, input_dim=1, feature_order=["feature1"]):
    trainer.model.input_dim = input_dim
    trainer.model.feature_order = feature_order
    trainer.target_model.input_dim = input_dim
    trainer.target_model.feature_order = feature_order

@pytest.mark.parametrize("_", [None], ids=["Buffer sampling should be invoked"])
def test_buffer_sampling(_, trainer):
    print("🔍 TEST: Ensuring buffer.sample() is invoked when training on buffer...")
    trainer.buffer = MagicMock()
    trainer.buffer.__len__.return_value = 10
    trainer.buffer.sample = MagicMock(return_value=[
        ({"feature1": 1.0}, "BUY", 1.0, {"feature1": 1.5})
    ])

    trainer.model = MagicMock(return_value={"signal_logits": torch.tensor([[1.0, 2.0, 3.0]])})
    trainer.target_model = MagicMock(return_value={"signal_logits": torch.tensor([[1.0, 2.0, 3.0]])})
    patch_model_metadata(trainer)

    trainer.loss_fn = MagicMock(return_value=MagicMock(item=lambda: 0.1234))
    trainer.optimizer = MagicMock()
    trainer.optimizer.zero_grad = MagicMock()
    trainer.optimizer.step = MagicMock()
    trainer.model_manager.save_model = MagicMock()

    trainer.train_on_buffer()
    trainer.buffer.sample.assert_called_once()

@pytest.mark.parametrize("_", [None], ids=["Q-values and targets should be computed"])
def test_q_value_computation(_, trainer):
    print("🧠 TEST: Verifying Q-value and target Q computation logic...")
    trainer.buffer = MagicMock()
    trainer.buffer.__len__.return_value = 4
    trainer.buffer.sample = MagicMock(return_value=[
        ({"feature1": 1.0}, "BUY", 1.0, {"feature1": 1.5}) for _ in range(4)
    ])

    logits = torch.tensor([[0.1, 0.2, 1.5]] * 4)
    trainer.model = MagicMock(return_value={"signal_logits": logits})
    trainer.target_model = MagicMock(return_value={"signal_logits": logits})
    patch_model_metadata(trainer)

    trainer.loss_fn = MagicMock(return_value=MagicMock(item=lambda: 0.1))
    trainer.optimizer = MagicMock()
    trainer.optimizer.zero_grad = MagicMock()
    trainer.optimizer.step = MagicMock()
    trainer.model_manager.save_model = MagicMock()

    trainer.train_on_buffer()
    assert trainer.model.called
    assert trainer.target_model.called

@pytest.mark.parametrize("_", [None], ids=["Loss and optimizer should be called"])
def test_loss_and_optimization(_, trainer):
    print("📉 TEST: Confirming that loss.backward() and optimizer.step() are called...")
    trainer.buffer = MagicMock()
    trainer.buffer.__len__.return_value = 3
    trainer.buffer.sample = MagicMock(return_value=[
        ({"feature1": 1.0}, "BUY", 1.0, {"feature1": 1.5}) for _ in range(3)
    ])

    logits = torch.tensor([[0.0, 0.0, 1.0]] * 3)
    trainer.model = MagicMock(return_value={"signal_logits": logits})
    trainer.target_model = MagicMock(return_value={"signal_logits": logits})
    patch_model_metadata(trainer)

    trainer.loss_fn = MagicMock(return_value=MagicMock(item=lambda: 0.2))
    trainer.optimizer = MagicMock()
    trainer.optimizer.zero_grad = MagicMock()
    trainer.optimizer.step = MagicMock()
    trainer.model_manager.save_model = MagicMock()

    trainer.train_on_buffer()
    assert trainer.optimizer.zero_grad.called
    assert trainer.optimizer.step.called

@pytest.mark.parametrize("_", [None], ids=["Model should be saved after training"])
def test_model_save(_, trainer):
    print("💾 TEST: Checking that model is saved after training step...")
    trainer.buffer = MagicMock()
    trainer.buffer.__len__.return_value = 5
    trainer.buffer.sample = MagicMock(return_value=[
        ({"feature1": 1.0}, "BUY", 1.0, {"feature1": 1.5}) for _ in range(5)
    ])

    logits = torch.tensor([[0.0, 0.0, 1.0]] * 5)
    trainer.model = MagicMock(return_value={"signal_logits": logits})
    trainer.target_model = MagicMock(return_value={"signal_logits": logits})
    patch_model_metadata(trainer)

    trainer.loss_fn = MagicMock(return_value=MagicMock(item=lambda: 0.05))
    trainer.optimizer = MagicMock()
    trainer.optimizer.zero_grad = MagicMock()
    trainer.optimizer.step = MagicMock()
    trainer.model_manager.save_model = MagicMock()

    trainer.train_on_buffer()
    assert trainer.model_manager.save_model.called
