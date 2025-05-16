import pytest
import torch
from unittest.mock import MagicMock, patch

def patch_model_metadata(trainer, input_dim=1, feature_order=["feature1"]):
    trainer.model.input_dim = input_dim
    trainer.model.feature_order = feature_order
    trainer.target_model.input_dim = input_dim
    trainer.target_model.feature_order = feature_order

def test_train_with_empty_buffer(trainer):
    print("🧪 TEST: Should skip training if experience buffer is empty")
    trainer.buffer = MagicMock()
    trainer.buffer.__len__.return_value = 0
    trainer.train_on_buffer()

def test_train_with_invalid_action(trainer):
    print("🧪 TEST: Should raise KeyError if action string is invalid")
    trainer.buffer = MagicMock()
    trainer.buffer.__len__.return_value = 1
    trainer.buffer.sample = MagicMock(return_value=[
        ({"feature1": 1.0}, "UNKNOWN_ACTION", 1.0, {"feature1": 1.1})
    ])
    trainer.model = MagicMock(return_value=torch.tensor([[1.0, 2.0, 3.0]]))
    trainer.target_model = MagicMock(return_value=torch.tensor([[1.0, 2.0, 3.0]]))
    patch_model_metadata(trainer)

    trainer.loss_fn = MagicMock(return_value=MagicMock(item=lambda: 0.1234))
    trainer.optimizer = MagicMock()
    trainer.optimizer.zero_grad = MagicMock()
    trainer.optimizer.step = MagicMock()
    trainer.model_manager.save_model = MagicMock()

    with pytest.raises(KeyError):
        trainer.train_on_buffer()

def test_train_with_dim_mismatch(trainer):
    print("🧪 TEST: Should raise ValueError on input dimension mismatch")
    trainer.buffer = MagicMock()
    trainer.buffer.__len__.return_value = 1
    trainer.buffer.sample = MagicMock(return_value=[
        ({"feature1": 1.0, "feature2": 2.0}, "BUY", 1.0, {"feature1": 1.1, "feature2": 2.1})
    ])
    trainer.model = MagicMock(return_value=torch.tensor([[1.0, 2.0, 3.0]]))
    trainer.target_model = MagicMock(return_value=torch.tensor([[1.0, 2.0, 3.0]]))

    # ⛔ input_dim is 1, but 2 features will be extracted → should raise
    trainer.model.input_dim = 1
    trainer.model.feature_order = ["feature1", "feature2"]
    trainer.target_model.input_dim = 1
    trainer.target_model.feature_order = ["feature1", "feature2"]

    trainer.loss_fn = MagicMock(return_value=MagicMock(item=lambda: 0.1))
    trainer.optimizer = MagicMock()
    trainer.optimizer.zero_grad = MagicMock()
    trainer.optimizer.step = MagicMock()
    trainer.model_manager.save_model = MagicMock()

    with pytest.raises(ValueError, match="Invalid input dim"):
        trainer.train_on_buffer()


def test_train_with_nan_state(trainer):
    print("🧪 TEST: Should sanitize NaN state and continue training")

    trainer.buffer = MagicMock()
    trainer.buffer.__len__.return_value = 1
    trainer.buffer.sample = MagicMock(return_value=[
        ({"feature1": float('nan')}, "BUY", 1.0, {"feature1": 1.5})
    ])

    logits = torch.tensor([[0.1, 0.2, 1.0]])
    trainer.model = MagicMock(return_value={"signal_logits": logits})
    trainer.model.input_dim = 1
    trainer.model.feature_order = ["feature1"]

    trainer.target_model = MagicMock(return_value={"signal_logits": logits})
    trainer.target_model.input_dim = 1
    trainer.target_model.feature_order = ["feature1"]

    trainer.loss_fn = MagicMock(return_value=MagicMock(item=lambda: 0.1))
    trainer.optimizer = MagicMock()
    trainer.optimizer.zero_grad = MagicMock()
    trainer.optimizer.step = MagicMock()
    trainer.model_manager.save_model = MagicMock()

    trainer.train_on_buffer()
