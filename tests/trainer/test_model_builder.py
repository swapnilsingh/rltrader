import pytest
import torch
from core.models.dynamic_qnetwork import DynamicQNetwork
from core.utils.yaml_loader import load_yaml


@pytest.fixture(scope="module")
def model_and_config(model_config_path):
    config = load_yaml(model_config_path)
    model = DynamicQNetwork(config, profile="balanced")
    return model, config


class TestDynamicQNetwork:

    def test_model_input_dim_matches_feature_order(self, model_and_config):
        model, config = model_and_config
        expected_input_dim = len(config["feature_order"])
        actual_input_dim = model.input_dim

        assert actual_input_dim == expected_input_dim, (
            f"❌ [Input Dim] Mismatch: expected {expected_input_dim}, got {actual_input_dim}"
        )
        print(f"✅ [Input Dim] {actual_input_dim} matches feature_order")

    def test_model_output_heads_match_config(self, model_and_config):
        model, config = model_and_config
        expected_heads = set(config["output_heads"].keys())
        actual_heads = set(model.heads.keys())

        assert actual_heads == expected_heads, (
            f"❌ [Output Heads] Mismatch:\n"
            f"  Missing: {expected_heads - actual_heads}\n"
            f"  Unexpected: {actual_heads - expected_heads}"
        )
        print(f"✅ [Output Heads] Match: {sorted(actual_heads)}")

    def test_forward_output_shapes_match(self, model_and_config):
        model, config = model_and_config
        batch_size = 4
        dummy_input = torch.randn(batch_size, model.input_dim)
        outputs = model(dummy_input)

        assert isinstance(outputs, dict), f"❌ Output is not a dict. Got: {type(outputs)}"
        assert len(outputs) == len(config["output_heads"]), (
            f"❌ [Output Count] Expected {len(config['output_heads'])}, got {len(outputs)}"
        )

        for head_name, output in outputs.items():
            expected_dim = (
                model.input_dim
                if config["output_heads"][head_name].get("shape_from") == "input_dim"
                else config["output_heads"][head_name]["shape"][0]
            )
            expected_shape = (batch_size, expected_dim)
            actual_shape = tuple(output.shape)

            assert actual_shape == expected_shape, (
                f"❌ [Shape Mismatch] {head_name}: expected {expected_shape}, got {actual_shape}"
            )

        print("✅ [Forward] All output shapes match expected dimensions.")

    def test_forward_fails_with_wrong_input_shape(self, model_and_config):
        model, _ = model_and_config
        bad_input = torch.randn(4, model.input_dim + 1)

        with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
            model(bad_input)

        print("✅ [Edge Case] Forward fails with incorrect input shape.")

    def test_model_raises_on_invalid_profile(self, model_config_path):
        config = load_yaml(model_config_path)
        invalid_profile = "nonexistent_profile"

        with pytest.raises(ValueError, match="Invalid profile"):
            DynamicQNetwork(config, profile=invalid_profile)

        print("✅ [Profile] Invalid profile raises ValueError.")

    def test_missing_shape_and_shape_from_raises(self, model_config_path):
        config = load_yaml(model_config_path)
        config["output_heads"]["bad_head"] = {
            "type": "classification",
            "activation": "softmax"
        }

        with pytest.raises(KeyError, match="must define either 'shape' or 'shape_from'"):
            DynamicQNetwork(config, profile="balanced")

        print("✅ [Edge Case] Missing shape/shape_from raises KeyError.")

    def test_invalid_activation_raises(self, model_config_path):
        config = load_yaml(model_config_path)
        config["output_heads"]["signal_logits"]["activation"] = "invalid_fn"

        with pytest.raises(KeyError, match="invalid_fn"):
            DynamicQNetwork(config, profile="balanced")

        print("✅ [Edge Case] Invalid activation raises KeyError.")

    def test_invalid_shape_from_value_raises(self, model_config_path):
        config = load_yaml(model_config_path)
        config["output_heads"]["signal_logits"].pop("shape", None)
        config["output_heads"]["signal_logits"]["shape_from"] = "unknown_source"

        with pytest.raises(ValueError, match="Unsupported shape_from"):
            DynamicQNetwork(config, profile="balanced")

        print("✅ [Edge Case] Unsupported shape_from raises ValueError.")
