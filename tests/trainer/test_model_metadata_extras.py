import pytest
from core.models.dynamic_qnetwork import DynamicQNetwork
from core.utils.yaml_loader import load_yaml


class TestDynamicQNetworkEdgeCases:

    def test_feature_order_missing_raises(self, model_config_path):
        config = load_yaml(model_config_path)
        config.pop("feature_order", None)

        with pytest.raises(ValueError, match="feature_order"):
            DynamicQNetwork(config)

        print("✅ [Edge Case] Missing 'feature_order' raises ValueError.")

    def test_output_head_missing_shape_raises(self, model_config_path):
        config = load_yaml(model_config_path)
        head = config["output_heads"]["signal_logits"]
        head.pop("shape", None)
        head.pop("shape_from", None)

        with pytest.raises(KeyError, match="must define either 'shape' or 'shape_from'"):
            DynamicQNetwork(config)

        print("✅ [Edge Case] Missing shape and shape_from raises KeyError.")

    def test_invalid_shape_from_raises(self, model_config_path):
        config = load_yaml(model_config_path)
        config["output_heads"]["signal_logits"].pop("shape", None)
        config["output_heads"]["signal_logits"]["shape_from"] = "invalid_key"

        with pytest.raises(ValueError, match="Unsupported shape_from"):
            DynamicQNetwork(config)

        print("✅ [Edge Case] Unsupported 'shape_from' raises ValueError.")

    def test_invalid_activation_raises(self, model_config_path):
        config = load_yaml(model_config_path)
        config["output_heads"]["signal_logits"]["activation"] = "invalid_activation"

        with pytest.raises(KeyError, match="invalid_activation"):
            DynamicQNetwork(config)

        print("✅ [Edge Case] Invalid activation function raises KeyError.")
