import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicQNetwork(nn.Module):
    def __init__(self, config, profile="balanced"):
        super().__init__()

        self.config = config

        # Extract and validate feature order
        self.feature_order = config.get("feature_order")
        if not self.feature_order or not isinstance(self.feature_order, list):
            raise ValueError("❌ 'feature_order' must be a non-empty list in model_config.yaml")

        self.input_dim = len(self.feature_order)

        # Load architecture profile
        profiles = config["architecture"]["profiles"]
        if profile not in profiles:
            raise ValueError(f"❌ Invalid profile '{profile}'. Available profiles: {list(profiles.keys())}")

        selected_profile = profiles[profile]
        hidden_layers = selected_profile["hidden_layers"]
        activation_fn = self._get_activation(selected_profile["activation"])
        dropout = selected_profile.get("dropout", 0.0)

        # Build shared backbone
        layers = []
        prev_dim = self.input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.backbone = nn.Sequential(*layers)

        # Build dynamic output heads
        self.heads = nn.ModuleDict()
        for head_name, head_cfg in config["output_heads"].items():
            if "shape_from" in head_cfg:
                if head_cfg["shape_from"] == "input_dim":
                    out_dim = self.input_dim
                else:
                    raise ValueError(f"Unsupported shape_from: {head_cfg['shape_from']} in head '{head_name}'")
            elif "shape" in head_cfg:
                out_dim = head_cfg["shape"][0]
            else:
                raise KeyError(f"❌ Output head '{head_name}' must define either 'shape' or 'shape_from'")

            self.heads[head_name] = nn.Sequential(
                nn.Linear(prev_dim, out_dim),
                self._get_activation(head_cfg["activation"])()
            )

    def forward(self, x):
        features = self.backbone(x)
        outputs = {name: head(features) for name, head in self.heads.items()}
        return outputs

    def _get_activation(self, name):
        return {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "softmax": lambda: nn.Softmax(dim=-1),
            "tanh": nn.Tanh,
            "linear": nn.Identity, 
        }[name]
