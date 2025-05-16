from core.decorators.decorators import inject_logger
from core.models.dynamic_qnetwork import DynamicQNetwork
from core.utils.type_safe import safe_float, safe_list

import torch
import numpy as np

@inject_logger()
class InferenceAgent:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint["config"]
        self.feature_order = config["feature_order"]
        self.output_heads = config["output_heads"]

        self.model = DynamicQNetwork(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval().to(device)

    def preprocess(self, state_dict):
        return np.array([safe_float(state_dict.get(k)) for k in self.feature_order], dtype=np.float32)

    def infer(self, state_vector):
        with torch.no_grad():
            x = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            raw_output = self.model(x)

            cleaned_output = {
                key: safe_list(value.squeeze().detach().cpu().numpy())
                for key, value in raw_output.items()
            }

            return cleaned_output.get("signal_logits"), cleaned_output

    def predict(self, state_vector):
        _, cleaned_output = self.infer(state_vector)
        return cleaned_output


