import os
import uuid
import torch
import copy

class ModelManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model, model_name="model.pt", metadata=None):
        path = os.path.join(self.model_dir, model_name)
        os.makedirs(self.model_dir, exist_ok=True)

        # 🧠 Ensure model is built before extracting heads
        if not hasattr(model, "output_heads"):
            dummy_input = torch.randn(1, getattr(model, "input_dim", 32))
            model(dummy_input)

        # 📦 Load previous metadata if model exists
        previous_metadata = {}
        if os.path.exists(path):
            try:
                _, previous_metadata = self.load_model(model, model_name)
            except Exception:
                pass  # safe fallback

        # 🧬 Merge with new metadata
        metadata = copy.deepcopy(metadata or {})
        merged_metadata = {
            **previous_metadata,
            **metadata,
            "input_dim": getattr(model, "input_dim", None),
            "feature_order": getattr(model, "feature_order", []),
            "output_heads": getattr(model, "output_heads_config", {}),  # ✅ Save full head config
            "model_version": metadata.get("model_version", "v1.0"),
            "strategy_id": metadata.get("strategy_id", "default"),
            "model_uid": previous_metadata.get("model_uid") or metadata.get("model_uid"),
        }

        payload = {
            "model_state_dict": model.state_dict(),
            "config": merged_metadata
        }

        torch.save(payload, path)
        print(f"✅ Model saved to {path} with metadata keys: {list(merged_metadata.keys())}")



    def load_model(self, model, model_name="model.pt"):
        
        path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")

        payload = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(payload["model_state_dict"])

        metadata = payload.get("config", {})

        # 🔁 Restore model metadata attributes
        if hasattr(model, "metadata"):
            model.metadata = metadata

        # 🔁 Restore output_heads_config
        if "output_heads" in metadata:
            model.output_heads_config = metadata["output_heads"]

        print(f"📦 Model loaded from {path} with metadata: {list(metadata.keys())}")
        return model, metadata


    def model_exists(self, model_name="model.pt"):
        return os.path.exists(os.path.join(self.model_dir, model_name))
