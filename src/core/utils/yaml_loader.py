import yaml
import os

def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ YAML file not found: {path}")
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"❌ Failed to parse YAML file {path}: {e}")
