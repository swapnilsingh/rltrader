import os
import yaml

def load_config(env="default", path="config.yaml"):
    """
    Load configuration from a single YAML file and merge `default` with `env` section.

    Args:
        env (str): Environment key to merge with default (e.g. "live" or "dry")
        path (str): Path to the config.yaml file

    Returns:
        dict: Final config dictionary with merged settings
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    default_cfg = raw.get("default", {})
    env_cfg = raw.get(env, {})
    return {**default_cfg, **env_cfg}
