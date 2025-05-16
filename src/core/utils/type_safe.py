import torch
import numpy as np

def safe_float(val, default=0.0):
    """
    Convert tensor/numpy/scalar to float safely.
    Returns default if conversion fails or value is None.
    """
    if val is None:
        return default
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu()
        return val.item() if val.numel() == 1 else float(val.mean().item())
    if isinstance(val, np.ndarray):
        return val.item() if val.size == 1 else float(np.mean(val))
    if isinstance(val, (int, float, np.number)):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_list(val):
    """
    Convert tensor/numpy/list to a plain Python list.
    """
    if val is None:
        return []
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().tolist()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return [safe_float(v) for v in val]
    return [safe_float(val)]
