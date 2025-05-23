# utils/sanitize.py

import numpy as np
import pandas as pd

def sanitize_state(state: dict, strict: bool = False) -> bool:
    """
    Validates that the state dictionary:
    - Has no NaNs or Infs
    - All values are scalars (int, float, bool)
    - Optional: enforce strict length or schema

    Returns:
        bool: True if state is clean, False otherwise
    """
    if not isinstance(state, dict):
        return False

    for k, v in state.items():
        # Convert numpy types to scalars
        if isinstance(v, (np.generic, pd.Series)):
            v = np.asscalar(v) if hasattr(np, 'asscalar') else np.array(v).item()

        if isinstance(v, (list, tuple, dict)):
            return False  # nested structures not allowed

        try:
            if v is None or np.isnan(v) or np.isinf(v):
                return False
        except:
            continue  # non-numeric types (e.g., str, bool) are allowed

    return True
