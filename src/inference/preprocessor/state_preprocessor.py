from core.decorators.decorators import inject_logger
import numpy as np

@inject_logger()
class StatePreprocessor:
    def __init__(self, feature_order):
        self.feature_order = feature_order

    def sanitize_and_vectorize(self, state_dict):
        vector = []
        for key in self.feature_order:
            val = state_dict.get(key, 0.0)
            try:
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
            except:
                val = 0.0
            vector.append(val)
        return vector
