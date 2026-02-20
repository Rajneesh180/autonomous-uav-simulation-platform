import numpy as np
from typing import List, Dict, Any

class FeatureScaler:
    """
    Semantic Intelligence Layer: Feature Normalization & Scaling Engine.
    Converts raw hardware telemetry and operational priorities into
    dimensionless, comparability-safe feature vectors for clustering/RL.
    """

    def __init__(self, method: str = 'minmax'):
        self.method = method
        self.params = {}

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fits the scaler to the multi-attribute data shape and returns normalized output.
        """
        if len(data) == 0:
            return data

        if self.method == 'minmax':
            self.params['min'] = np.min(data, axis=0)
            self.params['max'] = np.max(data, axis=0)
            
            # Avoid divide by zero for constant features
            range_vals = self.params['max'] - self.params['min']
            range_vals[range_vals == 0] = 1.0
            
            return (data - self.params['min']) / range_vals

        elif self.method == 'zscore':
            self.params['mean'] = np.mean(data, axis=0)
            self.params['std'] = np.std(data, axis=0)
            
            std_vals = self.params['std']
            std_vals[std_vals == 0] = 1.0  # Avoid division by zero
            
            return (data - self.params['mean']) / std_vals

        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

    @staticmethod
    def apply_time_decay(deadline_vector: np.ndarray, current_time: float, steepness: float = 1.0) -> np.ndarray:
        """
        Calculates Urgency Weights from Deadlines using Time Decay formulation.
        Urgency = 1 / (max(margin, 0) + 1)^steepness
        """
        # Exclude infinities from extreme penalization by capping them
        safe_deadlines = np.where(deadline_vector == np.inf, 99999.0, deadline_vector)
        margins = np.maximum(0, safe_deadlines - current_time)
        return 1.0 / np.power(margins + 1.0, steepness)
