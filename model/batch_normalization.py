import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool
    ) -> Tuple[List[List[float]], List[float], List[float]]:

        # Convert to numpy arrays
        x = np.array(x, dtype=np.float64)
        gamma = np.array(gamma, dtype=np.float64)
        beta = np.array(beta, dtype=np.float64)
        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)

        if training:
            # Compute batch statistics
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # Normalize
            x_hat = (x - mu) / np.sqrt(var + eps)

            # Update running statistics
            running_mean = (1 - momentum) * running_mean + momentum * mu
            running_var = (1 - momentum) * running_var + momentum * var

        else:
            # Use running statistics
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        # Scale and shift
        y = gamma * x_hat + beta

        # Round
        y = np.round(y, 4)
        running_mean = np.round(running_mean, 4)
        running_var = np.round(running_var, 4)

        return (
            y.tolist(),
            running_mean.tolist(),
            running_var.tolist()
        )