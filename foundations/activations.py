import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        ans = 1/ (1 + np.exp(-z))
        return np.round(ans, 5)


    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        ans = np.maximum(0, z)
        return np.round(ans, 5)
