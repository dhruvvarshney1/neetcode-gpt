import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def ReLu(self, z):
        return np.maximum(0, z)
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], bias: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)

        a = x

        num_layer = len(weights)

        for layer in range(num_layer):

            z = np.dot(a, weights[layer]) + bias[layer]

            if layer != num_layer-1:
                a = self.ReLu(z)
            else:
                a = z

        return np.round(a, 5)


            
            

            


        



