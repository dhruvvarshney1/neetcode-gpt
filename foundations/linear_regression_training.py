import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        return X @ weights

    def get_gradient(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        N = len(X)

        return (-2 / N) * (X.T @ (Y - y_pred))

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        w = initial_weights.copy()

        learning_rate = 0.01

        for _ in range(num_iterations):

            # Step 1: Predictions
            y_pred = self.get_model_prediction(X, w)

            # Step 2: Compute ALL gradients together
            gradient = self.get_gradient(X, Y, y_pred)

            # Step 3: Simultaneous update
            w -= learning_rate * gradient

        return np.round(w, 5)