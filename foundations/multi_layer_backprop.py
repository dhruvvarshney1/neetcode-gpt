import numpy as np
from typing import List


class Solution:
    def ReLu(self, z):
        return np.maximum(0, z)
        
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # Step 1: Converting the arrays to numpy

        x = np.array(x)        # Shape: (input_dim,)
        W1 = np.array(W1)      # Shape: (hidden_dim, input_dim)
        b1 = np.array(b1)      # Shape: (hidden_dim,)
        W2 = np.array(W2)      # Shape: (output_dim, hidden_dim)
        b2 = np.array(b2)      # Shape: (output_dim,)
        y_true = np.array(y_true)  # Shape: (output_dim,)

        # Step 2: Compute forward model

        z1 = np.dot(x, W1.T) + b1
        a1 = self.ReLu(z1)
        z2 = np.dot(a1, W2.T) + b2
        y_pred = z2
        loss = np.mean((y_pred - y_true)**2)

        # Step 3: Compute backward model
        
        # 1. Gradient of loss w.r.t the outputs (z2 / y_pred)
        # dL/dy_pred = 2/N * (y_pred - y_true)
        dZ2 = 2 * (y_pred - y_true) / y_true.size
        
        # 2. Gradients for Output Layer weights and biases
        # W2 shape: (output_dim, hidden_dim) -> outer product of dZ2 and a1
        dW2 = np.outer(dZ2, a1)
        db2 = dZ2
        
        # 3. Propagate the gradient back to the hidden layer activations
        # W2 shape is (output_dim, hidden_dim). dZ2 shape is (output_dim,)
        # np.dot(dZ2, W2) yields a vector of shape (hidden_dim,)
        da1 = np.dot(dZ2, W2)
        
        # 4. Pass gradient through the ReLU activation layer
        # We look at z1 (pre-activation) to see where the mask blocks the gradient
        dZ1 = da1.copy()
        dZ1[z1 <= 0] = 0
        
        # 5. Gradients for Hidden Layer weights and biases
        # W1 shape: (hidden_dim, input_dim) -> outer product of dZ1 and x
        dW1 = np.outer(dZ1, x)
        db1 = dZ1

       # Step 4: Clean negative zeros and Return
        loss_val = float(np.round(loss, 4))
        
        # Adding 0.0 eliminates -0.0 across numpy arrays safely
        dW1_clean = np.round(dW1, 4) + 0.0
        db1_clean = np.round(db1, 4) + 0.0
        dW2_clean = np.round(dW2, 4) + 0.0
        db2_clean = np.round(db2, 4) + 0.0

        return {
            'loss': loss_val,
            'dW1': dW1_clean.tolist(),
            'db1': db1_clean.tolist(),
            'dW2': dW2_clean.tolist(),
            'db2': db2_clean.tolist()
        }

        


        

