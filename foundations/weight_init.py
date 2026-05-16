import torch
import math
from typing import List

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        sigma = math.sqrt(2.0 / (fan_in + fan_out))
        w = torch.randn(fan_out, fan_in) * sigma
        return torch.round(w, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        sigma = math.sqrt(2.0 / fan_in)
        w = torch.randn(fan_out, fan_in) * sigma
        return torch.round(w, decimals=4).tolist()

    def random_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        # Standard random normal without variance scaling
        w = torch.randn(fan_out, fan_in) * 1.0
        return torch.round(w, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Set the seed exactly once at the start of the simulation
        torch.manual_seed(0)
        
        # 1. Generate all layer weights sequentially from the random stream FIRST
        layers_weights = []
        current_in = input_dim
        
        for i in range(num_layers):
            if init_type == "xavier":
                sigma = math.sqrt(2.0 / (current_in + hidden_dim))
            elif init_type == "kaiming":
                sigma = math.sqrt(2.0 / current_in)
            elif init_type == "random":
                sigma = 1.0  # No scaling factor applied
            else:
                raise ValueError("init_type must be 'xavier', 'kaiming', or 'random'")
                
            w = torch.randn(hidden_dim, current_in) * sigma
            layers_weights.append(w)
            current_in = hidden_dim

        # 2. Generate the input tensor AFTER the weights are established
        x = torch.randn(1, input_dim)
        
        std_history = []
        # Pair Xavier with Tanh, and Kaiming/Random with ReLU for comparison
        activation = torch.tanh if init_type == "xavier" else torch.relu
            
        # 3. Forward pass simulation
        for i in range(num_layers):
            weights = layers_weights[i]
            
            # Linear transformation: x * W^T
            z = torch.matmul(x, weights.t())
            
            # Apply activation function
            x = activation(z)
            
            # Use PyTorch's default sample standard deviation (N-1)
            layer_std = torch.std(x).item()
            std_history.append(round(layer_std, 2))
            
        return std_history