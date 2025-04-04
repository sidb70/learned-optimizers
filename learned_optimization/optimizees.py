import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Forward through layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU to all but the last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
        
        return x