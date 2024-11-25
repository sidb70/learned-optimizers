import torch 
import torch.nn as nn
from typing import Tuple

def flatten_model(model: nn.Module, grad=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens the model parameters into a single tensor.

    Args:
        model: PyTorch model.
        grad: Whether to return the gradients.

    Returns:
        params: Flattened model parameters.
        grads: Flattened model gradients. 0 tensor if grad=False.
        
    """
    params = torch.cat([p.data.view(-1) for p in model.parameters()])
    if grad:
    
        grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
        grads = torch.nan_to_num(grads, nan=0.0) # Set NaN grads to 0
    else:
        grads = torch.zeros_like(params)
    return params, grads