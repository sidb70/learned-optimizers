import torch
import torch.nn as nn
from gmn_lim.graph_construct.constants import NODE_TYPES, EDGE_TYPES, CONV_LAYERS, NORM_LAYERS, RESIDUAL_LAYERS, NODE_TYPE_TO_LAYER
from gmn_lim.graph_construct.utils import (
    make_node_feat,
    make_edge_attr,
    conv_to_graph,
    linear_to_graph,
    norm_to_graph,
    ffn_to_graph,
    basic_block_to_graph,
    self_attention_to_graph,
    equiv_set_linear_to_graph,
    triplanar_to_graph,
)
from gmn_lim.graph_construct.model_arch_graph import (
    seq_to_feats,
    sequential_to_arch,
    arch_to_graph,
    graph_to_arch,
    arch_to_named_params
)
from gmn_lim.graph_models import EdgeMPNNDiT




# Adam state tracker for mimicking
class AdamStateTracker:
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.states = {}
    
    def init_states(self, params):
        """Initialize Adam states for parameters."""
        for param_id, param in enumerate(params):
            self.states[param_id] = {
                'm': torch.zeros_like(param.data),  # First moment
                'v': torch.zeros_like(param.data),  # Second moment
                'step': 0  # Step counter
            }
    
    def compute_updates(self, params, grads):
        """Compute parameter updates using Adam."""
        updates = []
        
        for param_id, (param, grad) in enumerate(zip(params, grads)):
            # Get Adam state for this parameter
            state = self.states[param_id]
            
            # Update step count
            state['step'] += 1
            
            # Compute bias-corrected learning rate
            step = state['step']
            bias_correction1 = 1 - self.betas[0] ** step
            bias_correction2 = 1 - self.betas[1] ** step
            
            # Update moments
            state['m'] = self.betas[0] * state['m'] + (1 - self.betas[0]) * grad
            state['v'] = self.betas[1] * state['v'] + (1 - self.betas[1]) * (grad ** 2)
            
            # Compute bias-corrected moments
            m_hat = state['m'] / bias_correction1
            v_hat = state['v'] / bias_correction2
            
            # Compute update
            update = -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            updates.append(update)
        
        return updates

