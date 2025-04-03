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
# config = Config()
# # Create save directory if it doesn't exist
# os.makedirs(config.save_dir, exist_ok=True)

# LSTM-based meta-optimizer
class LSTMOptimizer(nn.Module):
    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.zeros_(layer.weight.data)
            layer.bias.data.fill_(0.0)
        

    def __init__(self, hidden_size):
        super(LSTMOptimizer, self).__init__()
        
        # lstm inputs: [param_grad, param_value, momentum, prev_update]
        self.lstm = nn.LSTMCell(4, hidden_size)
        
        # Output layers to compute the update
        self.update_layer = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(self.init_weights)
        
        # Initialize hidden state and cell state dictionaries
        self.hidden_states = {}
        self.cell_states = {}
        self.momentums = {}
        self.prev_updates = {}
    

    def init_states(self, params):
        """Initialize hidden states, cell states, and momentums for the parameters."""
        for param_id, param in enumerate(params):
            # Flatten parameter and ensure consistency in shapes
            flat_param = param.view(-1)
            
            # Initialize hidden and cell states
            self.hidden_states[param_id] = torch.zeros(flat_param.size(0), self.lstm.hidden_size, 
                                                    device=param.device)
            self.cell_states[param_id] = torch.zeros(flat_param.size(0), self.lstm.hidden_size, 
                                                    device=param.device)
            
            # Initialize momentum and previous update, matching the flattened parameter size
            self.momentums[param_id] = torch.zeros_like(flat_param)
            self.prev_updates[param_id] = torch.zeros_like(flat_param)  # Ensure matching size


    
    def forward(self, params, gradients):
        """
        Apply the learned optimizer to update model parameters.
        
        Args:
            params: List of model parameters
            gradients: List of parameter gradients
            
        Returns:
            updates: List of parameter updates
        """
        updates = []
        
        # Process each parameter
        for param_id, (param, grad) in enumerate(zip(params, gradients)):
            # Reshape tensors
            flat_param = param.view(-1)
            flat_grad = grad.view(-1)
            
            # Get states for this parameter
            h_state = self.hidden_states[param_id]
            c_state = self.cell_states[param_id]
            momentum = self.momentums[param_id]
            prev_update = self.prev_updates[param_id]
            if prev_update.numel() == 1:
                prev_update = prev_update.expand_as(flat_param)

            # Concatenate input
            lstm_input = torch.stack([flat_grad, flat_param, momentum, prev_update], dim=1)
            
            # Run LSTM cell
            h_state, c_state = self.lstm(lstm_input, (h_state, c_state))

            # Compute update
            update = self.update_layer(h_state).squeeze()
            
            # Update momentum
            momentum = 0.9 * momentum + update
            
            # Store updated states
            self.hidden_states[param_id] = h_state
            self.cell_states[param_id] = c_state
            self.momentums[param_id] = momentum
            self.prev_updates[param_id] = update
            
            # Reshape update back to parameter shape
            updates.append(update.view(param.shape))
        
        return updates

class GMNOptimizer(nn.Module):
    def init_weights(self):
        for block in self.convs:
            if block.update_node:
                nn.init.constant_(block.node_mlp[-1].weight, 1)
                nn.init.constant_(block.node_mlp[-1].bias, 1)

            if block.update_edge:
                nn.init.constant_(block.edge_mlp[-1].weight, 1)
                nn.init.constant_(block.edge_mlp[-1].bias, 1)

    def __init__(self, hidden_size):
        super(GMNOptimizer, self).__init__()
        
        # lstm inputs: [param_grad, param_value, momentum, prev_update]
        self.gmn = EdgeMPNNDiT(
            node_in_dim=3,
            edge_in_dim=7,
            hidden_dim=hidden_size,
            edge_out_dim=1,
            num_layers=3,
        )
        
        self.init_weights()
        self.arch = None


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

