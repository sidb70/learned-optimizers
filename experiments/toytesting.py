import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Callable
import copy
import time
import os

# Configuration
class Config:
    def __init__(self):
        # Meta-optimizer settings
        self.meta_lr = 0.001
        self.meta_batch_size = 4
        self.meta_iterations = 10000
        
        # Inner optimization settings
        self.unroll_steps = 20
        self.eval_unroll_steps = 50
        self.truncated_backprop_steps = 5
        
        # Task settings
        self.inner_batch_size = 128
        self.downsample_size = 16
        self.hidden_sizes = [128, 64]
        
        # Logging and saving
        self.eval_interval = 100
        self.save_interval = 500
        self.log_interval = 10
        self.save_dir = "learned_optimizer_checkpoints"
        
        # LSTM optimizer settings
        self.hidden_size = 20
        
        # Device settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# Create save directory if it doesn't exist
os.makedirs(config.save_dir, exist_ok=True)


# MNIST Task definition
class MNISTTask:
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self.setup_data()
        
    def setup_data(self):
        # Define transforms: resize to 16x16 and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((self.config.downsample_size, self.config.downsample_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        dataset = torchvision.datasets.MNIST(
            root='./.data', 
            train=self.train, 
            download=True, 
            transform=transform
        )
        
        # Create data loader
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.config.inner_batch_size,
            shuffle=True, 
            num_workers=2
        )
        
        # Create iterator to cycle through batches
        self.data_iter = iter(self.dataloader)
    
    def get_batch(self):
        # Get next batch, create new iterator if we've exhausted the current one
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        
        # Move batch to device
        inputs, targets = batch
        return inputs.to(self.config.device), targets.to(self.config.device)
    
    def compute_loss(self, model, inputs, targets):
        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).sum().item() / targets.size(0)
        
        return loss, accuracy


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


# LSTM-based meta-optimizer
class LSTMOptimizer(nn.Module):
    def __init__(self, hidden_size):
        super(LSTMOptimizer, self).__init__()
        
        # Input to LSTM: [param_grad, param_value, momentum, prev_update]
        self.lstm = nn.LSTMCell(4, hidden_size)
        
        # Output layers to compute the update
        self.update_layer = nn.Linear(hidden_size, 1)
        
        # Initialize hidden state and cell state dictionaries
        self.hidden_states = {}
        self.cell_states = {}
        self.momentums = {}
        self.prev_updates = {}
    
    def init_states(self, params):
        """Initialize hidden states, cell states, and momentums for the parameters."""
        for param_id, param in enumerate(params):
            # Initialize hidden and cell states
            self.hidden_states[param_id] = torch.zeros(param.numel(), self.lstm.hidden_size, 
                                                       device=param.device)
            self.cell_states[param_id] = torch.zeros(param.numel(), self.lstm.hidden_size, 
                                                     device=param.device)
            
            # Initialize momentum and previous update
            self.momentums[param_id] = torch.zeros_like(param.view(-1))
            self.prev_updates[param_id] = torch.zeros_like(param.view(-1))
    
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
            
            # Prepare LSTM input: [gradient, parameter, momentum, previous update]
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


# Meta-Trainer class
class MetaTrainer:
    def __init__(self, config):
        self.config = config
        
        # Initialize meta-optimizer (LSTM optimizer)
        self.meta_optimizer = LSTMOptimizer(hidden_size=config.hidden_size).to(config.device)
        
        # Initialize meta-optimizer's optimizer
        self.meta_opt = torch.optim.Adam(self.meta_optimizer.parameters(), lr=config.meta_lr)
        
        # Initialize training task
        self.train_task = MNISTTask(config, train=True)
        
        # Initialize evaluation task
        self.eval_task = MNISTTask(config, train=False)
    
    def create_model(self):
        """Create a new MLP model for MNIST."""
        input_size = self.config.downsample_size * self.config.downsample_size
        model = MLP(input_size, self.config.hidden_sizes, 10).to(self.config.device)
        return model
    
    def unrolled_optimization(self, task, model, unroll_steps):
        """
        Perform unrolled optimization steps using the meta-optimizer.
        
        Args:
            task: The task to optimize for
            model: The model to optimize
            unroll_steps: Number of unrolled optimization steps
            
        Returns:
            losses: List of losses during optimization
            accuracies: List of accuracies during optimization
        """
        # Clone the model to avoid modifying the original
        model = copy.deepcopy(model)
        
        # Get model parameters
        params = list(model.parameters())
        
        # Initialize meta-optimizer states
        self.meta_optimizer.init_states(params)
        
        # Track losses and accuracies
        losses = []
        accuracies = []
        
        # Perform optimization steps
        for step in range(unroll_steps):
            # Get a batch of data
            inputs, targets = task.get_batch()
            
            # Forward pass and compute loss
            loss, accuracy = task.compute_loss(model, inputs, targets)
            
            # Record metrics
            losses.append(loss.item())
            accuracies.append(accuracy)
            
            # Compute gradients
            model.zero_grad()
            loss.backward()
            
            # Get gradients
            gradients = [p.grad.clone() for p in params]
            
            # If in evaluation mode, don't update the model
            if not model.training:
                continue
            
            # Apply meta-optimizer to compute updates
            updates = self.meta_optimizer(params, gradients)
            
            # Apply updates to model parameters
            with torch.no_grad():
                for param, update in zip(params, updates):
                    param.add_(update)
        
        return losses, accuracies
    
    def train_step(self):
        """Perform a single meta-training step."""
        # Reset gradients of meta-parameters
        self.meta_opt.zero_grad()
        
        # Meta-batch loss
        meta_loss = torch.tensor(0.0, device=self.config.device, requires_grad=True)
        
        # Process multiple tasks in the meta-batch
        for _ in range(self.config.meta_batch_size):
            # Create a new MLP model
            model = self.create_model()
            model.train()
            
            # Perform optimization using our learned optimizer
            losses, _ = self.unrolled_optimization(
                self.train_task, 
                model, 
                self.config.unroll_steps
            )
            
            # Meta-loss is the final training loss
            # Make sure we're accumulating Tensor values, not floats
            meta_loss = meta_loss + losses[-1]
        
        # Average meta-loss over the batch
        meta_loss = meta_loss / self.config.meta_batch_size
        
        # Backpropagate through the meta-optimizer
        meta_loss.backward()
        
        # Update meta-parameters
        self.meta_opt.step()
        
        return meta_loss.item()
    
    def evaluate(self):
        """Evaluate the learned optimizer."""
        # Create a new MLP model
        model = self.create_model()
        model.eval()
        
        # Perform optimization using our learned optimizer
        losses, accuracies = self.unrolled_optimization(
            self.eval_task, 
            model, 
            self.config.eval_unroll_steps
        )
        
        # Return final loss and accuracy
        return losses[-1], accuracies[-1]
    
    def train(self):
        """Full meta-training loop."""
        print(f"Starting meta-training on {self.config.device}")
        
        # Lists to track metrics
        meta_losses = []
        eval_losses = []
        eval_accuracies = []
        
        # Time tracking
        start_time = time.time()
        
        # Training loop
        for iteration in range(self.config.meta_iterations):
            # Perform a meta-training step
            meta_loss = self.train_step()
            meta_losses.append(meta_loss)
            
            # Log progress
            if iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration}/{self.config.meta_iterations}, "
                      f"Meta-Loss: {meta_loss:.6f}, "
                      f"Time: {elapsed:.2f}s")
            
            # Evaluate
            if iteration % self.config.eval_interval == 0:
                eval_loss, eval_accuracy = self.evaluate()
                eval_losses.append(eval_loss)
                eval_accuracies.append(eval_accuracy)
                
                print(f"Evaluation - Loss: {eval_loss:.6f}, Accuracy: {eval_accuracy:.4f}")
                
                # Plot learning curves
                self.plot_learning_curves(meta_losses, eval_losses, eval_accuracies)
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(iteration)
        
        print("Meta-training complete")
        
        # Final evaluation
        eval_loss, eval_accuracy = self.evaluate()
        print(f"Final Evaluation - Loss: {eval_loss:.6f}, Accuracy: {eval_accuracy:.4f}")

        return meta_losses, eval_losses, eval_accuracies
    
    def save_checkpoint(self, iteration):
        """Save a checkpoint of the meta-optimizer."""
        checkpoint_path = os.path.join(self.config.save_dir, f"meta_optimizer_{iteration}.pt")
        torch.save({
            'iteration': iteration,
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'meta_opt_state_dict': self.meta_opt.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint of the meta-optimizer."""
        checkpoint = torch.load(checkpoint_path)
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.meta_opt.load_state_dict(checkpoint['meta_opt_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def plot_learning_curves(self, meta_losses, eval_losses, eval_accuracies):
        """Plot learning curves to visualize training progress."""
        plt.figure(figsize=(15, 5))
        
        # Plot meta-loss
        plt.subplot(1, 3, 1)
        plt.plot(meta_losses)
        plt.title('Meta-Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        
        # Plot evaluation loss
        plt.subplot(1, 3, 2)
        eval_x = [i * self.config.eval_interval for i in range(len(eval_losses))]
        plt.plot(eval_x, eval_losses)
        plt.title('Evaluation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        
        # Plot evaluation accuracy
        plt.subplot(1, 3, 3)
        plt.plot(eval_x, eval_accuracies)
        plt.title('Evaluation Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, 'learning_curves.png'))
        plt.close()


# Baseline evaluation
class BaselineEvaluator:
    def __init__(self, task, config):
        self.task = task
        self.config = config
    
    def evaluate_optimizer(self, optimizer_class, optimizer_kwargs, steps):
        """Evaluate a standard optimizer."""
        # Create a new model
        input_size = self.config.downsample_size * self.config.downsample_size
        model = MLP(input_size, self.config.hidden_sizes, 10).to(self.config.device)
        model.train()
        
        # Create optimizer
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        
        # Track metrics
        losses = []
        accuracies = []
        
        # Training loop
        for _ in range(steps):
            # Get batch
            inputs, targets = self.task.get_batch()
            
            # Forward pass
            loss, accuracy = self.task.compute_loss(model, inputs, targets)
            
            # Record metrics
            losses.append(loss.item())
            accuracies.append(accuracy)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update parameters
            optimizer.step()
        
        return losses, accuracies


# Main execution
def main():
    # Create meta-trainer
    trainer = MetaTrainer(config)
    
    # Train meta-optimizer
    meta_losses, eval_losses, eval_accuracies = trainer.train()
    
    # Compare with baseline optimizers
    print("\nComparing with baseline optimizers...")
    
    # Create baseline evaluator
    evaluator = BaselineEvaluator(trainer.eval_task, config)
    
    # Test SGD
    sgd_losses, sgd_accuracies = evaluator.evaluate_optimizer(
        torch.optim.SGD, 
        {'lr': 0.01, 'momentum': 0.9}, 
        config.eval_unroll_steps
    )
    
    # Test Adam
    adam_losses, adam_accuracies = evaluator.evaluate_optimizer(
        torch.optim.Adam, 
        {'lr': 0.001}, 
        config.eval_unroll_steps
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    # Loss comparison
    plt.subplot(1, 2, 1)
    plt.plot(sgd_losses, label='SGD')
    plt.plot(adam_losses, label='Adam')
    plt.plot(eval_losses[-1:], label='Learned Optimizer')
    plt.title('Loss Comparison')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy comparison
    plt.subplot(1, 2, 2)
    plt.plot(sgd_accuracies, label='SGD')
    plt.plot(adam_accuracies, label='Adam')
    plt.plot(eval_accuracies[-1:], label='Learned Optimizer')
    plt.title('Accuracy Comparison')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, 'optimizer_comparison.png'))
    plt.close()
    
    print(f"Final SGD Accuracy: {sgd_accuracies[-1]:.4f}")
    print(f"Final Adam Accuracy: {adam_accuracies[-1]:.4f}")
    print(f"Final Learned Optimizer Accuracy: {eval_accuracies[-1]:.4f}")


if __name__ == "__main__":
    main()