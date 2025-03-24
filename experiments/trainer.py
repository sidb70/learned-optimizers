import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torchviz import make_dot
from graphviz import Source
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
import os
import random
from config import Config
from tasks import MNISTTask, RosenbrockTask
from optimizees import MLP
from tasks import Task, MNISTTask, RosenbrockTask, QuadraticTask
from meta_optimizers import LSTMOptimizer, AdamStateTracker
from optimizees import MLP



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
        
        # Training loop
        for _ in range(steps):
            # Get batch
            inputs, targets = self.task.get_batch()
            
            # Forward pass
            loss = self.task.eval(model, inputs, targets)
            
            # Record metrics
            losses.append(loss.item())
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update parameters
            optimizer.step()
        
        return losses


# Meta-Trainer class
class MetaTrainer:
    def __init__(self, config):
        self.config = config
        
        # Initialize meta-optimizer (LSTM optimizer)
        self.meta_optimizer = LSTMOptimizer(hidden_size=config.hidden_size).to(config.device)
        
        # Initialize meta-optimizer's optimizer
        self.meta_opt = torch.optim.Adam(self.meta_optimizer.parameters(), lr=config.meta_lr)
        
        # Initialize training task
        self.train_task = config.task(config, train=True)
        
        # Initialize evaluation task
        self.eval_task = config.task(config, train=False)
        
        # Initialize Adam state tracker for mimicking
        self.adam_tracker = AdamStateTracker(lr=config.adam_mimic_lr)
        
        # Current meta-iteration counter
        self.current_iteration = 0

    def create_model(self):
        """Create a new MLP model for the task."""
        if self.config.task == MNISTTask:
            input_size = self.config.task_kwargs.get('downsample_size', 16) ** 2
            model = MLP(input_size, self.config.hidden_sizes, 10).to(self.config.device)
            return model
        elif self.config.task == RosenbrockTask or self.config.task == QuadraticTask:
            input_size = 2
            model = MLP(input_size, self.config.hidden_sizes, 1).to(self.config.device)
            return model
        else:
            raise ValueError("Model not implemented for task.")
    
    def is_in_mimic_phase(self):
        """Check if we're in the Adam mimicking phase."""
        return self.current_iteration < self.config.mimic_adam_epochs

            
    def debug_gradient_flow(self):
        """Simplified function with proper gradient flow."""
        # Reset gradients
        self.meta_opt.zero_grad()
        
        # Create a simple model with just one parameter
        model = nn.Linear(2, 1).to(self.config.device)
        params = list(model.parameters())
        
        # Initialize meta-optimizer states
        self.meta_optimizer.init_states(params)
        
        # Create synthetic data
        x = torch.randn(4, 2).to(self.config.device)
        y = torch.randn(4, 1).to(self.config.device)
        
        # Compute initial loss
        output = functional_call(model, params, (x,))
        loss = F.mse_loss(output, y)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Get updates from meta-optimizer (should keep computational graph)
        updates = self.meta_optimizer(params, grads)
        
        # Create new parameters by adding updates (functionally, not in-place)
        new_params = [p + u for p, u in zip(params, updates)]
        
        # Compute new loss with updated parameters
        new_output = functional_call(model, new_params, (x,))
        new_loss = F.mse_loss(new_output, y)
        
        # Backward on new loss
        new_loss.backward()
        
        ## Check gradients
        for name, param in self.meta_optimizer.named_parameters():
            if param.grad is None:
                print(f"DEBUG: No gradient for {name}")
            else:
                print(f"DEBUG: Gradient norm for {name}: {param.grad.norm().item()}")
        
        return new_loss.item()
    def unrolled_optimization(self, task, model, unroll_steps, mimic_adam=False):
        """
        Perform unrolled optimization steps using the meta-optimizer.

        Args:
            task: The task to optimize for.
            model: The model to optimize.
            unroll_steps: Number of unrolled optimization steps.
            mimic_adam: Whether to compute Adam updates for mimicking.

        Returns:
            losses: List of losses during optimization.
            mimic_losses: List of mimicking losses (if mimic_adam is True).
        """
        # Create a dictionary of parameters for functional_call
        params = {name: param.detach().clone().requires_grad_(True)
                for name, param in model.named_parameters()}

        # For the meta-optimizer and Adam tracker, work with a list of parameters
        param_list = list(params.values())

        # Initialize meta-optimizer states
        self.meta_optimizer.init_states(param_list)

        # Initialize Adam states if mimicking
        if mimic_adam:
            self.adam_tracker.init_states(param_list)

        losses, mimic_losses = [], []

        for step in range(unroll_steps):
            inputs, targets = task.get_batch()

            # Forward pass using functional_call with the dictionary of parameters
            outputs = functional_call(model, params, (inputs,))

            # Compute loss and accuracy
            loss = task.loss(outputs, targets)

            losses.append(loss.item())

            # Compute gradients with create_graph=True for higher-order derivatives.
            create_graph = step < self.config.truncated_backprop_steps
            grad_list = torch.autograd.grad(loss, param_list, create_graph=create_graph, allow_unused=True)
            # Convert gradients to a dictionary using the same keys as params.
            grads = {name: (torch.zeros_like(param) if grad is None else grad)
                    for (name, param), grad in zip(params.items(), grad_list)}

            # Compute meta-updates; pass lists of parameters and gradients.
            meta_updates_list = self.meta_optimizer(param_list, list(grads.values()))
            
            # Reassemble meta_updates as a dictionary matching params.
            meta_updates = {name: update for (name, _), update in zip(params.items(), meta_updates_list)}

            if mimic_adam:
                adam_updates_list = self.adam_tracker.compute_updates(param_list, list(grads.values()))
                adam_updates = {name: update for (name, _), update in zip(params.items(), adam_updates_list)}

                # Compute mimicking loss (mean squared error between meta and Adam updates)
                mimic_loss = sum(F.mse_loss(meta_updates[name], adam_updates[name])
                                for name in meta_updates) / len(meta_updates)
                mimic_losses.append(mimic_loss)

                if self.is_in_mimic_phase():
                    meta_updates = adam_updates

            # Update the parameter dictionary functionally
            params = {name: param + meta_updates[name] for name, param in params.items()}
            # Also update the param_list for the next iteration.
            param_list = list(params.values())

            # Optionally detach the parameters from the current graph
            if step == self.config.truncated_backprop_steps - 1 and step < unroll_steps - 1:
                params = {name: param.detach().requires_grad_() for name, param in params.items()} 
                param_list = list(params.values())

        # Compute final loss with updated parameters
        if model.training and unroll_steps > 0:
            inputs, targets = task.get_batch()
            final_outputs = functional_call(model, params, (inputs,))
            final_loss = task.loss(final_outputs, targets)
            losses[-1] = final_loss

        return (losses, mimic_losses) if mimic_adam else losses

    def train_step(self):
        """Perform a single meta-training step."""
        # Reset gradients of meta-parameters
        self.meta_opt.zero_grad()
        
        # Meta-batch loss
        meta_loss = 0
        mimic_meta_loss = 0 if self.is_in_mimic_phase() else None
        losses = [ ]
        # Process multiple tasks in the meta-batch
        for e in range(self.config.meta_batch_size):
            # Create a new MLP model
            model = self.create_model()
            model.train()
            
            # Perform optimization using our learned optimizer
            if self.is_in_mimic_phase():
                sample_losses, mimic_losses = self.unrolled_optimization(
                    self.train_task, 
                    model, 
                    self.config.unroll_steps,
                    mimic_adam=True
                )
                # Add mimicking loss
                mimic_meta_loss = mimic_meta_loss + sum(mimic_losses) / len(mimic_losses)
            else:
                sample_losses = self.unrolled_optimization(
                    self.train_task, 
                    model, 
                    self.config.unroll_steps
                )
            # meta loss = final training loss
            meta_loss = meta_loss + sample_losses[-1]

            losses.append(sample_losses)
            losses[-1][-1] = losses[-1][-1].item()
            # print("making dot")
            # make_dot(meta_loss).render("meta_loss")
            # Source.from_file("meta_loss").render("meta_loss", view=True, cleanup=True)
            # c = input("Press Enter to continue...")
        # Average meta-loss over the batch
        meta_loss = meta_loss / self.config.meta_batch_size
        
        # In mimicking phase, use both losses with weighting
        if self.is_in_mimic_phase():
            mimic_meta_loss = mimic_meta_loss / self.config.meta_batch_size
            total_loss = meta_loss + self.config.mimic_loss_weight * mimic_meta_loss
            
            # Backpropagate through the weighted loss
            total_loss.backward()
            
            # Return both losses
            return meta_loss.item(), mimic_meta_loss.item()
        else:
            # Backpropagate through the meta-optimizer
            meta_loss.backward()
            #make_dot(meta_loss).render("meta_loss_backward")
            #Source.from_file("meta_loss_backward").render("meta_loss_backward", view=True, cleanup=True)
            #c = input("Press Enter to continue...")
            
            # Debug gradient flow
            for name, param in self.meta_optimizer.named_parameters():
                if param.grad is None:
                    print(f"No gradient for {name}")
                # else:
                #     print(f"Gradient norm for {name}: {param.grad.norm().item()}")
            
            # Return just the meta-loss and losses
            return meta_loss, losses
    def evaluate(self):
        """Evaluate the learned optimizer."""
        # Create a new MLP model
        self.meta_optimizer.eval()
        model = self.create_model()
        model.eval()
        
        # Perform optimization using our learned optimizer
        losses = self.unrolled_optimization(
            self.eval_task, 
            model, 
            self.config.eval_unroll_steps
        )
        
        # Return final loss and accuracy
        return losses[-1]
    
    def train(self):
        """Full meta-training loop."""
        print(f"Starting meta-training on {self.config.device}")
        print(f"Will mimic Adam for {self.config.mimic_adam_epochs} iterations before meta-learning")
        
        # Lists to track metrics
        meta_losses = []
        mimic_losses = []
        eval_losses = []
        
        # Time tracking
        start_time = time.time()

        inner_losses_trajectories = []
        
        # Training loop
        for iteration in range(self.config.meta_iterations):
            self.current_iteration = iteration
            
            # Perform a meta-training step
            if self.is_in_mimic_phase():
                meta_loss, mimic_loss = self.train_step()
                meta_losses.append(meta_loss)
                mimic_losses.append(mimic_loss)
                
                # Log progress
                if iteration % self.config.log_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"Iteration {iteration}/{self.config.meta_iterations} (Adam Mimicking), "
                          f"Meta-Loss: {meta_loss:.6f}, Mimic-Loss: {mimic_loss:.6f}, "
                          f"Time: {elapsed:.2f}s")
            else:
                meta_loss, losses_trajectories = self.train_step()
                inner_losses_trajectories.append(losses_trajectories)
                meta_losses.append(meta_loss.item())
                
                # Log progress
                if iteration % self.config.log_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"Iteration {iteration}/{self.config.meta_iterations} (Meta-Learning), "
                          f"Meta-Loss: {meta_loss.item():.6f}, "
                          f"Time: {elapsed:.2f}s")
            

            self.meta_opt.step()
            self.meta_opt.zero_grad()
            
            
            # Evaluate
            if iteration % self.config.eval_interval == 0:
                eval_loss = self.evaluate()
                self.meta_optimizer.train()
                eval_losses.append(eval_loss)
                
                print(f"Evaluation - Loss: {eval_loss:.6f}")
                
                # Plot learning curves
                self.plot_learning_curves(meta_losses, mimic_losses, eval_losses)
                self.plot_inner_trajectory(inner_losses_trajectories)
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(iteration)
                
            # If transitioning from mimicking to meta-learning
            if iteration == self.config.mimic_adam_epochs - 1:
                print("\n" + "="*80)
                print(f"Finished Adam mimicking phase. Transitioning to meta-learning.")
                print("="*80 + "\n")
        
        print("Meta-training complete")
        
        # Final evaluation
        eval_loss = self.evaluate()
        print(f"Final Evaluation - Loss: {eval_loss:.6f}")

        return meta_losses, mimic_losses, eval_losses
    
    def save_checkpoint(self, iteration):
        """Save a checkpoint of the meta-optimizer."""
        checkpoint_path = os.path.join(self.config.save_dir, f"meta_optimizer_{iteration}.pt")
        torch.save({
            'iteration': iteration,
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'meta_opt_state_dict': self.meta_opt.state_dict(),
            'mimic_phase_completed': iteration >= self.config.mimic_adam_epochs,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint of the meta-optimizer."""
        checkpoint = torch.load(checkpoint_path)
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.meta_opt.load_state_dict(checkpoint['meta_opt_state_dict'])
        self.current_iteration = checkpoint['iteration'] + 1
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def plot_inner_trajectory(self, inner_losses_trajectories):
        '''Plot the inner trajectory of the model during meta-training'''
        plt.figure(figsize=(15, 10))
        for i, losses in enumerate(inner_losses_trajectories):
            plt.plot(range(len(losses)), losses, label=f"Task {i}")
        plt.title('Inner Trajectories')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, 'inner_trajectories.png'))
        plt.close()
    def plot_learning_curves(self, meta_losses, mimic_losses, eval_losses):
        """Plot learning curves to visualize training progress."""
        plt.figure(figsize=(15, 10))
        
        # Plot meta-loss
        plt.subplot(2, 2, 1)
        plt.plot(meta_losses)
        plt.axvline(x=self.config.mimic_adam_epochs, color='r', linestyle='--', 
                   label='End of mimicking phase')
        plt.title('Meta-Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot mimic-loss if available
        if mimic_losses:
            plt.subplot(2, 2, 2)
            plt.plot(range(len(mimic_losses)), mimic_losses)
            plt.title('Adam Mimicking Loss')
            plt.xlabel('Iteration')
            plt.ylabel('MSE Loss')
        
        # Plot evaluation loss
        plt.subplot(2, 2, 3)
        eval_x = [i * self.config.eval_interval for i in range(len(eval_losses))]
        plt.plot(eval_x, eval_losses)
        plt.axvline(x=self.config.mimic_adam_epochs, color='r', linestyle='--', 
                   label='End of mimicking phase')
        plt.title('Evaluation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, 'learning_curves.png'))
        plt.close()

