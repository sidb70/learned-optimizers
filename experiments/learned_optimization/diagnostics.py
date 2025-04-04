from meta_optimizers import LSTMOptimizer, AdamStateTracker, MetaTrainer
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# Rosenbrock function 
def rosenbrock(x):
    a, b = 1.0, 100.0
    loss = (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    grad = torch.autograd.grad(loss, x, create_graph=True)[0]
    return loss, grad

# basic 2d test function
def quadratic(x):
    """Compute a simple quadratic function."""
    loss = 0.5 * (x[0]**2 + x[1]**2)
    grad = torch.autograd.grad(loss, x, create_graph=True)[0]
    return loss, grad

class OptimizerDiagnostics:
    def __init__(self, config):
        self.config = config
        self.diagnostic_dir = "optimizer_diagnostics"
        os.makedirs(self.diagnostic_dir, exist_ok=True)
    
    def analyze_updates(self, meta_optimizer, params, grads):
        """
        Analyze the updates produced by the meta-optimizer.
        
        Args:
            meta_optimizer: The meta-optimizer
            params: Model parameters
            grads: Parameter gradients
            
        Returns:
            Dictionary with analysis results
        """
        # Get updates from meta-optimizer
        meta_optimizer.init_states(params)
        updates = meta_optimizer(params, grads)
        
        # Get updates from Adam for comparison
        adam_tracker = AdamStateTracker(lr=0.001)
        adam_tracker.init_states(params)
        adam_updates = adam_tracker.compute_updates(params, grads)
        
        results = {}
        
        # Analyze update magnitudes
        meta_update_norms = [update.norm().item() for update in updates]
        adam_update_norms = [update.norm().item() for update in adam_updates]
        
        results['meta_update_norms'] = meta_update_norms
        results['adam_update_norms'] = adam_update_norms
        
        # Analyze update directions (cosine similarity with Adam)
        similarities = []
        for meta_update, adam_update in zip(updates, adam_updates):
            cosine_sim = F.cosine_similarity(
                meta_update.view(1, -1), 
                adam_update.view(1, -1)
            ).item()
            similarities.append(cosine_sim)
        
        results['cosine_similarities'] = similarities
        
        # Check for vanishing/exploding gradients in the meta-optimizer
        # by adding small noise to inputs and measuring output variance
        noise_scale = 1e-4
        noisy_grads = [grad + noise_scale * torch.randn_like(grad) for grad in grads]
        meta_optimizer.init_states(params)
        noisy_updates = meta_optimizer(params, noisy_grads)
        
        # Measure sensitivity to input noise
        sensitivities = []
        for clean_update, noisy_update in zip(updates, noisy_updates):
            relative_change = (noisy_update - clean_update).norm() / (clean_update.norm() + 1e-8)
            sensitivities.append(relative_change.item())
        
        results['gradient_sensitivities'] = sensitivities
        
        return results
    
    def run_2d_test_function(self, meta_optimizer, test_function, steps=100):
        """
        Test the optimizer on a simple 2D function.
        
        Args:
            meta_optimizer: The meta-optimizer to test
            test_function: A function that takes a tensor and returns loss and grad
            steps: Number of optimization steps
            
        Returns:
            Trajectory of parameters and losses
        """
        # Initialize parameters at a challenging point
        params = [torch.tensor([2.0, 2.0], requires_grad=True, device=self.config.device)]
        
        # Initialize meta-optimizer states
        meta_optimizer.init_states(params)
        
        # Track trajectory
        trajectory = [params[0].detach().clone().cpu().numpy()]
        losses = []
        
        # Optimization loop
        for _ in range(steps):
            # Compute loss and gradient
            loss, grad = test_function(params[0])
            losses.append(loss.item())
            
            # Get update from meta-optimizer
            updates = meta_optimizer([params[0]], [grad])
            
            # Apply update
            params[0] = params[0] - updates[0]
            
            # Record trajectory
            trajectory.append(params[0].detach().clone().cpu().numpy())
        
        return np.array(trajectory), np.array(losses)
    
    def compare_optimizers_on_2d_function(self, test_2d_function):
        """Compare different optimizers on a 2D test function."""
        # Initialize optimizers
        meta_optimizer = LSTMOptimizer(hidden_size=self.config.hidden_size).to(self.config.device)
        
        # Test each optimizer
        meta_trajectory, meta_losses = self.run_2d_test_function(meta_optimizer, test_2d_function)
        
        # Compare with Adam
        adam_params = [torch.tensor([2.0, 2.0], requires_grad=True, device=self.config.device)]
        adam_opt = torch.optim.Adam([adam_params[0]], lr=0.01)
        
        adam_trajectory = [adam_params[0].detach().clone().cpu().numpy()]
        adam_losses = []
        
        for _ in range(len(meta_losses)):
            loss, grad = test_2d_function(adam_params[0])
            adam_losses.append(loss.item())
            
            adam_params[0].grad = grad
            adam_opt.step()
            adam_opt.zero_grad()
            
            adam_trajectory.append(adam_params[0].detach().clone().cpu().numpy())
        
        # Plot results
        self.plot_2d_optimization(
            [meta_trajectory, np.array(adam_trajectory)],
            [meta_losses, adam_losses],
            ['Meta-Optimizer', 'Adam'],
            str(test_2d_function.__name__)
        )
    
    def plot_2d_optimization(self, trajectories, losses, labels, title):
        """
        Plot optimization trajectories and loss curves.
        
        Args:
            trajectories: List of parameter trajectories
            losses: List of loss curves
            labels: List of optimizer labels
            title: Title for the plot
        """
        plt.figure(figsize=(15, 6))
        
        # Plot trajectories
        plt.subplot(1, 2, 1)
        colors = ['blue', 'red', 'green', 'orange']
        
        # Create contour plot of test function
        x = np.linspace(np.min([t[:, 0].min() for t in trajectories]),
                        np.max([t[:, 0].max() for t in trajectories]), 100)
        y = np.linspace(np.min([t[:, 1].min() for t in trajectories]),
                        np.max([t[:, 1].max() for t in trajectories]), 100)
        X, Y = np.meshgrid(x, y)
        Z = (1 - X)**2 + 100 * (Y - X**2)**2
        plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), alpha=0.3)
        
        for i, (trajectory, label) in enumerate(zip(trajectories, labels)):
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', color=colors[i], 
                     label=label, alpha=0.7, markersize=3)
            plt.plot(trajectory[0, 0], trajectory[0, 1], 'o', color=colors[i], markersize=6)
            plt.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=colors[i], markersize=6)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Optimization Trajectories - {title}')
        plt.legend()
        plt.grid(True)
        
        # Plot loss curves
        plt.subplot(1, 2, 2)
        for i, (loss_curve, label) in enumerate(zip(losses, labels)):
            plt.semilogy(loss_curve, color=colors[i], label=label)
        
        plt.xlabel('Step')
        plt.ylabel('Loss (log scale)')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.diagnostic_dir, f'optimization_{title}.png'))
        plt.close()
    
    def analyze_meta_learning_dynamics(self, meta_trainer):
        """
        Analyze the learning dynamics of the meta-optimizer.
        
        Args:
            meta_trainer: MetaTrainer instance
            
        Returns:
            Dictionary with analysis results
        """
        # Create a simple model for analysis
        model = meta_trainer.create_model()
        model.train()
        
        # Initialize params and gradients
        inputs, targets = meta_trainer.train_task.get_batch()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        model.zero_grad()
        loss.backward()
        
        params = list(model.parameters())
        grads = [p.grad.clone() for p in params]
        
        # Analyze updates over meta-training
        update_stats = []
        
        for i in range(5):  # Analyze at different points
            # Save current meta-optimizer state
            state_dict = meta_trainer.meta_optimizer.state_dict()
            
            # Perform some meta-training
            for _ in range(20):
                meta_trainer.train_step()
            
            # Analyze updates with current meta-optimizer
            update_info = self.analyze_updates(meta_trainer.meta_optimizer, params, grads)
            update_stats.append(update_info)
            
            # Restore original state for next iteration
            meta_trainer.meta_optimizer.load_state_dict(state_dict)
        
        # Analyze results
        results = {
            'update_norm_trends': [stats['meta_update_norms'] for stats in update_stats],
            'cosine_sim_trends': [stats['cosine_similarities'] for stats in update_stats],
            'sensitivity_trends': [stats['gradient_sensitivities'] for stats in update_stats]
        }
        
        # Plot trends
        self.plot_meta_learning_trends(results)
        
        return results
    
    def plot_meta_learning_trends(self, results):
        """
        Plot trends in meta-learning dynamics.
        
        Args:
            results: Dictionary with analysis results
        """
        plt.figure(figsize=(15, 15))
        
        # Plot update norm trends
        plt.subplot(3, 1, 1)
        for i, norms in enumerate(results['update_norm_trends']):
            plt.plot(norms, 'o-', label=f'Iteration {i*20}')
        plt.xlabel('Parameter Index')
        plt.ylabel('Update Norm')
        plt.title('Update Magnitude Trends')
        plt.legend()
        plt.grid(True)
        
        # Plot cosine similarity trends
        plt.subplot(3, 1, 2)
        for i, sims in enumerate(results['cosine_sim_trends']):
            plt.plot(sims, 'o-', label=f'Iteration {i*20}')
        plt.xlabel('Parameter Index')
        plt.ylabel('Cosine Similarity with Adam')
        plt.title('Update Direction Trends')
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot sensitivity trends
        plt.subplot(3, 1, 3)
        for i, sensitivities in enumerate(results['sensitivity_trends']):
            plt.plot(sensitivities, 'o-', label=f'Iteration {i*20}')
        plt.xlabel('Parameter Index')
        plt.ylabel('Gradient Sensitivity')
        plt.title('Gradient Sensitivity Trends')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.diagnostic_dir, 'meta_learning_trends.png'))
        print("Saved meta-learning trends plot to ", os.path.join(self.diagnostic_dir, 'meta_learning_trends.png'))
        plt.close()

def main():
    config = Config()
    
    # Initialize diagnostics
    diagnostics = OptimizerDiagnostics(config)
    
    # Compare optimizers on a 2D function
    diagnostics.compare_optimizers_on_2d_function(quadratic)
    
    meta_trainer = MetaTrainer(config)
    diagnostics.analyze_meta_learning_dynamics(meta_trainer)

if __name__ == "__main__":
    main()