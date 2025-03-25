from config import Config
from tasks import RosenbrockTask, MNISTTask
from trainer import MetaTrainer, BaselineEvaluator
import torch
import os
import copy
import json
import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor  
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

class HyperparameterSearch:
    def __init__(self, base_config):
        """
        Initialize hyperparameter search with a base configuration.
        
        Args:
            base_config: The base Config object to modify
        """
        self.base_config = base_config
        self.results = []
        self.best_config = None
        self.best_accuracy = 0
        
        # Create a directory for hyperparameter search results
        self.results_dir = "hyperparameter_search_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def generate_configs(self, param_grid):
        """
        Generate configurations from a parameter grid.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of Config objects with different parameter combinations
        """
        # Get all parameter combinations
        keys = param_grid.keys()
        values = param_grid.values()
        configs = []
        
        for combination in itertools.product(*values):
            # Create a copy of the base config
            config = copy.deepcopy(self.base_config)
            
            # Apply the parameter combination
            for key, value in zip(keys, combination):
                # Handle nested attributes (e.g., 'optimizer.lr')
                if '.' in key:
                    obj_name, attr_name = key.split('.')
                    setattr(getattr(config, obj_name), attr_name, value)
                else:
                    setattr(config, key, value)
            
            # Add an identifier for this configuration
            config.run_id = f"run_{len(configs)}"
            config.save_dir = os.path.join(self.results_dir, config.run_id)
            os.makedirs(config.save_dir, exist_ok=True)
            
            configs.append(config)
        
        return configs
    
    def evaluate_config(self, config):
        """
        Evaluate a single configuration.
        
        Args:
            config: Config object to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating configuration {config.run_id}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(config.seed)
        
        # Create trainer with this configuration
        trainer = MetaTrainer(config)
        
        # Reduce the number of meta iterations for the search
        original_iterations = config.meta_iterations
        config.meta_iterations = 200# Use fewer iterations for faster search
        
        # Train the meta-optimizer
        meta_losses, mimic_losses, eval_losses = trainer.train()
        
        # Restore original iterations
        config.meta_iterations = original_iterations
        
        # Evaluate the final model
        final_losses = trainer.evaluate()
        
        # Save the final model
        trainer.save_checkpoint(config.meta_iterations)
        
        # Save the config parameters as JSON
        config_dict = {key: value for key, value in vars(config).items() 
                       if not key.startswith('_') and not callable(value)}
        
        # Convert device to string
        if 'device' in config_dict:
            config_dict['device'] = str(config_dict['device'])
        
        with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        # Return evaluation results
        result = {
            'run_id': config.run_id,
            'final_losses': final_losses,
            'final_loss': final_losses[-1],
            'config': {k: getattr(config, k) for k in vars(config) 
                      if not k.startswith('_') and not callable(getattr(config, k))
                      and not isinstance(getattr(config, k), torch.device)},
            'config_path': os.path.join(config.save_dir, 'config.json'),
            'checkpoint_path': os.path.join(config.save_dir, f"meta_optimizer_{config.meta_iterations}.pt")
        }
        
        return result
    
    def run_search(self, param_grid, num_workers=1):
        """
        Run hyperparameter search in parallel.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            num_workers: Number of parallel workers
            
        Returns:
            DataFrame with results
        """
        configs = self.generate_configs(param_grid)
        print(f"Generated {len(configs)} configurations to evaluate")
        
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                self.results = list(executor.map(self.evaluate_config, configs))
        else:
            self.results = [self.evaluate_config(config) for config in configs]
        
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(self.results_dir, 'search_results.csv'), index=False)
        
        # Find best configuration
        best_idx = results_df['final_loss'].idxmin()
        self.best_config = configs[best_idx]
        self.best_loss = results_df.loc[best_idx, 'final_loss']
        
        print(f"Best configuration: {self.best_config.run_id}")
        print(f"Best loss: {self.best_loss:.4f}")
        
        return results_df
    
    def visualize_results(self, results_df):
        """
        Visualize hyperparameter search results.
        
        Args:
            results_df: DataFrame with search results
        """
        # Create a summary figure
        plt.figure(figsize=(12, 8))
        
        # Sort by accuracy
        sorted_results = results_df.sort_values('final_loss', ascending=True)
        
        plt.subplot(2, 1, 1)
        plt.bar(range(len(sorted_results)), sorted_results['final_loss'])
        plt.xlabel('Configuration Rank')
        plt.ylabel('Final Loss')
        plt.title('Hyperparameter Search Results')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'search_results.png'))
        plt.close()
        
        # For top parameters, create more detailed visualization
        if results_df.ndim > 1:
            self._visualize_parameter_effects(results_df)
    
    def _visualize_parameter_effects(self, results_df):
        """
        Visualize the effects of individual parameters.
        
        Args:
            results_df: DataFrame with search results
        """
        # Extract configuration parameters
        config_df = pd.DataFrame([r['config'] for r in self.results])
        # config_df = pd.DataFrame([
        #     {k: tuple(v) if isinstance(v, list) else v for k, v in r['config'].items()}
        #     for r in self.results
        # ])

        
        # Combine with results
        combined_df = pd.concat([results_df[['run_id', 'final_loss']], config_df], axis=1)

        # convert entries with lists to tuples
        varying_params = []
        for col in config_df.columns:
            if pd.api.types.is_list_like(config_df[col].iloc[0]):
                config_df[col] = config_df[col].apply(tuple)
            if col not in ['run_id', 'task_kwargs', 'save_dir']  and len(config_df[col].unique()) > 1:
                varying_params.append(col)
        
        if not varying_params:
            return
        
        # Create visualization for each varying parameter
        plt.figure(figsize=(15, 5 * len(varying_params)))
        
        for i, param in enumerate(varying_params):
            # Skip non-numeric parameters
            if not pd.api.types.is_numeric_dtype(combined_df[param]):
                continue
                
            plt.subplot(len(varying_params), 1, i+1)
            plt.scatter(combined_df[param], combined_df['final_loss'])
            plt.xlabel(param)
            plt.ylabel('Final Loss')
            plt.title(f'Effect of {param} on Loss')
            
            # Add trend line if more than 2 points
            if len(combined_df) > 2:
                try:
                    z = np.polyfit(combined_df[param], combined_df['final_loss'], 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(combined_df[param].unique()), 
                             p(sorted(combined_df[param].unique())), "r--")
                except:
                    pass
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'parameter_effects.png'))
        plt.close()
    
    def train_best_config(self, full_iterations=True):
        """
        Train the best configuration for the full number of iterations.
        
        Args:
            full_iterations: Whether to use the full number of iterations
            
        Returns:
            Trained MetaTrainer
        """
        if self.best_config is None:
            raise ValueError("No best configuration found. Run search first.")
        
        print(f"Training best configuration: {self.best_config.run_id}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.best_config.seed)
        torch.cuda.manual_seed(self.best_config.seed)
        torch.cuda.manual_seed_all(self.best_config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.best_config.seed)
        
        # Create trainer with the best configuration
        trainer = MetaTrainer(self.best_config)
        
        # Use the original number of iterations
        if not full_iterations:
            self.best_config.meta_iterations = 200 # Reduced for testing
            
        # Train the meta-optimizer
        trainer.train()
        
        return trainer


# Define the hyperparameter grid
def main_hyperparameter_search():
    # Create a base configuration
    base_config = Config()
    
    # Define parameter grid
    param_grid = {
        'meta_lr': [0.0005, 0.001, 0.002],
        'hidden_size': [20, 40, 60],
        'unroll_steps': [10, 15, 25, 50],
        'meta_batch_size': [2, 4, 8],
        'mimic_adam_epochs': [0, 100, 200],
        'adam_mimic_lr': [0.0005, 0.001, 0.002]
    }
    
    # Initialize search
    search = HyperparameterSearch(base_config)
    
    # Run hyperparameter search (use more workers on multi-core systems)
    results_df = search.run_search(param_grid, num_workers=1)
    
    # Visualize results
    search.visualize_results(results_df)
    
    # Train best configuration
    best_trainer = search.train_best_config(full_iterations=False)
    
    # Compare with baselines
    print("\nComparing with baseline optimizers...")
    
    # Create baseline evaluator
    evaluator = BaselineEvaluator(best_trainer.eval_task, best_trainer.config)
    
    # Test SGD
    sgd_losses = evaluator.evaluate_optimizer(
        torch.optim.SGD, 
        {'lr': 0.01, 'momentum': 0.9}, 
        best_trainer.config.eval_unroll_steps
    )
    
    # Test Adam
    adam_losses = evaluator.evaluate_optimizer(
        torch.optim.Adam, 
        {'lr': 0.001}, 
        best_trainer.config.eval_unroll_steps
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    # Evaluate the best model
    eval_losses = best_trainer.evaluate()
    
    # Loss comparison
    plt.subplot(1, 2, 1)
    plt.plot(sgd_losses, label='SGD')
    plt.plot(adam_losses, label='Adam')
    plt.axhline(y=eval_losses, color='g', linestyle='-', label='Learned Optimizer')
    plt.title('Loss Comparison')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(best_trainer.config.save_dir, 'final_optimizer_comparison.png'))
    plt.close()
    
    # Print final results
    print(f"Final Adam Loss: {adam_losses[-1]:.6f}")
    print(f"Final SGD Loss: {sgd_losses[-1]:.6f}")
    print(f"Final Learned Optimizer Loss: {eval_losses[-1]:.6f}")
    
    return search

# Define a smaller grid for faster testing
def quick_hyperparameter_search():
    # Create a base configuration
    base_config = Config()
    
    # Use a smaller grid for testing
    param_grid = {
        'meta_lr': [0.001, 0.002],
        'hidden_size': [40, 60],
        'unroll_steps': [10, 50],
        'meta_batch_size': [4, 8],
    }
    
    # Initialize search
    search = HyperparameterSearch(base_config)
    
    # Reduce iterations for quick testing
    base_config.meta_iterations = 200
    # Run search
    results_df = search.run_search(param_grid, num_workers=base_config.num_workers)
    
    # Visualize results
    search.visualize_results(results_df)
    
    return search
# Add learning rate scheduling for meta-optimizer
class MetaLRScheduler:
    def __init__(self, meta_optimizer, meta_opt, initial_lr, decay_factor=0.5, patience=500):
        """
        Learning rate scheduler for meta-optimizer.
        
        Args:
            meta_optimizer: The meta-optimizer model
            meta_opt: The optimizer for the meta-optimizer
            initial_lr: Initial learning rate
            decay_factor: Factor to multiply learning rate when decaying
            patience: Number of iterations without improvement before decaying
        """
        self.meta_optimizer = meta_optimizer
        self.meta_opt = meta_opt
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
    
    def step(self, loss):
        """
        Update learning rate based on loss.
        
        Args:
            loss: Current loss value
            
        Returns:
            bool: Whether learning rate was decayed
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
            return False
        
        self.wait += 1
        
        if self.wait >= self.patience:
            self.current_lr *= self.decay_factor
            for param_group in self.meta_opt.param_groups:
                param_group['lr'] = self.current_lr
            self.wait = 0
            return True
        
        return False