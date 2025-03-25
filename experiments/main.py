from hpo import quick_hyperparameter_search, main_hyperparameter_search
from config import Config
from trainer import MetaTrainer, BaselineEvaluator
import torch
from torch.optim import Adam
from tasks import Task, RosenbrockTask, MNISTTask, QuadraticTask
from meta_optimizers import LSTMOptimizer
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def main():
        
    # Run hyperparameter search
    # For quick testing, use quick_hyperparameter_search()
    # For full search, use main_hyperparameter_search()
    # search = quick_hyperparameter_search()
    
    # # Train the best configuration
    # best_trainer = search.train_best_config(full_iterations=False)
    
    # # # #Final evaluation
    # eval_losses = best_trainer.evaluate()
    # print(f"Final Best Configuration - Loss: {eval_losses[-1]:.6f}")

    config_path = './experiments/config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = Config()
    config.load(config_dict)

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
    config.meta_iterations = 500 
    
    # Train the meta-optimizer
    # meta_losses, mimic_losses, eval_losses = trainer.train()
    
    # Restore original iterations
    config.meta_iterations = original_iterations
    
    # Evaluate the final model
    state_dict = torch.load('/home/siddhartha/learned-optimizers/hyperparameter_search_results/run_10/meta_optimizer_200.pt')['meta_optimizer_state_dict']
    trainer.meta_optimizer.load_state_dict(state_dict)
    final_losses = trainer.evaluate()
    
    # Save the final model
    # trainer.save_checkpoint(config.meta_iterations)

    
    evaluator = BaselineEvaluator(config.task(config), config)
    # Test SGD
    sgd_losses = evaluator.evaluate_optimizer(
        torch.optim.SGD, 
        {'lr': 0.03, 'momentum': 0.9}, 
        config.eval_unroll_steps
    )
    
    # Test Adam
    adam_losses = evaluator.evaluate_optimizer(
        torch.optim.Adam, 
        {'lr': 0.03, 'betas': (0.9, 0.999)},
        config.eval_unroll_steps
    )

    # plot results
    plt.figure(figsize=(10, 6))
    plt.plot(final_losses, label='LSTM')
    plt.plot(sgd_losses, label='SGD')
    plt.plot(adam_losses, label='Adam')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Meta-Iteration')
    plt.ylabel('Loss')
    plt.title('Final Optimizer Comparison on Quadratic Task')
    plt.savefig('final_results.png')
    plt.show()



if __name__ == '__main__':
    main()