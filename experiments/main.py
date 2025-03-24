from hpo import quick_hyperparameter_search, main_hyperparameter_search

def main():
        
    # Run hyperparameter search
    # For quick testing, use quick_hyperparameter_search()
    # For full search, use main_hyperparameter_search()
    search = quick_hyperparameter_search()
    
    # Train the best configuration
    best_trainer = search.train_best_config(full_iterations=False)
    
    # Final evaluation
    eval_losses = best_trainer.evaluate()
    print(f"Final Best Configuration - Loss: {eval_losses[-1]:.6f}")

if __name__ == '__main__':
    main()