from hpo import quick_hyperparameter_search

def main():
        
    # Run hyperparameter search
    # For quick testing, use quick_hyperparameter_search()
    # For full search, use main_hyperparameter_search()
    search = quick_hyperparameter_search()
    
    # Train the best configuration
    best_trainer = search.train_best_config(full_iterations=False)
    
    # Final evaluation
    eval_loss, eval_accuracy = best_trainer.evaluate()
    print(f"Final Best Configuration - Loss: {eval_loss:.6f}, Accuracy: {eval_accuracy:.4f}")

if __name__ == '__main__':
    main()