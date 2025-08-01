#!/usr/bin/env python3
"""
Example script showing how to use TextAutoML with custom classification heads and Optuna optimization.

Usage:
    python automl_with_optuna_example.py --dataset amazon --use-custom-head --optimize
"""

import argparse
import pandas as pd
import wandb
from pathlib import Path
from automl.optuna_core import TextAutoML
from automl.datasets import AmazonReviewsDataset, AGNewsDataset, IMDBDataset, DBpediaDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["amazon", "ag_news", "imdb", "dbpedia"])
    parser.add_argument("--data-path", type=Path, default=".data")
    parser.add_argument("--use-custom-head", action="store_true", help="Use custom classification head")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna optimization")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=1800, help="Optimization timeout in seconds")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (if not optimizing)")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Dataset mapping
    dataset_classes = {
        "amazon": AmazonReviewsDataset,
        "ag_news": AGNewsDataset,
        "imdb": IMDBDataset,
        "dbpedia": DBpediaDataset
    }
    
    # Initialize wandb
    wandb.init(
        project="automl-custom-head-optuna",
        name=f"{args.dataset}_custom_head_{args.use_custom_head}_optimize_{args.optimize}",
        config=vars(args),
        tags=[args.dataset, "custom-head" if args.use_custom_head else "standard", "optuna" if args.optimize else "manual"]
    )
    
    # Load dataset
    dataset_class = dataset_classes[args.dataset]
    data_info = dataset_class(args.data_path).create_dataloaders(
        val_size=0.2, 
        random_state=args.seed,
        use_class_weights=True
    )
    
    train_df = data_info['train_df']
    val_df = data_info['val_df']
    test_df = data_info['test_df']
    num_classes = data_info['num_classes']
    normalized_class_weights = data_info['normalized_class_weights']
    
    print(f"Dataset: {args.dataset}")
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    print(f"Number of classes: {num_classes}")
    
    # Initialize AutoML model
    automl = TextAutoML(
        normalized_class_weights=normalized_class_weights,
        seed=args.seed,
        max_epochs=args.epochs,
        use_custom_head=args.use_custom_head,
        head_dropout_rate=0.2,
        head_hidden_layers=2,
        head_hidden_dim=256,
    )
    
    if args.optimize:
        print("Starting Optuna optimization...")
        best_params, best_value = automl.optimize_hyperparameters(
            train_df=train_df,
            val_df=val_df,
            num_classes=num_classes,
            n_trials=args.n_trials,
            timeout=args.timeout,
            study_name=f"automl_{args.dataset}_custom_{args.use_custom_head}"
        )
        
        print(f"Optimization complete!")
        print(f"Best validation error: {best_value:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Log best results to wandb
        wandb.log({
            "best_val_error": best_value,
            "best_params": best_params
        })
        
    else:
        print("Training with manual hyperparameters...")
        val_error = automl.fit(
            train_df=train_df,
            val_df=val_df,
            num_classes=num_classes
        )
        print(f"Validation error: {val_error:.4f}")
    
    # Test the final model
    test_preds, test_labels = automl.predict(test_df)
    
    if test_labels is not None and not pd.isna(test_labels).any():
        from sklearn.metrics import accuracy_score, classification_report
        test_acc = accuracy_score(test_labels, test_preds)
        print(f"Test accuracy: {test_acc:.4f}")
        
        wandb.log({
            "test_accuracy": test_acc,
            "test_error": 1 - test_acc
        })
        
        print("Classification Report:")
        print(classification_report(test_labels, test_preds))
    
    wandb.finish()

if __name__ == "__main__":
    main()
