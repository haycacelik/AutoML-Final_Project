"""A STARTER KIT SCRIPT for SS25 AutoML Exam --- Modality III: Text

You are not expected to follow this script or be constrained to it.

For a test run:
1) Download datasets (see, README) at chosen path
2) Run the script: 
```
python run.py \
    --dataset amazon \
    --epochs 1
```

"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import yaml
import wandb

from automl.core import TextAutoML
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
)

FINAL_TEST_DATASET=...  # TBA later


def main_loop(
        dataset: str,
        output_path: Path,
        data_path: Path,
        seed: int,
        val_size: float,
        vocab_size: int,
        token_length: int,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        fraction_layers_to_finetune: float,
        data_fraction: int,
        load_path: Path = None,
    ) -> None:
    
    # Initialize wandb
    wandb.init(
        project="text-automl",
        name=f"{dataset}_epochs{epochs}_lr{lr}_bs{batch_size}",  # Custom run name
        config={
            "dataset": dataset,
            "seed": seed,
            "val_percentage": val_size,
            "vocab_size": vocab_size,
            "token_length": token_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "fraction_layers_to_finetune": fraction_layers_to_finetune,
            "data_fraction": data_fraction,
        },
        tags=[dataset, "distilbert", "text-classification"]  # Add tags for easy filtering
    )
    
    match dataset:
        case "ag_news":
            dataset_class = AGNewsDataset
        case "imdb":
            dataset_class = IMDBDataset
        case "amazon":
            dataset_class = AmazonReviewsDataset
        case "dbpedia":
            dataset_class = DBpediaDataset
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    print("Fitting Text AutoML")

    # Get the dataset and create dataloaders
    data_path = Path(data_path) if isinstance(data_path, str) else data_path
    data_info = dataset_class(data_path).create_dataloaders(val_size=val_size, random_state=seed, use_class_weights = True)

    train_df = data_info['train_df']
    val_df = data_info.get('val_df', None)
    test_df = data_info['test_df']
    num_classes = data_info["num_classes"]
    if data_info["normalized_class_weights"] is not None:
        normalized_class_weights = data_info["normalized_class_weights"]
    else:
        normalized_class_weights = None

    # if i want to use like 0.5 so half of the data i can do it here, useful for succesive halving
    if data_fraction < 1:
        _subsample = np.random.choice(
            list(range(len(train_df))),
            size=int(data_fraction * len(train_df)),
            replace=False,
        )
        train_df = train_df.iloc[_subsample]
    
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Update wandb config with dataset info
    wandb.config.update({
        "train_size": len(train_df),
        "val_size": len(val_df) if val_df is not None else 0,
        "test_size": len(test_df),
        "num_classes": num_classes
    }, allow_val_change=True)

    # TODO implement bohb for the model hyperparameters

    # Initialize the TextAutoML instance with the best parameters
    automl = TextAutoML(
        # normalized_class_weights=normalized_class_weights,
        normalized_class_weights=None,
        seed=seed,
        vocab_size=vocab_size,
        token_length=token_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        fraction_layers_to_finetune=fraction_layers_to_finetune,
    )

    # Fit the AutoML model on the training and validation datasets
    val_err = automl.fit(
        train_df,
        val_df,
        num_classes=num_classes,
        load_path=load_path,
        save_path=output_path,
    )
    print("Training complete")

    # Predict on the test set
    test_preds, test_labels = automl.predict(test_df)

    # Write the predictions of X_test to disk
    print("Writing predictions to disk")
    with (output_path / "score.yaml").open("w") as f:
        yaml.safe_dump({"val_err": float(val_err)}, f)
    print(f"Saved validation score at {output_path / 'score.yaml'}")
    with (output_path / "test_preds.npy").open("wb") as f:
        np.save(f, test_preds)
    print(f"Saved test prediction at {output_path / 'test_preds.npy'}")

    # Log validation error to wandb
    wandb.log({"val_error": float(val_err)})

    # In case of running on the final exam data, also add the predictions.npy
    # to the correct location for auto evaluation.
    if dataset == FINAL_TEST_DATASET: 
        test_output_path = output_path / "predictions.npy"
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # Check if test_labels has missing data
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        print(f"Accuracy on test set: {acc}")
        
        # Log test accuracy to wandb
        wandb.log({"test_accuracy": acc, "test_error": float(1-acc)})
        
        with (output_path / "score.yaml").open("a+") as f:
            yaml.safe_dump({"test_err": float(1-acc)}, f)
        
        # Log detailed classification report for better insight
        print("Classification Report:")
        report = classification_report(test_labels, test_preds)
        print(f"\n{report}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        print(f"No test labels available for dataset '{dataset}'")

    wandb.finish()
    return val_err


if __name__ == "__main__":
    # Random seed for reproducibility if you are using any randomness,
    # i.e. torch, numpy, pandas, sklearn, etc.
    seed = 42
    dataset = "amazon"
    print(f"Running AutoML for dataset: {dataset}")
    output_path =  (
            Path.cwd().absolute() / 
            "results" / 
            f"dataset={dataset}" / 
            f"seed={seed}"
        )
    output_path = Path(output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = Path.cwd().absolute() / ".data"
    load_path = None
    # "Subsampling of training set, in fraction (0, 1]. 1 is all the data"
    
    main_loop(
        dataset=dataset,
        output_path=Path(output_path).absolute(),
        data_path=Path(data_path).absolute(),
        seed=seed,
        vocab_size=1000,
        token_length=128,
        epochs= 10,
        batch_size=32,
        lr=5e-6,
        weight_decay=0.01,
        data_fraction=1.0,
        load_path=Path(load_path) if load_path is not None else None,
        val_size = 0.2,
        fraction_layers_to_finetune=1.0,  # 1.0 means finetune all layers
    )

# end of file