import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import yaml
import wandb
from wandb.sdk.wandb_run import Run
import time

from automl.core import TextAutoML
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
    get_fraction_of_data,
)
from config.config_loader import load_config, validate_config

FINAL_TEST_DATASET=...  # TBA later


def main_loop(
        dataset: str,
        output_path: Path,
        data_path: Path,
        seed: int,
        val_percentage: float,
        token_length: int,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        fraction_layers_to_finetune: float,
        data_fraction: float,
        classification_head_hidden_dim: int,
        classification_head_dropout_rate: float,
        classification_head_hidden_layers: int,
        wandb_logger: Run,
        load_path: Path = None,
    ) -> None:

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
    data_info = dataset_class(data_path).create_dataloaders(val_size=val_percentage, random_state=seed, use_class_weights = True)

    train_df = data_info['train_df']
    val_df = data_info.get('val_df', None)
    test_df = data_info['test_df']
    num_classes = data_info["num_classes"]
    if data_info["normalized_class_weights"] is not None:
        normalized_class_weights = data_info["normalized_class_weights"]
    else:
        normalized_class_weights = None

    # if i want to use like 0.5 so half of the data i can do it here, useful for successive halving
    if data_fraction < 1:
        print(f"Subsampling training data to {data_fraction * 100}%")
        train_df = get_fraction_of_data(train_df, data_fraction)
    
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Update wandb config with dataset info
    wandb_logger.config.update({
        "train_size": len(train_df),
        "val_size": len(val_df) if val_df is not None else 0,
        "test_size": len(test_df),
        "num_classes": num_classes
    }, allow_val_change=True)

    # TODO implement BOHB for the model hyperparameters, can remove the parts below and add it in the optimizer

    # Initialize the TextAutoML instance with the best parameters
    automl = TextAutoML(
        # normalized_class_weights=normalized_class_weights,
        normalized_class_weights=None,
        seed=seed,
        token_length=token_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        fraction_layers_to_finetune=fraction_layers_to_finetune,
        classification_head_hidden_dim=classification_head_hidden_dim,
        classification_head_dropout_rate=classification_head_dropout_rate,
        classification_head_hidden_layers=classification_head_hidden_layers,
        train_df=train_df,
        val_df=val_df,
        num_classes=num_classes,
        load_path=load_path,
        save_path=output_path,
        wandb_logger=wandb_logger,
    )

    # Fit the AutoML model on the training and validation datasets
    val_err = automl.fit()
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
    wandb_logger.log({"val_error": float(val_err)})

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
        wandb_logger.log({"test_accuracy": acc, "test_error": float(1-acc)})
        
        with (output_path / "score.yaml").open("a+") as f:
            yaml.safe_dump({"test_err": float(1-acc)}, f)
        
        # Log detailed classification report for better insight
        print("Classification Report:")
        report = classification_report(test_labels, test_preds)
        print(f"\n{report}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        print(f"No test labels available for dataset '{dataset}'")


if __name__ == "__main__":
    config: dict = load_config("config/config.yaml")
    validate_config(config)

    dataset = "imdb"

    output_path = Path(config.pop("output_path")) / f"dataset={dataset}" / f"seed={config['seed']}"
    data_path = Path(config.pop("data_path"))

    print(f"Running AutoML for dataset: {dataset}")

    # Initialize wandb
    wandb_run = wandb.init(
        project="text-automl",
        name=f"{dataset}_epochs{config['epochs']}_lr{config['lr']}_bs{config['batch_size']}",  # Custom run name
        config={**config, "dataset": dataset},
        tags=[dataset, "distilbert", "text-classification"]  # Add tags for easy filtering
    )

    # get start time
    start_time = time.time()

    main_loop(
        dataset=dataset,
        output_path=output_path,
        data_path=data_path,
        seed=config["seed"],
        token_length=config["token_length"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        data_fraction=config["data_fraction"],
        val_percentage=config["val_percentage"],
        fraction_layers_to_finetune=config["fraction_layers_to_finetune"],
        classification_head_hidden_dim=config["classification_head_hidden_dim"],
        classification_head_dropout_rate=config["classification_head_dropout_rate"],
        classification_head_hidden_layers=config["classification_head_hidden_layers"],
        wandb_logger=wandb_run
    )

    # Print total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    # turn it to minute and seconds
    elapsed_minutes = elapsed_time // 60
    elapsed_seconds = elapsed_time % 60
    print(f"Total execution time: {elapsed_minutes:.0f} minutes and {elapsed_seconds:.2f} seconds")

    # Log total execution time to wandb
    # TODO add it to the results file, also make the results file better.
    wandb_run.log({"total_execution_time": elapsed_time})

    wandb_run.finish()

# end of file