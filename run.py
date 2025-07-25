import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import yaml
import wandb
import time
import optuna
from functools import partial

from automl.core import TextAutoML
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
    get_fraction_of_data,
)

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
        classification_head_activation: str,
        load_path: Path = None,
    ) -> None:

    # get start time
    start_time = time.time()
    
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

    # TODO implement BOHB for the model hyperparameters, can remove the parts below and add it in the optimizer

    # TODO implement NAS
    from automl.automl_methods.nas import optuna_nas

    # Create a TPESampler with custom parameters

    # sampler = optuna.samplers.TPESampler(
    #     n_startup_trials=10,   # Number of random trials before using TPE
    #     gamma=0.25,            # Fraction of good trials for modeling (lower means more exploitation)
    #     n_ei_candidates=24,    # Number of candidates sampled when optimizing acquisition function
    #     multivariate=True,     # Model joint distribution of parameters
    #     seed=42
    # )
    objective_fn = partial(
    optuna_nas.objective,
    dataset=dataset,
    epochs=epochs,
    lr=lr,
    batch_size=batch_size,
    seed=seed,
    val_percentage=val_percentage,
    token_length=token_length,
    weight_decay=weight_decay,
    fraction_layers_to_finetune=fraction_layers_to_finetune,
    data_fraction=data_fraction,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    num_classes=num_classes,
    load_path=load_path,
    output_path=output_path
)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_fn, n_trials=5)
    print("Best score:", 1 - study.best_value)
    print("Best params:", study.best_params)

    wandb_nas_run = wandb.init(
        project="text-automl",
        name=f"nas_{dataset}_seed{seed}_best_params",
        config={
            "sampler": "TPESampler",
            "n_trials": 5,
        },
        tags=[dataset, "distilbert", "text-classification", "nas"]  # Add tags for easy filtering
    )

    # TODO get the best parameters from the study, this is incorrect
    best_params = study.best_params
    hidden_dim = best_params.get("hidden_dim", classification_head_hidden_dim)
    dropout_rate = best_params.get("dropout_rate", classification_head_dropout_rate)
    hidden_layer = best_params.get("hidden_layer", classification_head_hidden_layers)
    activation = best_params.get("activation", classification_head_activation)

    wandb_run = wandb.init(
        project="text-automl",
        name=f"nas_{dataset}_hidden_dim{hidden_dim}_dropout{dropout_rate}_hidden_layers{hidden_layer}_activation_{activation}",  # Custom run name
        config={
                "dataset": dataset,
                "seed": seed,
                "val_percentage": val_percentage,
                "token_length": token_length,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "fraction_layers_to_finetune": fraction_layers_to_finetune,
                "data_fraction": data_fraction,
                "classification_head_hidden_dim": hidden_dim,
                "classification_head_dropout_rate": dropout_rate,
                "classification_head_hidden_layers": hidden_layer,
                "classification_head_activation": activation,
                "train_size": len(train_df),
                "val_size": len(val_df) if val_df is not None else 0,
                "test_size": len(test_df),
                "num_classes": num_classes
            },
            tags=[dataset, "distilbert", "text-classification", "nas"]  # Add tags for easy filtering
        )

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
        classification_head_hidden_dim=hidden_dim,
        classification_head_dropout_rate=dropout_rate,
        classification_head_hidden_layers=hidden_layer,
        classification_head_activation=activation,
        train_df=train_df,
        val_df=val_df,
        num_classes=num_classes,
        load_path=load_path,
        save_path=output_path,
        wandb_logger=wandb_run,
    )

    # Fit the AutoML model on the training and validation datasets
    val_err = automl.fit(last_model=True)
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
    wandb_run.log({"val_error": float(val_err)})

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
        wandb_run.log({"test_accuracy": acc, "test_error": float(1-acc)})
        
        with (output_path / "score.yaml").open("a+") as f:
            yaml.safe_dump({"test_err": float(1-acc)}, f)
        
        # Log detailed classification report for better insight
        print("Classification Report:")
        report = classification_report(test_labels, test_preds)
        print(f"\n{report}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        print(f"No test labels available for dataset '{dataset}'")

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

    return val_err


if __name__ == "__main__":
    # seed for reproducibility
    seed = 42
    dataset = "imdb"
    print(f"Running AutoML for dataset: {dataset}")
    output_path =  (
            Path.cwd().absolute() / 
            "results" / 
            f"dataset={dataset}" / 
            f"seed={seed}"
        )
    output_path = Path(output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = Path.cwd().absolute() / "data"
    load_path = None

    main_loop(
        dataset=dataset,
        output_path=Path(output_path).absolute(),
        data_path=Path(data_path).absolute(),
        seed=seed,
        token_length=128,
        epochs= 10,
        batch_size=32,
        lr=5e-6,
        weight_decay=0.01,
        data_fraction=1.0, # "Subsampling of training set, in fraction (0, 1]. 1 is all the data"
        val_percentage = 0.2,
        fraction_layers_to_finetune=0.0,  # 1.0 means finetune all layers
        classification_head_hidden_dim=64,
        classification_head_dropout_rate = 0.2,
        classification_head_hidden_layers = 4,
        classification_head_activation = 'ReLU',  # Default activation, can be changed later
        load_path = Path(load_path) if load_path is not None else None
    )

# end of file