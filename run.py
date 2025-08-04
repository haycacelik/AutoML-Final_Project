import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import yaml
from wandb.sdk.wandb_run import Run
import time
print("Starting the AutoML run...")  # Debugging line to check if the script starts correctly
print("Optuna imported successfully")  # Debugging line to check if Optuna imports correctly
from automl.optuna_core import TextAutoML
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
)
from config.config_loader import load_config, validate_config
from automl.automl_methods.hpo.our_method import method

FINAL_TEST_DATASET=...  # TBA later


def load_data(dataset: str,
              data_path: Path,
              val_percentage: float,
              seed: int,
              data_fraction: float
              ):
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
    data_info = dataset_class(data_path).create_dataloaders(val_size=val_percentage, random_state=seed,
                                                            use_class_weights=True, data_fraction=data_fraction)

    train_df = data_info['train_df']
    val_df = data_info.get('val_df', None)
    test_df = data_info['test_df']
    num_classes = data_info["num_classes"]
    if data_info["normalized_class_weights"] is not None:
        normalized_class_weights = data_info["normalized_class_weights"]
    else:
        normalized_class_weights = None

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    return train_df, val_df, test_df, num_classes, normalized_class_weights

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
        data_fraction: float,
    ) -> None:

    # get start time
    start_time = time.time()

    train_df, val_df, test_df, num_classes, normalized_class_weights = load_data(dataset,
                                              data_path=Path(data_path),
                                              val_percentage=0.2,
                                              seed=42,
                                              data_fraction=1.0,
                                              )

    best_n_trials = method(max_epochs=epochs,
           seed=seed,
           output_path=output_path,
           train_df=train_df,
           val_df=val_df,
           num_classes=num_classes,
           normalized_class_weights=normalized_class_weights,
           dataset=dataset,
           get_best_n_trials=3,  # Get this amount of best trials
           load_study=True,
           load_study_name="optuna_4"
           )
    
    # Print total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    # turn it to minute and seconds
    elapsed_minutes = elapsed_time // 60
    elapsed_seconds = elapsed_time % 60
    print(f"Total execution time: {elapsed_minutes:.0f} minutes and {elapsed_seconds:.2f} seconds")
    
    for trial in best_n_trials:
        automl = TextAutoML(
            normalized_class_weights=normalized_class_weights,
            seed=seed,
            token_length=trial["params"]["token_length"],
            max_epochs=epochs,
            batch_size=64,
            lr=trial["params"]["lr"],
            weight_decay=trial["params"]["weight_decay"],
            train_df=train_df,
            val_df=val_df,
            )

        # go one dir lower from trial["load_path"]
        best_model_load_path = trial["load_path"] / "best_version"
        automl.load_model(best_model_load_path)

        # Predict on the test set
        test_preds, test_labels = automl.predict(test_df)

        # Write the predictions of X_test to disk
        print("Writing predictions to disk")
        with (output_path / f"trial_{trial['trial_id']}_score.yaml").open("w") as f:
            yaml.safe_dump({"val_err": float(trial["best_val_err"])}, f)
        trial_id = trial['trial_id']
        print(f"Saved validation score at {output_path / f'trial_{trial_id}_score.yaml'}")
        with (output_path / f"trial_{trial_id}_test_preds.npy").open("wb") as f:
            np.save(f, test_preds)
        print(f"Saved test prediction at {output_path / f'trial_{trial_id}_test_preds.npy'}")

        # # In case of running on the final exam data, also add the predictions.npy
        # # to the correct location for auto evaluation.
        # if dataset == FINAL_TEST_DATASET: 
        #     test_output_path = output_path / "predictions.npy"
        #     test_output_path.parent.mkdir(parents=True, exist_ok=True)
        #     with test_output_path.open("wb") as f:
        #         np.save(f, test_preds)

        # Check if test_labels has missing data
        if not np.isnan(test_labels).any():
            acc = accuracy_score(test_labels, test_preds)
            print(f"Accuracy on test set: {acc}")

            with (output_path / "score.yaml").open("a+") as f:
                yaml.safe_dump({"test_err": float(1-acc)}, f)
            
            # Log detailed classification report for better insight
            print("Classification Report:")
            report = classification_report(test_labels, test_preds)
            print(f"\n{report}")
        else:
            # This is the setting for the exam dataset, you will not have access to the labels
            print(f"No test labels available for dataset '{dataset}'")

        # add total execution time to the results file, also make the results file better.
        with (output_path / "score.yaml").open("a+") as f:
            yaml.safe_dump({"total_execution_time": elapsed_time}, f)

    return


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

    main_loop(
        dataset=dataset,
        output_path=Path(output_path).absolute(),
        data_path=Path(data_path).absolute(),
        seed=seed,
        token_length=128,
        epochs= 15,
        batch_size=16,
        lr=5e-5,
        weight_decay=0.01,
        data_fraction=1.0, # "Subsampling of training set, in fraction (0, 1]. 1 is all the data"
        val_percentage = 0.2,
    )

# end of file