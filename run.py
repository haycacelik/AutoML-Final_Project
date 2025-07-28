import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import yaml
import wandb
import time
print("Starting the AutoML run...")  # Debugging line to check if the script starts correctly
import optuna
print("Optuna imported successfully")  # Debugging line to check if Optuna imports correctly
from functools import partial
import optuna.visualization as vis
# import matplotlib.pyplot as plt

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline

from automl.core import TextAutoML
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
    get_fraction_of_data,
)

FINAL_TEST_DATASET=...  # TBA later

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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

    # # TODO implement NAS
    # from automl.automl_methods.nas.optuna_nas import objective
    # n_trials = 15  # Number of trials to run
    # print(f"Running NAS with {n_trials} trials")

    # for i in range (0,10):
    #     nas_study_name = f"nas_{i+1}"
    #     # check if folder exists
    #     nas_study_path = output_path / nas_study_name
    #     if not nas_study_path.exists():
    #         nas_study_path.mkdir(parents=True, exist_ok=True)
    #         break
    #     if i == 9:
    #         raise ValueError(f"You already have 10 in {output_path}, please remove some.")

    # # a wand just for the NAS run, might no need it
    # wandb_nas_run = wandb.init(
    #     project="text-automl",
    #     name=f"nas_{dataset}_seed{seed}_{nas_study_name}",
    #     config={
    #         "sampler": "TPESampler",
    #         "n_trials": n_trials,
    #         "dataset": dataset,
    #         "seed": seed,
    #         "val_percentage": val_percentage,
    #         "token_length": token_length,
    #         "epochs": epochs,
    #         "batch_size": batch_size,
    #         "lr": lr,
    #         "weight_decay": weight_decay,
    #         "data_fraction": data_fraction,
    #         "train_size": len(train_df),
    #         "val_size": len(val_df) if val_df is not None else 0,
    #         "test_size": len(test_df),
    #         "num_classes": num_classes
    #     },
    #     tags=[dataset, "distilbert", "text-classification", "nas"]  # Add tags for easy filtering
    # )

    # # Create a TPESampler with custom parameters
    # sampler = optuna.samplers.TPESampler(
    #     n_startup_trials=8,   # Number of random trials before using TPE
    #     seed=42,
    #     multivariate=True,  # Enable multivariate sampling
    # )

    # objective_fn = partial(
    #     objective,
    #     epochs=epochs,
    #     lr=lr,
    #     batch_size=batch_size,
    #     seed=seed,
    #     token_length=token_length,
    #     weight_decay=weight_decay,
    #     train_df=train_df,
    #     val_df=val_df,
    #     num_classes=num_classes,
    #     output_path=nas_study_path,
    #     normalized_class_weights=normalized_class_weights,
    #     wandb_run=wandb_nas_run
    # )
        
    # study = optuna.create_study(
    #     direction="minimize",
    #     sampler=sampler,
    #     study_name=nas_study_name,
    #     storage=f"sqlite:///{nas_study_path / 'study.db'}",
    #     )
    # # study.optimize(objective_fn, n_trials=5, callbacks=[visualize_kernels])
    # study.optimize(objective_fn, n_trials=n_trials)

    # load study from the database
    nas_study_path = output_path / "nas_2"
    print("Loading study from:", nas_study_path)
    nas_study_load_path = nas_study_path / "study.db"
    study = optuna.load_study(
        study_name="nas_2",
        storage=f"sqlite:///{nas_study_load_path}",
    )

    val_err = 1 - study.best_value
    print("Best score:", 1 - study.best_value)
    print("Best params:", study.best_params)

    # get the path of the best model
    trial_id = study.best_trial.number
    print(f"Best trial ID: {trial_id}")
    folder = nas_study_path / f"trial_{trial_id}" / "best_model"
    # get the files in the folder
    files = list(folder.glob("*.pth"))
    for file in files:
        # get the file that ents with .pth
        if file.suffix == ".pth":
            model_path = file
            print(f"Best model found: {model_path}")
            break
    else:
        raise ValueError(f"No model found in {folder} with .pth suffix.")

    # # for running it once
    # wandb_run = wandb.init(
    #     project="text-automl",
    #     name=f"nas_{dataset}_hm{classification_head_activation}_do{classification_head_dropout_rate}_hl{classification_head_hidden_layers}_{classification_head_activation}",  # Custom run name
    #     config={
    #             "dataset": dataset,
    #             "seed": seed,
    #             "val_percentage": val_percentage,
    #             "token_length": token_length,
    #             "epochs": epochs,
    #             "batch_size": batch_size,
    #             "lr": lr,
    #             "weight_decay": weight_decay,
    #             "fraction_layers_to_finetune": 0.0,
    #             "data_fraction": data_fraction,
    #             "classification_head_hidden_dim": 64,
    #             "classification_head_dropout_rate": 0.2,
    #             "classification_head_hidden_layers": 4,
    #             "classification_head_activation": 'LeakyReLU',
    #             "train_size": len(train_df),
    #             "val_size": len(val_df) if val_df is not None else 0,
    #             "test_size": len(test_df),
    #             "num_classes": num_classes
    #         },
    #         tags=[dataset, "distilbert", "text-classification", "nas"]  # Add tags for easy filtering
    #     )


    # Initialize the TextAutoML instance with the best parameters
    automl = TextAutoML(
        normalized_class_weights=normalized_class_weights,
        # normalized_class_weights=None,
        seed=seed,
        token_length=token_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        train_df=train_df,
        val_df=val_df,
        save_path=output_path,
        wandb_logger=None, # 
    )
    # # if you want to create a new model to train
    # automl.create_model(
    #     fraction_layers_to_finetune=0.0,
    #     num_classes=num_classes,
    #     classification_head_hidden_dim=64,
    #     classification_head_dropout_rate=0.2,
    #     classification_head_hidden_layers=4,  # [1,4]
    #     classification_head_activation='LeakyReLU',  # Default activation, can be changed later
    # )

    # if you want to load a pre-trained model
    automl.load_model(model_path=model_path)

    # Fit the AutoML model on the training and validation datasets, for single run
    # val_err = automl.fit()

    # Print total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    # turn it to minute and seconds
    elapsed_minutes = elapsed_time // 60
    elapsed_seconds = elapsed_time % 60
    print(f"Total execution time: {elapsed_minutes:.0f} minutes and {elapsed_seconds:.2f} seconds")
    # Log total execution time to wandb
    # wandb_nas_run.log({"total_execution_time": elapsed_time})

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

    ### no need because its just 1 - accuracy
    # # Log validation error to wandb
    # wandb_run.log({"val_error": float(val_err)})

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
        
        with (output_path / "score.yaml").open("a+") as f:
            yaml.safe_dump({"test_err": float(1-acc)}, f)
        
        # Log detailed classification report for better insight
        print("Classification Report:")
        report = classification_report(test_labels, test_preds)
        print(f"\n{report}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        print(f"No test labels available for dataset '{dataset}'")

    # Log total execution time to wandb
    # TODO add it to the results file, also make the results file better.
    with (output_path / "score.yaml").open("a+") as f:
        yaml.safe_dump({"total_execution_time": elapsed_time}, f)

    # Save plots instead of showing them (since we're in a non-interactive environment)
    plots_dir = nas_study_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("Saving optimization plots...")
    
    try:
        fig = vis.plot_optimization_history(study)
        fig.write_html(plots_dir / "optimization_history.html")
        print("Saved optimization history plot")
    except Exception as e:
        print(f"Error saving optimization history: {e}")
    
    try:
        fig = vis.plot_contour(study)
        fig.write_html(plots_dir / "contour.html") 
        print("Saved contour plot")
    except Exception as e:
        print(f"Error saving contour plot: {e}")
    
    try:
        fig = vis.plot_edf(study)
        fig.write_html(plots_dir / "edf.html")
        print("Saved EDF plot")
    except Exception as e:
        print(f"Error saving EDF plot: {e}")
    
    try:
        fig = vis.plot_intermediate_values(study)
        fig.write_html(plots_dir / "intermediate_values.html")
        print("Saved intermediate values plot")
    except Exception as e:
        print(f"Error saving intermediate values plot: {e}")
    
    try:
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(plots_dir / "parallel_coordinate.html")
        print("Saved parallel coordinate plot")
    except Exception as e:
        print(f"Error saving parallel coordinate plot: {e}")
    
    try:
        fig = vis.plot_param_importances(study)
        fig.write_html(plots_dir / "param_importances.html")
        print("Saved parameter importances plot")
    except Exception as e:
        print(f"Error saving parameter importances plot: {e}")
    
    try:
        fig = vis.plot_rank(study)
        fig.write_html(plots_dir / "rank.html")
        print("Saved rank plot")
    except Exception as e:
        print(f"Error saving rank plot: {e}")
    
    try:
        fig = vis.plot_slice(study)
        fig.write_html(plots_dir / "slice.html")
        print("Saved slice plot")
    except Exception as e:
        print(f"Error saving slice plot: {e}")
    
    try:
        fig = vis.plot_param_importances(
            study, target=lambda t: t.duration.total_seconds(), target_name="duration"
        )
        fig.write_html(plots_dir / "param_importances_duration.html")
        print("Saved parameter importances (duration) plot")
    except Exception as e:
        print(f"Error saving parameter importances duration plot: {e}")
    
    try:
        fig = vis.plot_timeline(study)
        fig.write_html(plots_dir / "timeline.html")
        print("Saved timeline plot")
    except Exception as e:
        print(f"Error saving timeline plot: {e}")
    
    print(f"All plots saved to {plots_dir}")
    print("Note: Open the .html files in a web browser to view the interactive plots")

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

    main_loop(
        dataset=dataset,
        output_path=Path(output_path).absolute(),
        data_path=Path(data_path).absolute(),
        seed=seed,
        token_length=128,
        epochs= 15,
        batch_size=16,  # Reduced from 32 to avoid CUDA memory errors
        lr=5e-5,
        weight_decay=0.01,
        data_fraction=1.0, # "Subsampling of training set, in fraction (0, 1]. 1 is all the data"
        val_percentage = 0.2,
    )

# end of file