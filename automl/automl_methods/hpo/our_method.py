# first layer 
import optuna
from pathlib import Path
import time
print("Starting the AutoML run...")  # Debugging line to check if the script starts correctly
import optuna
print("Optuna imported successfully")  # Debugging line to check if Optuna imports correctly
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
    get_fraction_of_data,
)

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
                                                            use_class_weights=True)

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

    return train_df, val_df, num_classes, normalized_class_weights

# Load or create a study with completed 16 trials
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=8,  # after 8 trials, it uses TPE
        seed=42,
        multivariate=True
    ),
    direction="maximize"  # or "minimize"
)

# Assume 16 trials have already been completed
# (You can load from storage or run them before this)

# Now: ask TPE for 4 new configurations
new_trials = []
for _ in range(4):
    trial = study.ask()  # This uses the TPE sampler now
    params = trial.params  # These are your new sampled params
    new_trials.append(params)

# Optionally print
for i, p in enumerate(new_trials):
    print(f"Config {i+1}: {p}")


top_trials = study.best_trials[:4]  # Top 4 trials by objective value
for i, trial in enumerate(top_trials):
    print(f"Top {i+1} Trial:")
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
