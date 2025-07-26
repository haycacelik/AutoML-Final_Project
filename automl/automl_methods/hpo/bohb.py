import os
from pathlib import Path
import wandb
from ray import tune
from ray.tune.experiment import Trial
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS
from automl import AGNewsDataset, IMDBDataset, AmazonReviewsDataset, DBpediaDataset
from automl.datasets import get_fraction_of_data
from automl import TextAutoML


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def short_trial_name(trial: Trial):
    config = trial.config
    dataset = config["dataset"]
    bs = config["batch_size"]
    ep = config["epochs"]
    lr = config["lr"]
    dim = config["classification_head_hidden_dim"]

    return f"{dataset}_bs{bs}_ep{ep}_lr{lr:.1e}_dim{dim}_{trial.trial_id}"

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

    return train_df, val_df, num_classes


# 3. Define training function
def train_model(config, data):
    output_path = PROJECT_ROOT / config["output_path"]

    wandb_run = wandb.init(
        project="text-automl",
        name=tune.get_context().get_trial_name(),
        config={**config, "dataset": config["dataset"], "epochs": config["epochs"]},
        tags=[config["dataset"], "distilbert", "text-classification"],  # Add tags for easy filtering,
    )

    automl = TextAutoML(
        normalized_class_weights=None,
        seed=config["seed"],
        token_length=int(config["token_length"]),
        epochs=config["epochs"],
        batch_size=int(config["batch_size"]),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        fraction_layers_to_finetune=config["fraction_layers_to_finetune"],
        classification_head_hidden_dim=int(config["classification_head_hidden_dim"]),
        classification_head_dropout_rate=config["classification_head_dropout_rate"],
        classification_head_hidden_layers=config["classification_head_hidden_layers"],
        train_df=data["train_df"],
        val_df=data["val_df"],
        num_classes=data["num_classes"],
        load_path=None,
        save_path=output_path,
        wandb_logger=wandb_run,
    )

    # Fit the AutoML model on the training and validation datasets
    automl.fit()


def BOHB(dataset):

    # Define ConfigSpace search space
    config_space = CS.ConfigurationSpace()
    config_space.add([
        CS.Constant("epochs", 10),
        CS.Constant("dataset", dataset),
        CS.Constant("seed", 42),
        CS.Constant("val_percentage", 0.2),
        CS.Constant("output_path", "results"),
        CS.Constant("data_path", "data"),
        CS.Constant("data_fraction", 1.0),
        CS.CategoricalHyperparameter("batch_size", [16, 32, 64]),
        CS.CategoricalHyperparameter("token_length", [64, 128, 256, 512]),
        CS.CategoricalHyperparameter("classification_head_hidden_dim", [32, 64, 128, 256]),
        CS.UniformFloatHyperparameter("classification_head_dropout_rate", 0.0, 0.5),
        CS.UniformIntegerHyperparameter("classification_head_hidden_layers", 1, 6),
        CS.UniformFloatHyperparameter("weight_decay", 1e-6, 0.1, log=True),
        CS.UniformFloatHyperparameter("lr", 1e-4, 1e-1, log=True),
        CS.UniformFloatHyperparameter("fraction_layers_to_finetune", 0.0, 1.0)
    ])

    # 2. Define scheduler and search algorithm
    bohb_search = TuneBOHB(space=config_space, metric="val_acc", mode="max")
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=1)
    bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", metric="val_acc", max_t=10, mode="max")

    train_df, val_df, num_classes = load_data("imdb",
                                              data_path=Path(PROJECT_ROOT / "data"),
                                              val_percentage=0.2,
                                              seed=42,
                                              data_fraction=1.0
                                              )

    # 4. Run BOHB
    tune.run(
        tune.with_parameters(train_model, data={"train_df": train_df, "val_df": val_df, "num_classes": num_classes}),
        name="bohb_exp",
        search_alg=bohb_search,
        scheduler=bohb_scheduler,
        num_samples=20,
        resources_per_trial={"cpu": 2, "gpu": 1},
        storage_path=str(PROJECT_ROOT / "experiments/bohb_results"),
        trial_dirname_creator=short_trial_name,
        trial_name_creator=short_trial_name,
    )

if __name__ == "__main__":
    BOHB("imdb")