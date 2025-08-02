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
from automl.bohb_core import TextAutoML


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def short_trial_name(trial: Trial):
    config = trial.config
    dataset = config["dataset"]
    lr = config["lr"]
    wd = config["weight_decay"]
    finetune = config["amount_of_layers_to_finetune"]
    dropout = config["dropout_rate"]

    return f"{dataset}_lr{lr:.1e}_wd{wd:.1e}_ft{finetune}_dr{dropout:.2f}_{trial.trial_id}"

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


# 3. Define training function
def train_model(config, data):

    automl = TextAutoML(
        normalized_class_weights=data["normalized_class_weights"],
        seed=config["seed"],
        token_length=int(config["token_length"]),
        max_epochs=config["epochs"],
        batch_size=int(config["batch_size"]),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        train_df=data["train_df"],
        val_df=data["val_df"],
    )
    
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            print("loading model from checkpoint")
            automl.load_model(temp_dir = Path(checkpoint_dir))
            # model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
            # optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
            # start_epoch = torch.load(os.path.join(checkpoint_dir, "extra_state.pt"))["epoch"] + 1
    else:
        # If no checkpoint, start from scratch
        print("no checkpoint for this trial, creating a new model")
        automl.create_model(
            fraction_layers_to_finetune=config["amount_of_layers_to_finetune"],
            num_classes=data["num_classes"],
            classification_head_hidden_dim=int(config["hidden_dim"]),
            classification_head_dropout_rate=config["dropout_rate"],
            classification_head_hidden_layers=int(config["hidden_layers"]),
            classification_head_activation=config["activation"],
            use_layer_norm=config["use_layer_norm"]
        )

    # Fit the AutoML model on the training and validation datasets
    automl.fit()


def BOHB(dataset, hidden_dim, hidden_layers, activation, use_layer_norm):

    # Define ConfigSpace search space
    config_space = CS.ConfigurationSpace()
    config_space.add([
        # NAS hyperparameters
        CS.Constant("hidden_dim", hidden_dim),
        CS.Constant("hidden_layers", hidden_layers),
        CS.Constant("activation", activation),
        CS.Constant("use_layer_norm", use_layer_norm),
        CS.Constant("epochs", 10),
        CS.Constant("dataset", dataset),
        CS.Constant("seed", 42),
        CS.Constant("val_percentage", 0.2),
        CS.Constant("output_path", "results"),
        CS.Constant("data_path", "data"),
        CS.Constant("data_fraction", 1.0),
        CS.Constant("batch_size", 64), # its better to use the biggest possible batch size for your GPU
        CS.CategoricalHyperparameter("token_length", [64, 128, 256, 512]),
        # CS.CategoricalHyperparameter("batch_size", [16, 32, 64]),
        CS.UniformFloatHyperparameter("weight_decay", 1e-6, 0.1, log=True),
        CS.UniformFloatHyperparameter("lr", 1e-6, 1e-3, log=True),
        CS.UniformIntegerHyperparameter("amount_of_layers_to_finetune", 0, 6),
        CS.UniformFloatHyperparameter("dropout_rate", 0.1, 0.5)
    ])

    # 2. Define scheduler and search algorithm
    bohb_search = TuneBOHB(space=config_space, metric="val_acc", mode="max")
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=1)
    bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", 
                                      metric="val_acc", 
                                      max_t=10,
                                      mode="max", 
                                      reduction_factor=3,
                                      )

    train_df, val_df, num_classes, normalized_class_weights = load_data("imdb",
                                              data_path=Path(PROJECT_ROOT / "data"),
                                              val_percentage=0.2,
                                              seed=42,
                                              data_fraction=1.0,
                                              )
    print("running BOHB with the following parameters:")

    # do wandb 
    wandb_run = wandb.init(
    project="text-automl",
    name=tune.get_context().get_trial_name(),
    config={"dataset": dataset, "epochs": 10, "hidden_dim": hidden_dim, "hidden_layers": hidden_layers, "activation": activation, "use_layer_norm": use_layer_norm},
    tags=[dataset, "distilbert", "text-classification"],
    )

    # 4. Run BOHB
    tune.run(
        tune.with_parameters(train_model, data={"train_df": train_df, "val_df": val_df, "num_classes": num_classes, "normalized_class_weights": normalized_class_weights}),
        name="bohb_exp",
        search_alg=bohb_search,
        scheduler=bohb_scheduler,
        num_samples=20, # this is the sum of all chosen hyperparameter configurations, if its 10 per succesive halving run then there will be 2 succesive halving runs
        resources_per_trial={"cpu": 10, "gpu": 1},
        storage_path=str(PROJECT_ROOT / "experiments/bohb_results"),
        trial_dirname_creator=short_trial_name,
        trial_name_creator=short_trial_name,
        verbose=2,  # Set verbose=2 to see trial logs in console
        log_to_file=True,  # Enable logging to files
    )

    


if __name__ == "__main__":
    BOHB("imdb", hidden_dim=128, hidden_layers=2, activation="ReLU", use_layer_norm=True)

