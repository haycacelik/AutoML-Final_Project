# pip install ray[tune] ConfigSpace hpbandster
import os
from pathlib import Path
import wandb
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS
from ray.air import session

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def short_trial_name(trial):
    config = trial.config
    dataset = config["dataset"]
    bs = config["batch_size"]
    lr = config["lr"]
    dim = config["classification_head_hidden_dim"]
    epochs = config["epochs"]

    return f"{dataset}_bs{bs}_lr{lr:.1e}_dim{dim}_ep{epochs}_{trial.trial_id}"


def BOHB(dataset):

    # Define ConfigSpace search space
    config_space = CS.ConfigurationSpace()
    config_space.add([
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
        CS.UniformFloatHyperparameter("fraction_layers_to_finetune", 0.0, 1.0),
        CS.UniformIntegerHyperparameter("epochs", lower=1, upper=10)
    ])

    # 2. Define scheduler and search algorithm
    bohb_search = TuneBOHB(space=config_space, metric="val_error", mode="min")
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=1)
    bohb_scheduler = HyperBandForBOHB(time_attr="epochs", metric="val_error", mode="min")

    # 3. Define training function
    def train_model(config):
        from run import main_loop

        dataset = config["dataset"]

        output_path = PROJECT_ROOT / config["output_path"] / f"seed={config['seed']}"
        data_path = PROJECT_ROOT / config["data_path"]

        # Initialize wandb
        wandb_run = wandb.init(
            project="text-automl",
            name=session.get_trial_name(),
            config={**config, "dataset": dataset, "epochs": config["epochs"]},
            tags=[dataset, "distilbert", "text-classification"],  # Add tags for easy filtering,
        )

        val_err = main_loop(
            dataset=dataset,
            output_path=output_path,
            data_path=data_path,
            seed=config["seed"],
            token_length=int(config["token_length"]),
            batch_size=int(config["batch_size"]),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            data_fraction=config["data_fraction"],
            val_percentage=config["val_percentage"],
            fraction_layers_to_finetune=config["fraction_layers_to_finetune"],
            classification_head_hidden_dim=int(config["classification_head_hidden_dim"]),
            classification_head_dropout_rate=config["classification_head_dropout_rate"],
            classification_head_hidden_layers=config["classification_head_hidden_layers"],
            wandb_logger=wandb_run,
            epochs=config["epochs"]
        )

        session.report({"val_error": val_err})

    # 4. Run BOHB
    tune.run(
        train_model,
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