# pip install ray[tune] ConfigSpace hpbandster

from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
import ConfigSpace as CS


def use_BOHB(hyperparameters):

    # 1. Define ConfigSpace search space
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("lr", 1e-4, 1e-1, log=True))
    config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("batch_size", 16, 128))

    # 2. Define scheduler and search algorithm
    algo = TuneBOHB(config_space)
    scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=10)

    # 3. Define training function
    def train(config):
        for i in range(10):
            acc = train_model(config["lr"], config["batch_size"])  # your logic
            tune.report(mean_accuracy=acc)

    # 4. Run BOHB
    tune.run(
        train,
        name="bohb_exp",
        scheduler=scheduler,
        search_alg=algo,
        num_samples=20,
        resources_per_trial={"cpu": 1, "gpu": 0},
    )
