def evaluate(step, width, height, activation):
    time.sleep(0.1)
    activation_boost = 10 if activation=="relu" else 1
    return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost

def objective(config):
    start = 0
    if tune.get_checkpoint():
        with tune.get_checkpoint().as_directory() as checkpoint_dir:
            start = int((Path(checkpoint_dir) / "data.ckpt").read_text())

    for step in range(start, config["steps"]):
        score = evaluate(step, config["width"], config["height"], config["activation"])
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            (Path(checkpoint_dir) / "data.ckpt").write_text(str(step))
            tune.report(
                {"iterations": step, "mean_loss": score},
                checkpoint=tune.Checkpoint.from_directory(checkpoint_dir)
            )

search_space = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
    "activation": tune.choice(["relu", "tanh"]),
}

algo = TuneBOHB()
algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)
scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=100,
    reduction_factor=4,
    stop_last_trials=False,
)

num_samples = 1000

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=num_samples,
    ),
    run_config=tune.RunConfig(
        name="bohb_exp",
        stop={"training_iteration": 100},
    ),
    param_space=search_space,
)
results = tuner.fit()