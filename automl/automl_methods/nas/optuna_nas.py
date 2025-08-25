
from automl.optuna_core import TextAutoML
import torch
import gc
import wandb

def objective(trial, epochs, lr, batch_size, seed, token_length,
                     weight_decay,
                     train_df, val_df, num_classes, output_path, normalized_class_weights, wandb_run):
    trial_id = trial.number

    hidden_dim = trial.suggest_categorical(f"hidden_size", [64, 128, 256])
    activation = trial.suggest_categorical("activation_function", ["ReLU", "GELU", "LeakyReLU"])
    hidden_layer = trial.suggest_int("hidden_layers", 1, 4)
    use_layer_norm = trial.suggest_categorical("use_layer_norm", [True, False])

    print(f"Running trial {trial_id} with hidden_dim={hidden_dim}, activation={activation}, hidden_layer={hidden_layer}, use_layer_norm={use_layer_norm}")

    automl = TextAutoML(
        normalized_class_weights=normalized_class_weights,
        seed=seed,
        token_length=token_length,
        max_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        train_df=train_df,
        val_df=val_df,
        save_path=output_path / f"trial_{trial_id}",
        wandb_logger=wandb_run,
    )
    automl.create_model(
        fraction_layers_to_finetune=0.0,
        classification_head_hidden_dim=hidden_dim,
        classification_head_dropout_rate=0.1,
        classification_head_hidden_layers=hidden_layer,
        classification_head_activation=activation,
        num_classes=num_classes,
        use_layer_norm=use_layer_norm
    )

    val_err = automl.fit()
    
    # Clean up automl object to free memory
    del automl
    gc.collect()  # Force garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return val_err

def run_nas(output_path, dataset, seed):
    
    # implement NAS
    from automl.automl_methods.nas.optuna_nas import objective
    n_trials = 15  # Number of trials to run
    print(f"Running NAS with {n_trials} trials")

    for i in range (0,10):
        nas_study_name = f"nas_{i+1}"
        # check if folder exists
        nas_study_path = output_path / nas_study_name
        if not nas_study_path.exists():
            nas_study_path.mkdir(parents=True, exist_ok=True)
            break
        if i == 9:
            raise ValueError(f"You already have 10 in {output_path}, please remove some.")

    # a wand just for the NAS run, might no need it
    wandb_nas_run = wandb.init(
        project="text-automl",
        name=f"nas_{dataset}_seed{seed}_{nas_study_name}",
        config={
            "sampler": "TPESampler",
            "n_trials": n_trials,
            "dataset": dataset,
            "seed": seed,
        },
        tags=[dataset, "distilbert", "text-classification", "nas"]  # Add tags for easy filtering
    )

    # Create a TPESampler with custom parameters
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=8,   # Number of random trials before using TPE
        seed=42,
        multivariate=True,  # Enable multivariate sampling
    )

    objective_fn = partial(
        objective,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        token_length=token_length,
        weight_decay=weight_decay,
        train_df=train_df,
        val_df=val_df,
        num_classes=num_classes,
        output_path=nas_study_path,
        normalized_class_weights=normalized_class_weights,
        wandb_run=wandb_nas_run
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=nas_study_name,
        storage=f"sqlite:///{nas_study_path / 'study.db'}",
        )
    # study.optimize(objective_fn, n_trials=5, callbacks=[visualize_kernels])
    study.optimize(objective_fn, n_trials=n_trials)

    # load study from the database
    # nas_study_path = output_path / "nas_2"
    # print("Loading study from:", nas_study_path)
    # nas_study_load_path = nas_study_path / "study.db"
    # study = optuna.load_study(
    #     study_name="nas_2",
    #     storage=f"sqlite:///{nas_study_load_path}",
    # )
