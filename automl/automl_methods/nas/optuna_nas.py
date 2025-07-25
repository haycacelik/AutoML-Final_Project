
from automl.core import TextAutoML
import wandb

def objective(trial, dataset, epochs, lr, batch_size, seed, val_percentage, token_length,
                     weight_decay, fraction_layers_to_finetune, data_fraction,
                     train_df, val_df, test_df, num_classes, load_path, output_path):
    # TODO add values to start with
    # TODO get an image of the kernels each time

    hidden_dim = trial.suggest_categorical(f"hidden_size", [64, 128, 256, 512, 1024])
    activation = trial.suggest_categorical("activation_function", ["ReLU", "GELU", "LeakyReLU"])
    # use_norm = trial.suggest_categorical("use_normalization", [True, False])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    hidden_layer = trial.suggest_int("hidden_layers", 1, 4)

    print(f"Running trial with hidden_dim={hidden_dim}, activation={activation}, dropout_rate={dropout_rate}, hidden_layer={hidden_layer}")

    #TODO can i give each run a wandb name with an int that show the loop
    wandb_run = wandb.init(
        project="text-automl",
        name=f"nas_{dataset}_hd{hidden_dim}_do{dropout_rate}_hl{hidden_layer}_{activation}",  # Custom run name
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

    val_err = automl.fit()

    return val_err


# # Create a TPESampler with custom parameters
# sampler = optuna.samplers.TPESampler(
#     n_startup_trials=10,   # Number of random trials before using TPE
#     gamma=0.25,            # Fraction of good trials for modeling (lower means more exploitation)
#     n_ei_candidates=24,    # Number of candidates sampled when optimizing acquisition function
#     multivariate=True,     # Model joint distribution of parameters
#     seed=42
# )

# study = optuna.create_study(direction="minimize", sampler=sampler)
# study.optimize(objective, n_trials=50)

# print("Best score:", 1 - study.best_value)
# print("Best params:", study.best_params)
