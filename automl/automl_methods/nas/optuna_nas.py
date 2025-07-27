
from automl.core import TextAutoML
import torch
import gc


def objective(trial, epochs, lr, batch_size, seed, token_length,
                     weight_decay,
                     train_df, val_df, num_classes, output_path, normalized_class_weights, wandb_run):
    trial_id = trial.number

    hidden_dim = trial.suggest_categorical(f"hidden_size", [64, 128, 256])
    activation = trial.suggest_categorical("activation_function", ["ReLU", "GELU", "LeakyReLU"])
    # use_norm = trial.suggest_categorical("use_normalization", [True, False])
    # i want to add fine tuning here as well but its always going to be better, so maybe we should turn this into a mutli objective
    fraction_layers_to_finetune = trial.suggest_float("fraction_layers_to_finetune", 0.0, 1.0)
    # fraction_layers_to_finetune = 0.0  # Fixed for now, can be tuned later
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    hidden_layer = trial.suggest_int("hidden_layers", 1, 4)
    use_layer_norm = trial.suggest_categorical("use_layer_norm", [True, False])

    print(f"Running trial {trial_id} with hidden_dim={hidden_dim}, activation={activation}, dropout_rate={dropout_rate}, hidden_layer={hidden_layer}, use_layer_norm={use_layer_norm}")

    automl = TextAutoML(
        normalized_class_weights=normalized_class_weights,
        seed=seed,
        token_length=token_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        train_df=train_df,
        val_df=val_df,
        save_path=output_path / f"trial_{trial_id}",
        wandb_logger=wandb_run,
    )
    automl.create_model(
        fraction_layers_to_finetune=fraction_layers_to_finetune,
        classification_head_hidden_dim=hidden_dim,
        classification_head_dropout_rate=dropout_rate,
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

