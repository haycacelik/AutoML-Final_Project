# first layer 
import optuna
from pathlib import Path
from automl.automl_methods.nas.optuna_nas import tpe_objective
print("Starting the AutoML run...")  # Debugging line to check if the script starts correctly
import optuna
print("Optuna imported successfully")  # Debugging line to check if Optuna imports correctly

from automl.optuna_core import TextAutoML
import torch
import gc
import wandb
from functools import partial

def tpe_objective(trial, epochs, seed, train_df, val_df, num_classes, output_path, normalized_class_weights):
    # TODO use this with the paths
    trial_id = trial.number

    hidden_dim = trial.suggest_categorical(f"hidden_size", [64, 128, 256])
    activation = trial.suggest_categorical("activation_function", ["ReLU", "GELU", "LeakyReLU"])
    hidden_layer = trial.suggest_int("hidden_layers", 1, 4)
    use_layer_norm = trial.suggest_categorical("use_layer_norm", [True, False])
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    token_length = trial.suggest_categorical("token_length", [64, 128, 256, 512])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 0.1, log=True)
    amount_of_layers_to_finetune = trial.suggest_float("amount_of_layers_to_finetune", 0.0, 1.0)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    batch_size = 64

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
    )
    automl.create_model(
        fraction_layers_to_finetune=amount_of_layers_to_finetune,
        classification_head_hidden_dim=hidden_dim,
        classification_head_dropout_rate=dropout_rate,
        classification_head_hidden_layers=hidden_layer,
        classification_head_activation=activation,
        num_classes=num_classes,
        use_layer_norm=use_layer_norm

    )

    # val_err = automl.fit(output_path = output_path)
    train_accuracies, val_accuracies, val_err = automl.fit(output_path = output_path)

    # because we want to make a plot once its all over
    trial.set_user_attr("train_accuracies", train_accuracies)
    trial.set_user_attr("val_accuracies", val_accuracies)

    # Clean up automl object to free memory
    del automl
    gc.collect()  # Force garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return val_err

def sh_objective(trial, epochs, seed, train_df, val_df, num_classes, output_path, normalized_class_weights):
    
    

    def objective(trial):
        # Select a fixed config index from a trial parameter
        idx = trial.suggest_int("config_idx", 0, len(configs)-1)
        config = configs[idx]

        for step in range(10):
            # Dummy validation error decreases as epochs increase
            val_error = 1.0 / ((step + 1) * config["lr"] * 100)
            trial.report(val_error, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_error

    study = optuna.create_study(pruner=optuna.pruners.SuccessiveHalvingPruner())

    study.optimize(objective, n_trials=len(configs))

def method(max_epochs, seed, output_path, train_df, val_df, num_classes, normalized_class_weights, dataset):
    for i in range (0,10):
        optuna_study_name = f"optuna_{i+1}"
        # check if folder exists
        optuna_study_path = output_path / optuna_study_name
        if not optuna_study_path.exists():
            optuna_study_path.mkdir(parents=True, exist_ok=True)
            print(f"Created new study folder: {optuna_study_path}")
            break
        if i == 9:
            raise ValueError(f"You already have 10 in {output_path}, please remove some.")
        
    wandb_run = wandb.init(
        project="text-automl",
        name=f"nas_{dataset}_seed{seed}_{optuna_study_name}",
        config={
            "dataset": dataset,
            "seed": seed,
            "epochs": max_epochs,
            "train_size": len(train_df),
            "val_size": len(val_df) if val_df is not None else 0,
            "num_classes": num_classes
        },
        tags=[dataset, "distilbert", "text-classification", "hpo"]  # Add tags for easy filtering
    )

    ## START OF LAYER 1
    # Create a TPESampler with custom parameters
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=24,   # Number of random trials before using TPE
        seed=42,
        multivariate=True,  # Enable multivariate sampling
    )

    objective_fn = partial(
        tpe_objective,
        epochs=1,
        seed=seed,
        train_df=train_df,
        val_df=val_df,
        num_classes=num_classes,
        output_path=optuna_study_path,
        normalized_class_weights=normalized_class_weights,
    )

    # Load or create a study with completed 16 trials
    study_first_layer = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=optuna_study_name,
        storage=f"sqlite:///{optuna_study_path / 'study.db'}",
        )

    study_first_layer.optimize(objective_fn, n_trials=32)

    # 32 trials have been completed
    # Now: ask TPE for 8 new configurations
    new_trials = []
    for _ in range(8):
        trial = study_first_layer.ask()  # This uses the TPE sampler now
        params = trial.params  # These are your new sampled params
        new_trials.append(params)

    # Optionally print
    for i, p in enumerate(new_trials):
        print(f"New Config {i+1}: {p}")

    top_trials = study_first_layer.best_trials[:8]  # Top 8 trials by objective value
    for i, trial in enumerate(top_trials):
        print(f"Top {i+1} Trial:")
        print(f"  Value: {trial.value}")
        print(f"  Params: {trial.params}")


    # for all trials get their val_accuracies list
    all_train_accuracies = {}
    all_val_accuracies = {}
    for trial in study_first_layer.trials:
        # if trial.state == optuna.trial.TrialState.COMPLETE:
        train_accuracies = trial.user_attrs.get("train_accuracies", [])
        val_accuracies = trial.user_attrs.get("val_accuracies", [])
        all_train_accuracies[trial.number] = train_accuracies
        all_val_accuracies[trial.number] = val_accuracies

    ### START OF SECOND PHASE ### 16 samples will be trained for 2 epochs
    # from here just succesive halving
    # new_trials is a list of parameters for the new trials
    results = []
    # now evaluate the new trials
    for new_trial_idx, param in enumerate(new_trials):
        trial_idx = study_first_layer.trials[-1].number + new_trial_idx + 1  # New trial index
        # create new model with the new parameters
        automl = TextAutoML(
        normalized_class_weights=normalized_class_weights,
        seed=seed,
        token_length= param["token_length"],
        max_epochs= 4,
        batch_size= param["batch_size"],
        lr= param["lr"],
        weight_decay= param["weight_decay"],
        train_df=train_df,
        val_df=val_df,
        )
        automl.create_model(
            fraction_layers_to_finetune= param["amount_of_layers_to_finetune"],
            classification_head_hidden_dim= param["hidden_dim"],
            classification_head_dropout_rate= param["dropout_rate"],
            classification_head_hidden_layers= param["hidden_layers"],
            classification_head_activation= param["activation"],
            num_classes=num_classes,
            use_layer_norm= param["use_layer_norm"]
        )
        train_accuracies, val_accuracies, val_err = automl.fit(output_path = output_path)
        results.append((new_trial_idx, val_err))
        all_train_accuracies[trial_idx] = train_accuracies
        all_val_accuracies[trial_idx] = val_accuracies

    for idx, trial in enumerate(top_trials):
        # continue training the top trials
        trial_id = trial.number
        params = trial.params

        # TODO load the model with the parameters of the trial
        automl = TextAutoML(
        normalized_class_weights=normalized_class_weights,
        seed=seed,
        token_length= param["token_length"],
        max_epochs= 4,
        batch_size= param["batch_size"],
        lr= param["lr"],
        weight_decay= param["weight_decay"],
        train_df=train_df,
        val_df=val_df,
        )

        automl.load_model(temp_dir= load_path)
        train_accuracies, val_accuracies, val_err = automl.fit(output_path = output_path)
        results.append((trial_id, val_err))
        # append the train accuracies to the all_train_accuracies dict
        all_train_accuracies[trial_id].append(train_accuracies)
        all_val_accuracies[trial_id].append(val_accuracies)

    # Sort results by validation error, low to high
    results.sort(key=lambda x: x[1])  # Sort by validation error (second element)
    
    # Select the top 8 trials based on validation error
    top_trials = results[:8]

    ## END OF SECOND PHASE ##
    ## START OF THIRD PHASE ##

### THIRD PHASE
### FOURTH PHASE