
from automl.optuna_core import TextAutoML
import optuna
from functools import partial
import torch
import gc

def objective(trial, epochs, seed, train_df, val_df, num_classes, output_path, normalized_class_weights, last_id, config_count):
    # +1 bucause both trail.number and last_id start at 0 so its 1 too little
    #   40         16         39       - 16    + 1
    #   41         17         39        -16    +1
    #   45         8          44       - 8     + 1
    trial_id = trial.number + last_id - config_count + 1
    print("compiling new trial with id", trial_id)
    # hidden_dim = trial.suggest_categorical(f"hidden_size", [64, 128, 256])
    # activation = trial.suggest_categorical("activation_function", ["ReLU", "GELU", "LeakyReLU"])
    # hidden_layer = trial.suggest_int("hidden_layers", 1, 4)
    # use_layer_norm = trial.suggest_categorical("use_layer_norm", [True, False])

    hidden_dim = 128
    activation = "ReLU"
    hidden_layer = 2
    use_layer_norm = True

    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    token_length = trial.suggest_categorical("token_length", [64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 0.1, log=True)
    amount_of_layers_to_finetune = trial.suggest_int("amount_of_layers_to_finetune", 0,6)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    batch_size = 128

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
    val_accuracies, val_err = automl.fit(save_dir = output_path / f"trial_{trial_id}")

    # because we want to make a plot once its all over
    trial.set_user_attr("val_accuracies", val_accuracies)

    # Clean up automl object to free memory
    del automl
    gc.collect()  # Force garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return val_err

class Ensemble_SV:
    def __init__(self, 
                all_trials, 
                budget, 
                reduction_factor=2, 
                last_trial_id=None, 
                distributions=None, 
                total_layer_budget=32, 
                ):
        """
        configs: list of configs
        budgets: budgets for each sh (e.g. [2, 4, 8, 16])
        the budget: the number of epoch for each layer in the successive halving (32)
        reduction_factor: fraction of configs to keep after each round (e.g. 2 means keep half)
        load_study_names: list of study names to load, if not given it will create a new study [None, None, None, None]
        """
        # in config have the trial_id, load_path and params
        self.all_trials = all_trials
        self.budgets = budget
        self.total_layer_budget = total_layer_budget
        self.reduction_factor = reduction_factor
        self.last_trial_id = last_trial_id
        self.distributions = distributions

    def successive_halving(self, 
                           normalized_class_weights, 
                           seed, 
                           train_df, 
                           val_df, 
                           optuna_study_path, 
                           num_classes, 
                           optuna_study_name):
        """
        configs: list of configs
        budgets: list of budgets to try, increasing (e.g. [2,4,8,16])
        reduction_factor: fraction of configs to keep after each round (e.g. 2 means keep half)
        """
        results = []
        for layer_idx, budget in enumerate(self.budgets):
            # if the budget is 32 and budget for this round is 
            config_count = int(self.total_layer_budget / budget)
            results = []  # Reset results for each budget
            # if its not the first budget eliminate half of the configs
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=20,   # Number of random trials before using TPE
                seed=42,
                multivariate=True,  # Enable multivariate sampling
            )
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                study_name=optuna_study_name,
                storage=f"sqlite:///{optuna_study_path / f'study_layer_{layer_idx+2}.db'}",
                )
            
            objective_fn = partial(
                objective,
                epochs=budget,
                seed=seed,
                train_df=train_df,
                val_df=val_df,
                num_classes=num_classes,
                output_path=optuna_study_path,
                normalized_class_weights=normalized_class_weights,
                last_id=self.last_trial_id,
                config_count=config_count,
            )
            for trial in self.all_trials.values():
                if trial["stopped"] != -1:
                    print(f"Skipping trial {trial['trial_id']} with best val {trial['best_val_err']} as it has already been stopped.")
                    continue

                automl = TextAutoML(
                    normalized_class_weights=normalized_class_weights,
                    seed=seed,
                    token_length=trial["params"]["token_length"],
                    max_epochs=budget,
                    batch_size=128,
                    lr=trial["params"]["lr"],
                    weight_decay=trial["params"]["weight_decay"],
                    train_df=train_df,
                    val_df=val_df,
                    )
                
                automl.load_model(trial["load_path"])
                # this is for when we are loading a model that has already been trained for some epochs
                val_accuracies, val_err = automl.fit(save_dir=optuna_study_path / f"trial_{trial['trial_id']}")
                results.append((trial["trial_id"], val_err, val_accuracies))

                # change the config to have the val_accuracies
                # TODO i think i should have done extend insteaad of append because print("val_accuracies", val_accuracies) gives val_accuracies [(1, 0.8726)]
                trial["val_accuracies"].extend(val_accuracies)
                print(f"val_accuracies: {trial['val_accuracies']}")
                trial["best_val_err"] = val_err
                trial["load_path"] = optuna_study_path / f"trial_{trial['trial_id']}"

                calculated_trial = optuna.trial.create_trial(
                    params=trial["params"],
                    distributions=self.distributions,
                    value=val_err,
                )
                study.add_trial(calculated_trial)

            # every time we evaluate, print the results of all trials
            for trial in self.all_trials.values():
                print(f"Trial {trial['trial_id']} - Best Val Err: {trial['best_val_err']} - Stopped: {trial['stopped']} - vall_accuracies: {trial['val_accuracies']}")

            # ELIMINATION PHASE OF EACH BUDGET
            # Sort results by validation error (lower is better)
            results.sort(key=lambda x: x[1])
            print("Results after training for budget {}:".format(budget))

            # Keep top fraction of trials based on validation error
            sample = True
            # if self.reduction_factor*2 > config_count:
            #     n_to_keep = 1
            #     sample = False
            # if self.reduction_factor*2 == config_count:
            #     n_to_keep = 2
            #     sample = False
            # else:
            n_to_keep = config_count / (self.reduction_factor*2)
            if n_to_keep == 1:
                sample = False
            if n_to_keep < 1:
                return self.all_trials
            n_to_keep = int(n_to_keep)

            print("---------------n to keep----------------- ", n_to_keep)
            top_trials = results[:n_to_keep]
            print(f"Keeping top {n_to_keep} trials")
            top_trials_ids = [trial[0] for trial in top_trials]

            for one_trial in self.all_trials.values():
                # if trial is still running, not in the top trials, stop it
                if one_trial["trial_id"] not in top_trials_ids and one_trial["stopped"] == -1:
                    one_trial["stopped"] = budget

            if sample:
                # next budget sampling phase, samples and optimizes n to keep trials
                print(f"at layer {layer_idx+1}, sampling {n_to_keep} trials from {config_count} trials")
                study.optimize(objective_fn, n_trials=n_to_keep)
                for study_trial in study.trials:
                    if study_trial.state == optuna.trial.TrialState.COMPLETE and study_trial.number > config_count - 1:
                        self.last_trial_id += 1
                        self.all_trials[f"trial_{self.last_trial_id}"] = {
                            "trial_id": self.last_trial_id, # if trial id is lower than 24 it random, every one after that is a tpe sampled
                            "load_path": optuna_study_path / f"trial_{self.last_trial_id}",
                            "params": study_trial.params,
                            "val_accuracies": study_trial.user_attrs.get("val_accuracies", []),
                            "best_val_err": study_trial.value,
                            "stopped": -1
                        }

        return self.all_trials
    
    def get_best_trial(self):
        """
        Get the best trial from all trials
        """
        best_trial = None
        best_val_err = float('inf')
        for trial in self.all_trials.values():
            if trial["stopped"] == -1 and trial["best_val_err"] < best_val_err:
                best_val_err = trial["best_val_err"]
                best_trial = trial
        return best_trial