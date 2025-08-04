# first layer 
import optuna
from pathlib import Path
from automl.automl_methods.nas.optuna_nas import objective
print("Starting the AutoML run...")  # Debugging line to check if the script starts correctly
import optuna
print("Optuna imported successfully")  # Debugging line to check if Optuna imports correctly

from automl.optuna_core import TextAutoML
import torch
import gc
# import wandb
from functools import partial
import optuna.visualization as vis

def objective(trial, epochs, seed, train_df, val_df, num_classes, output_path, normalized_class_weights):
    trial_id = trial.number

    # the nas hyperparameters that we no longer use
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
    val_accuracies, val_err = automl.fit(save_dir = output_path / f"trial_{trial_id}")

    # because we want to make a plot once its all over
    trial.set_user_attr("val_accuracies", val_accuracies)

    # Clean up automl object to free memory
    del automl
    gc.collect()  # Force garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return val_err


def method(max_epochs, seed, output_path, train_df, val_df, num_classes, normalized_class_weights, dataset, get_best_n_trials, load_study_name=None, load_study=False):

    # if you want to load a study you must provide a load_study_name
    if load_study and load_study_name is None:
        raise ValueError("You need to provide a load_study_name to load a study.")

    # Create an output path for the study
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

    # # start the wandb run
    # wandb_run = wandb.init(
    #     project="text-automl-optuna",
    #     name=f"nas_{dataset}_seed{seed}_{optuna_study_name}",
    #     config={
    #         "dataset": dataset,
    #         "seed": seed,
    #         "epochs": max_epochs,
    #         "train_size": len(train_df),
    #         "val_size": len(val_df) if val_df is not None else 0,
    #         "num_classes": num_classes
    #     },
    #     tags=[dataset, "distilbert", "text-classification", "hpo"]  # Add tags for easy filtering
    # )

    # ## START OF LAYER 1 ##
    # in this layer the configs are all sampled from the tpe, the tpe is random until 24 configs are evaluated.
    # the other 8 are not random
    # Create a TPESampler with custom parameters
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=24,   # Number of random trials before using TPE
        seed=42,
        multivariate=True,  # Enable multivariate sampling
    )

    # this is for passing the parameters to the objective function
    objective_fn = partial(
        objective,
        epochs=1,
        seed=seed,
        train_df=train_df,
        val_df=val_df,
        num_classes=num_classes,
        output_path=optuna_study_path,
        normalized_class_weights=normalized_class_weights,
    )

    # Create a study, if we want to leadone we will take the studies from the load study
    study_first_layer = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=optuna_study_name,
        storage=f"sqlite:///{optuna_study_path / 'study_layer_1.db'}",
        )

    # if it is not loaded start optimizing 
    if load_study is False:
        study_first_layer.optimize(objective_fn, n_trials=32+8)
    # if it is loaded is should already be optimized
    else:
        # Load study from the database
        study_load_path = output_path / load_study_name / "study.db"
        old_study = optuna.load_study(
            study_name=load_study_name,
            storage=f"sqlite:///{study_load_path}",
        )
        for trial in old_study.trials:
            # this is only added because when i was debugging i accidentally addded trials with no value
            # normally there should be no trials with no value
            if trial.value is not None:
                study_first_layer.add_trial(trial)

    # The tpe phase 

    # 32 trials have been completed
    first_layer_trials = [trial for trial in study_first_layer.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.number <= 31]
    top_trials = sorted(first_layer_trials, key=lambda t: t.value)[:8]
    for trial in top_trials:
        print(f"Top Trial {trial.number}: Value = {trial.value}, Params = {trial.params}")

    all_trials = {}
    for trial in study_first_layer.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            all_trials[f"trial_{trial.number}"] = {
                "trial_id": trial.number, # if trial id is lower than 24 it random, every one after that is a tpe sampled
                "load_path": output_path / load_study_name / f"trial_{trial.number}" if load_study else optuna_study_path / f"trial_{trial.number}",
                "params": trial.params,
                "val_accuracies": trial.user_attrs.get("val_accuracies", []),
                "best_val_err": trial.value,
                "stopped": False if trial in top_trials or trial.number > 31 else 1
            }

    # make plots from what the tpe has seen so far
    plots_dir = optuna_study_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    # plots(study=study_first_layer, all_trials=all_trials, plots_dir=plots_dir)
    final_plot(all_trials, plots_dir)


    ### START OF SECOND PHASE ### 16 samples will be trained for 2 epochs
    # from here just succesive halving
    from automl.automl_methods.hpo.succesive_halving import SUCCESSIVE_HALVING

    successive_halving = SUCCESSIVE_HALVING(
        all_trials=all_trials,
        budget=[2, 4, 8, 16],  # Budgets for each round
        total_layer_budget=32,
        reduction_factor=2,
        last_trial_id=39,
        distributions=study_first_layer.best_trial.distributions,
    )
    all_trials = successive_halving.successive_halving(
        normalized_class_weights=normalized_class_weights,
        seed=seed,
        train_df=train_df,
        val_df=val_df,
        optuna_study_path=optuna_study_path,
        num_classes=num_classes,
        optuna_study_name=optuna_study_name,
      
    )
    final_plot(all_trials, plots_dir)
    for trial in all_trials.values():
        print(f"Trial ID {trial['trial_id']} accuracy : {1 - trial['best_val_err']}, stopped at epoch: {trial['stopped']}")

    # get the best config from all trials
    best_trial = successive_halving.get_best_trial()
    print(f"Best Trial: {best_trial['trial_id']} with Value = {best_trial['best_val_err']}, Params = {best_trial['params']}")

    # close wandb run
    wandb_run.finish()
    

    # we want to return the best n trials
    sorted_trials = sorted(all_trials.values(), key=lambda x: x["best_val_err"])
    best_n_trials = sorted_trials[:get_best_n_trials]

    return best_n_trials

def final_plot(all_trials, plots_dir):

    import plotly.graph_objs as go

    fig = go.Figure()

    
    # Define color map by trial ID range
    def get_color(trial_id):
        if 0 <= trial_id <= 31:
            return 'red'
        elif 32 <= trial_id <= 39:
            return 'blue'
        elif 40 <= trial_id <= 43:
            return 'green'
        elif 44 <= trial_id <= 45:
            return 'yellow'
        else:
            return 'gray'  # fallback for unexpected IDs

    for trial in all_trials.values():
        trial_id = trial["trial_id"]
        results = trial["val_accuracies"]
        last_epoch = 16  # Default last epoch if not stopped
        if trial["stopped"] is False:
            # If the trial is not stopped, we assume it ran for 16 epochs
            last_epoch = 16

        values_list = []
        previous_val = results[0][1]
        first_epoch = results[0][0]

        for epoch in range(first_epoch, last_epoch):
            result = next((t[1] for t in results if t[0] == epoch), None)
            if result is None:
                result = previous_val
            values_list.append((epoch, result))
            previous_val = result

        epochs, values = zip(*values_list)
        color = get_color(trial_id)

        fig.add_trace(go.Scatter(
            x=epochs,
            y=values,
            mode='lines+markers',
            line=dict(color=color),
            marker=dict(color=color),
            name=f"Trial {trial_id}",  # This will show in the legend
            showlegend=False,
            hovertemplate=f'Trial {trial_id}<br>Epoch: %{{x}}<br>Value: %{{y}}'
        ))

    fig.update_layout(
        title='Validation Accuracy per Epoch',
        xaxis_title='Epoch',
        yaxis_title='Validation Accuracy',
        hovermode='x unified'
    )

    fig.write_html(plots_dir / "val_accuracy_plot.html")

def plots(study, plots_dir):
    print("Saving optimization plots...")
    try:
        fig = vis.plot_param_importances(study)
        fig.write_html(plots_dir / "param_importances.html")
        print("Saved parameter importances plot")
    except Exception as e:
        print(f"Error saving parameter importances plot: {e}")

    try:
        fig = vis.plot_timeline(study)
        fig.write_html(plots_dir / "timeline.html")
        print("Saved timeline plot")
    except Exception as e:
        print(f"Error saving timeline plot: {e}")

    try:
        fig = vis.plot_optimization_history(study)
        fig.write_html(plots_dir / "optimization_history.html")
        print("Saved optimization history plot")
    except Exception as e:
        print(f"Error saving optimization history: {e}")

    try:
        fig = vis.plot_param_importances(
            study, target=lambda t: t.duration.total_seconds(), target_name="duration"
        )
        fig.write_html(plots_dir / "param_importances_duration.html")
        print("Saved parameter importances (duration) plot")
    except Exception as e:
        print(f"Error saving parameter importances duration plot: {e}")
    try:
        fig = vis.plot_contour(study)
        fig.write_html(plots_dir / "contour.html")
        print("Saved contour plot")
    except Exception as e:
        print(f"Error saving contour plot: {e}")

    try:
        fig = vis.plot_edf(study)
        fig.write_html(plots_dir / "edf.html")
        print("Saved EDF plot")
    except Exception as e:
        print(f"Error saving EDF plot: {e}")

    try:
        fig = vis.plot_intermediate_values(study)
        fig.write_html(plots_dir / "intermediate_values.html")
        print("Saved intermediate values plot")
    except Exception as e:
        print(f"Error saving intermediate values plot: {e}")

    try:
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(plots_dir / "parallel_coordinate.html")
        print("Saved parallel coordinate plot")
    except Exception as e:
        print(f"Error saving parallel coordinate plot: {e}")

    try:
        fig = vis.plot_param_importances(study)
        fig.write_html(plots_dir / "param_importances.html")
        print("Saved parameter importances plot")
    except Exception as e:
        print(f"Error saving parameter importances plot: {e}")

    try:
        fig = vis.plot_rank(study)
        fig.write_html(plots_dir / "rank.html")
        print("Saved rank plot")
    except Exception as e:
        print(f"Error saving rank plot: {e}")

    try:
        fig = vis.plot_slice(study)
        fig.write_html(plots_dir / "slice.html")
        print("Saved slice plot")
    except Exception as e:
        print(f"Error saving slice plot: {e}")




