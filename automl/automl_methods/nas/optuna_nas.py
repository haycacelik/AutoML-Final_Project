
from automl.core import TextAutoML
import wandb
# import seaborn as sns
# import matplotlib.pyplot as plt
# from optuna.samplers import TPESampler
# import optuna
# import numpy as np
# from pathlib import Path
# import os
# from optuna.trial import TrialState
# from scipy import stats


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
    
    return val_err


# def visualize_kernels(study, trial):
#     """Visualize TPE Parzen estimators (kernels) for good and bad trials."""
    
#     sampler = study.sampler
#     if not isinstance(sampler, TPESampler):
#         print("Sampler is not TPE. Cannot visualize.")
#         return

#     # Get completed trials
#     trials = study.get_trials(states=[TrialState.COMPLETE])
#     if len(trials) < sampler._n_startup_trials:
#         print(f"Not enough trials ({len(trials)}) to start TPE modeling. Need {sampler._n_startup_trials}.")
#         return

#     trial_id = trial.number
    
#     # Create plots directory
#     plots_dir = Path("./plots")
#     plots_dir.mkdir(exist_ok=True)
    
#     # Split trials into good (below) and bad (above) based on gamma
#     n_completed = len(trials)
#     gamma_value = sampler._gamma(n_completed)
    
#     # Sort trials by objective value
#     if study.direction.name == 'MINIMIZE':
#         sorted_trials = sorted(trials, key=lambda t: t.value)
#     else:
#         sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)
    
#     good_trials = sorted_trials[:gamma_value]
#     bad_trials = sorted_trials[gamma_value:]
    
#     print(f"Trial {trial_id}: {len(good_trials)} good trials, {len(bad_trials)} bad trials")
    
#     # Get parameter names from the search space
#     param_names = list(study.best_params.keys()) if study.best_params else []
#     if not param_names and trials:
#         param_names = list(trials[0].params.keys())
    
#     # Create subplot grid
#     n_params = len(param_names)
#     if n_params == 0:
#         print("No parameters to visualize")
#         return
        
#     fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 8))
#     if n_params == 1:
#         axes = [axes]
#     elif n_params <= 2:
#         axes = axes.flatten()
#     else:
#         axes = axes.flatten()
    
#     for i, param_name in enumerate(param_names):
#         ax = axes[i] if i < len(axes) else plt.subplot(2, (n_params + 1) // 2, i + 1)
        
#         # Extract parameter values
#         good_values = [t.params[param_name] for t in good_trials if param_name in t.params]
#         bad_values = [t.params[param_name] for t in bad_trials if param_name in t.params]
        
#         if not good_values:
#             ax.text(0.5, 0.5, f'No data for {param_name}', ha='center', va='center')
#             ax.set_title(param_name)
#             continue
            
#         # Check if parameter is categorical or numerical
#         if isinstance(good_values[0], (str, bool)):
#             # Categorical parameter - use bar plots
#             all_categories = list(set(good_values + bad_values))
            
#             good_counts = {cat: good_values.count(cat) for cat in all_categories}
#             bad_counts = {cat: bad_values.count(cat) for cat in all_categories}
            
#             x = np.arange(len(all_categories))
#             width = 0.35
            
#             ax.bar(x - width/2, [good_counts[cat] for cat in all_categories], 
#                   width, label='Good (l(x))', alpha=0.7, color='green')
#             ax.bar(x + width/2, [bad_counts[cat] for cat in all_categories], 
#                   width, label='Bad (g(x))', alpha=0.7, color='red')
            
#             ax.set_xlabel(param_name)
#             ax.set_ylabel('Count')
#             ax.set_xticks(x)
#             ax.set_xticklabels(all_categories, rotation=45)
#             ax.legend()
            
#         else:
#             # Numerical parameter - use KDE plots
#             if len(good_values) > 1:
#                 # Create KDE for good values
#                 good_kde = stats.gaussian_kde(good_values)
#                 x_range = np.linspace(min(good_values + bad_values), 
#                                     max(good_values + bad_values), 100)
#                 good_density = good_kde(x_range)
#                 ax.plot(x_range, good_density, 'g-', label='Good l(x)', linewidth=2)
#                 ax.fill_between(x_range, good_density, alpha=0.3, color='green')
            
#             if len(bad_values) > 1:
#                 # Create KDE for bad values
#                 bad_kde = stats.gaussian_kde(bad_values)
#                 x_range = np.linspace(min(good_values + bad_values), 
#                                     max(good_values + bad_values), 100)
#                 bad_density = bad_kde(x_range)
#                 ax.plot(x_range, bad_density, 'r-', label='Bad g(x)', linewidth=2)
#                 ax.fill_between(x_range, bad_density, alpha=0.3, color='red')
            
#             # Plot actual trial points
#             ax.scatter(good_values, np.zeros(len(good_values)), 
#                       color='green', alpha=0.6, s=20, label='Good trials')
#             ax.scatter(bad_values, np.zeros(len(bad_values)), 
#                       color='red', alpha=0.6, s=20, label='Bad trials')
            
#             # Highlight current trial's parameter value if available
#             if param_name in trial.params:
#                 current_value = trial.params[param_name]
#                 ax.axvline(current_value, color='blue', linestyle='--', 
#                           label=f'Current trial', linewidth=2)
            
#             ax.set_xlabel(param_name)
#             ax.set_ylabel('Density')
#             ax.legend()
        
#         ax.set_title(f'{param_name} - Trial {trial_id}')
#         ax.grid(True, alpha=0.3)
    
#     # Hide empty subplots
#     for j in range(i + 1, len(axes)):
#         axes[j].set_visible(False)
    
#     plt.suptitle(f'TPE Parzen Estimators - Trial {trial_id}\n'
#                 f'Objective: {trial.value:.4f} ({"Good" if trial in good_trials else "Bad"})', 
#                 fontsize=14)
#     plt.tight_layout()
    
#     # Save the plot
#     save_path = plots_dir / f"tpe_kernels_trial_{trial_id:03d}.png"
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     print(f"Saved TPE visualization to {save_path}")
    
#     # Also log to wandb if available
#     try:
#         wandb.log({f"tpe_kernels_trial_{trial_id}": wandb.Image(str(save_path))})
#     except:
#         pass  # wandb might not be initialized in some contexts
    
#     plt.close()


# def create_tpe_animation(study, output_path="./plots/tpe_evolution.gif"):
#     """Create an animated GIF showing TPE kernel evolution over trials."""
#     try:
#         from PIL import Image
#         import glob
        
#         # Get all kernel plot files
#         kernel_files = sorted(glob.glob("./plots/tpe_kernels_trial_*.png"))
        
#         if len(kernel_files) < 2:
#             print("Need at least 2 trial visualizations to create animation")
#             return
            
#         # Create animation
#         images = []
#         for filename in kernel_files:
#             images.append(Image.open(filename))
        
#         # Save as GIF
#         images[0].save(
#             output_path,
#             save_all=True,
#             append_images=images[1:],
#             duration=1000,  # 1 second per frame
#             loop=0
#         )
#         print(f"Created TPE evolution animation: {output_path}")
        
#     except ImportError:
#         print("PIL not available. Install with: pip install Pillow")
#     except Exception as e:
#         print(f"Error creating animation: {e}")


# def visualization_callback(study, trial):
#     """Callback function for optuna study to visualize TPE kernels after each trial."""
#     try:
#         visualize_kernels(study, trial)
#     except Exception as e:
#         print(f"Error in visualization callback for trial {trial.number}: {e}")


# def run_nas_with_visualization(dataset, epochs, lr, batch_size, seed, val_percentage, 
#                              token_length, weight_decay, fraction_layers_to_finetune, 
#                              data_fraction, train_df, val_df, test_df, num_classes, 
#                              load_path, output_path, n_trials=20):
#     """Run NAS with TPE kernel visualization."""
    
#     # Create TPE sampler with multivariate enabled for better performance
#     sampler = TPESampler(
#         n_startup_trials=5,  # Reduced for faster visualization
#         n_ei_candidates=24,
#         multivariate=True,
#         seed=seed
#     )
    
#     # Create study
#     study = optuna.create_study(
#         direction="minimize", 
#         sampler=sampler,
#         study_name=f"nas_{dataset}_{seed}"
#     )
    
#     # Define objective with fixed parameters
#     def objective_with_fixed_params(trial):
#         return objective(
#             trial=trial,
#             dataset=dataset,
#             epochs=epochs,
#             lr=lr,
#             batch_size=batch_size,
#             seed=seed,
#             val_percentage=val_percentage,
#             token_length=token_length,
#             weight_decay=weight_decay,
#             fraction_layers_to_finetune=fraction_layers_to_finetune,
#             data_fraction=data_fraction,
#             train_df=train_df,
#             val_df=val_df,
#             test_df=test_df,
#             num_classes=num_classes,
#             load_path=load_path,
#             output_path=output_path
#         )
    
#     # Run optimization with visualization callback
#     print(f"Starting NAS with TPE visualization for {n_trials} trials...")
#     study.optimize(objective_with_fixed_params, n_trials=n_trials, callbacks=[visualization_callback])
    
#     # Create animation of TPE evolution
#     create_tpe_animation(study, f"./plots/tpe_evolution_{dataset}.gif")
    
#     print("NAS completed!")
#     print(f"Best score: {1 - study.best_value:.4f}")
#     print(f"Best params: {study.best_params}")
    
#     return study

