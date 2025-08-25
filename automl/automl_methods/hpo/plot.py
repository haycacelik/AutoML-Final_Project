import optuna
import matplotlib.pyplot as plt
from optuna.importance import get_param_importances, FanovaImportanceEvaluator
import matplotlib as mpl
import matplotlib.colors as mcolors

study_path = "study/study_layer_1.db"
study_name = "optuna_3"

study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_path}")

def plot_optimization_history(study):
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    trial_numbers = [t.number for t in completed_trials]
    objective_values = [t.value for t in completed_trials]

    best_so_far = []
    current_best = float("inf")
    for val in objective_values:
        current_best = min(current_best, val)
        best_so_far.append(current_best)

    plt.figure(figsize=(10, 6))
    # Dots for actual objective values
    plt.scatter(trial_numbers, objective_values, color='tab:blue', label="Objective Value", zorder=3)
    plt.plot(trial_numbers, best_so_far, color='tab:orange', label="Best Value So Far", linewidth=2, zorder=2)

    new_bests_x = [trial_numbers[i] for i in range(len(objective_values)) if objective_values[i] == best_so_far[i]]
    new_bests_y = [objective_values[i] for i in range(len(objective_values)) if objective_values[i] == best_so_far[i]]
    plt.scatter(new_bests_x, new_bests_y, color='orange', edgecolor='black', label="New Best", zorder=4, s=50)

    plt.title("Optimization History")
    plt.xlabel("Trial Number")
    plt.ylabel("Objective Value")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_param_importance(study):
    evaluator = FanovaImportanceEvaluator(seed=42)
    importances = get_param_importances(study, evaluator=evaluator)

    params = list(importances.keys())
    scores = list(importances.values())

    def clean_param_name(param):
        if param == "lr":
            return "Learning rate"
        else:
            parts = param.split("_")
            return parts[0].capitalize() + " " + " ".join(parts[1:]) if len(parts) > 1 else parts[0].capitalize()

    params = [clean_param_name(param) for param in params]
    plt.figure(figsize=(10, len(params) * 0.5))  # dynamic height based on number of bars

    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
    colors = mpl.colormaps['inferno_r'](norm(scores))  # or try 'viridis', 'magma', 'plasma'
    bars = plt.barh(params, scores, color=colors, height=0.6, edgecolor='gray')

    plt.xlabel("Importance")
    plt.title("Parameter Importances for Objective Value")
    plt.gca().invert_yaxis()  # Most important on top

    # Add value labels
    for bar, score in zip(bars, scores):
        if score > 0.3:
            text_color = 'white'  # Or maybe a soft off-white like '#f0f0f0'
        else:
            text_color = '#121212'  # dark gray instead of pure black
        plt.text(score - 0.003 if score > 0.05 else score + 0.01,
                 bar.get_y() + bar.get_height() / 2 + 0.045,
                 f'{score:.3f}', va='center', ha='right' if score > 0.05 else 'left',
                 color=text_color, fontsize=10, fontweight=500)


    plt.grid(axis='x', linestyle='--', linewidth=0.8, color='#222222', alpha=0.3)
    plt.gca().set_axisbelow(True)  # Make sure grid is behind bars

    plt.xlim(0, max(scores) * 1.1)  # Add 10% headroom

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 11
    })

    plt.tight_layout()
    plt.show()


plot_optimization_history(study)
plot_param_importance(study)