import yaml
from pathlib import Path


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
        print(f"Processing trial {trial_id} with best validation error {trial['best_val_err']}")
        results = trial["val_accuracies"]
        last_epoch = trial["stopped"]  # Default last epoch if not stopped
        if trial["stopped"] == -1 or trial["stopped"] == False:
            # If the trial is not stopped, we assume it ran for 16 epochs
            last_epoch = 16

        values_list = []
        previous_val = results[0][1]
        first_epoch = results[0][0]
        print(f"First epoch: {first_epoch}, Last epoch: {last_epoch}")

        for epoch in range(first_epoch, last_epoch):
            result = next((t[1] for t in results if t[0] == epoch), None)
            if result is None:
                result = previous_val
            values_list.append((epoch, result))
            previous_val = result
        print(values_list)

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

    fig.write_html(plots_dir / "dummy.html")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def final_plot_seaborn(all_trials, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)

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
            return 'gray'

    records = []

    for trial in all_trials.values():
        trial_id = trial["trial_id"]
        results = trial["val_accuracies"]
        last_epoch = trial["stopped"]
        if last_epoch == -1 or last_epoch is False:
            last_epoch = 16

        previous_val = results[0][1]
        first_epoch = results[0][0]

        records.append({
                "Epoch": 0,
                "Validation Accuracy": 0,
                "Trial ID": trial_id,
                "Color": get_color(trial_id)
            })

        for epoch in range(first_epoch, last_epoch):
            result = next((t[1] for t in results if t[0] == epoch), None)
            if result is None:
                result = previous_val
            previous_val = result

            records.append({
                "Epoch": epoch + 1,
                "Validation Accuracy": result,
                "Trial ID": trial_id,
                "Color": get_color(trial_id)
            })

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Set seaborn style
    sns.set(style="whitegrid", palette="deep")

    plt.figure(figsize=(12, 7))

    # Plot manually grouped by color, because seaborn doesn't support individual colors per line well
    for color in df["Color"].unique():
        subset = df[df["Color"] == color]
        sns.lineplot(
            data=subset,
            x="Epoch",
            y="Validation Accuracy",
            hue="Trial ID",
            legend=False,
            linewidth=3,
            alpha=0.9,
            palette=[color] * subset["Trial ID"].nunique()
        )
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red',   label='Sampled at Layer 1'),
        Patch(facecolor='blue',  label='Sampled at Layer 2'),
        Patch(facecolor='green', label='Sampled at Layer 3'),
        Patch(facecolor='yellow', label='Sampled at Layer 4')
    ]
    plt.legend(handles=legend_elements, title="Sampling Layer", loc="best")

    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.tight_layout()

    save_path = plots_dir / "final_plot_seaborn.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")


# Load the YAML file
with open("./results/dataset=yelp/seed=42/optuna_2/all_trials.yaml", "r") as file:
    data = yaml.safe_load(file)

# Now `data` is a dictionary
print(list(data.values())[0])
final_plot_seaborn(data, Path("./results"))

