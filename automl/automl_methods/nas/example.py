# optuna
def output_head_layerwise_search_space(trial):
    n_layers = trial.suggest_int("n_layers", 1, 5)

    hidden_sizes = []
    activations = []
    use_norms = []
    
    for i in range(n_layers):
        hidden_size = trial.suggest_categorical(f"hidden_size_{i}", [64, 128, 256, 512, 1024])
        activation = trial.suggest_categorical(f"activation_{i}", ["ReLU", "GELU", "Tanh", "LeakyReLU"])
        use_norm = trial.suggest_categorical(f"use_norm_{i}", [True, False])

        hidden_sizes.append(hidden_size)
        activations.append(activation)
        use_norms.append(use_norm)
    
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    return {
        "n_layers": n_layers,
        "hidden_sizes": hidden_sizes,
        "activations": activations,
        "use_norms": use_norms,
        "dropout_rate": dropout_rate
    }