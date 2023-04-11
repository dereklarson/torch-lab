# EmbedMLP Models

mod_add_113 = {
    # Data
    "data_seed": 1,
    "operation": "add",
    "value_count": 2,
    "value_range": 113,
    "result_mod": 113,
    "training_fraction": 0.4,
    "use_operators": False,
    # Model
    "torch_seed": 5,
    "d_embed": 24,
    "mlp_layers": [24],
    "n_ctx": 2,
    "n_outputs": 113,
    "n_vocab": 113,
    "use_bias": False,
    "layer_type": "Linear",
    # Optimization
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "adam_betas": (0.9, 0.98),
    "n_epochs": 5000,
}

parity_tiny = {
    # Data
    "data_seed": 1,
    "operation": "add",
    "value_count": 2,
    "value_range": 8,
    "result_mod": 2,
    "training_fraction": 0.6,
    "use_operators": False,
    # Model
    "torch_seed": 5,
    "d_embed": 1,
    "mlp_layers": [4],
    "n_ctx": 2,
    "n_outputs": 2,
    "n_vocab": 8,
    "use_bias": False,
    "layer_type": "Linear",
    # Optimization
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "adam_betas": (0.9, 0.98),
    "n_epochs": 5000,
}
