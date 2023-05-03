from tlab.datasets import Algorithmic, Shakespeare
from tlab.models import EmbedMLP, Transformer

# Karpathy's NanoGPT model, see:
# https://github.com/karpathy/nanoGPT

nano_gpt = {
    "dataset_class": Shakespeare,
    "data_seed": 1,
    # Model
    "model_class": Transformer,
    "torch_seed": 1,
    "activation_type": "GeLU",
    "d_embed": 384,
    "d_head": 1,  # dependent (64)
    "d_mlp": 1,  # dependent (1536)
    "n_ctx": 256,
    "n_blocks": 6,
    "n_heads": 6,
    "n_vocab": 65,
    "unembed_type": "tied",
    "use_bias": False,
    # Optimization
    "n_epochs": 15,
    "learning_rate": 0.001,
    "warmup_iters": 100,
    "weight_decay": 0.2,
    "grad_clip": 1.0,
    "adam_betas": (0.9, 0.99),
}


# EmbedMLP Models

mod_add_113 = {
    # Data
    "dataset_class": Algorithmic,
    "data_seed": 1,
    "operation": "add",
    "value_count": 2,
    "value_range": 113,
    "result_mod": 113,
    "training_fraction": 0.4,
    "use_operators": False,
    # Model
    "model_class": EmbedMLP,
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
    "dataset_class": Algorithmic,
    "data_seed": 1,
    "operation": "add",
    "value_count": 2,
    "value_range": 8,
    "result_mod": 2,
    "training_fraction": 0.6,
    "use_operators": False,
    # Model
    "model_class": EmbedMLP,
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
