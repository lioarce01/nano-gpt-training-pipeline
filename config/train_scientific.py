"""
Config for pretraining GPT-38M on Scientific Papers (500k arXiv papers) from scratch.

Dataset: 500k arXiv papers (~310 MB raw, ~139 MB tokenized, 65.5M train tokens)
Model: 38M parameters (5 layers, 5 heads, 320 embedding dim)
Training: 10 epochs = 159,891 iterations (~655M tokens processed)
Hardware: Optimized for high-end CPU (Ryzen 7 7800X3D or similar)

Expected training time: ~36 hours (1.5 days) on Ryzen 7 7800X3D
Expected final loss: ~1.8-2.2 (validation)
Token/Param ratio: 17.2x (Good, close to Chinchilla optimal ~20x)

Note: More epochs needed because arXiv abstracts are shorter than expected.
"""

config = {
    # Model output directory (standardized naming)
    "out_dir": "models/gpt-38M-scientific-pretrain",

    # Evaluation & Logging
    "eval_interval": 1600,  # Evaluate every 1600 steps (~1% of training)
    "eval_iters": 100,      # 100 batches for validation
    "log_interval": 50,     # Log every 50 steps for progress tracking

    # Initialization
    "init_from": "scratch",  # Pretraining from scratch

    # Dataset
    "dataset": "scientific_papers",  # Uses data/scientific_papers/{train,val}.bin
    "gradient_accumulation_steps": 4,  # Effective batch = 16
    "batch_size": 4,        # Per-step batch (CPU-friendly)
    "block_size": 256,      # Context length (tokens)

    # Model Architecture (38M params)
    "n_layer": 5,           # 5 transformer blocks
    "n_head": 5,            # 5 attention heads
    "n_embd": 320,          # 320 embedding dimension
    "dropout": 0.2,         # Dropout for regularization

    # Optimizer (AdamW)
    "learning_rate": 3e-4,  # Peak learning rate
    "max_iters": 159891,    # 10 epochs over 500k dataset (655M tokens)
    "weight_decay": 1e-1,   # L2 regularization
    "beta1": 0.9,
    "beta2": 0.95,

    # Learning Rate Schedule (Cosine with Warmup)
    "decay_lr": True,
    "warmup_iters": 1600,   # ~1% of training for warmup
    "lr_decay_iters": 159891,  # Decay over full training
    "min_lr": 3e-5,         # 10% of peak LR

    # System
    "device": "cpu",        # CPU training (change to "cuda" for GPU)
    "dtype": "float32",     # CPU requires float32
    "compile": False,       # Disable torch.compile on CPU
    "seed": 1337,
}

# Training notes:
# - 500k samples = 65.5M train tokens (actual from prepare_hf.py)
# - Effective batch = 16 samples = 4,096 tokens per step
# - 10 epochs = 159,891 iterations (more epochs due to shorter abstracts)
# - Total tokens processed: ~655M
# - Expected checkpoints: 100 evaluations (every 1.6k steps)
# - Estimated time: 36 hours on Ryzen 7 7800X3D (~0.8 sec/step)
# - Token/Param ratio: 17.2x (Good, close to Chinchilla optimal)
#
# Why 10 epochs? Dataset has fewer tokens than expected (short abstracts),
# so we compensate with more passes over the data to reach good tok/param ratio.

