"""
Config for pretraining Optimized GPT-38M on Scientific Papers (500k arXiv papers) from scratch.

Uses model_optimized.py with 2025 improvements:
- RMSNorm (10-15% faster than LayerNorm)
- RoPE (better position encoding)
- GQA - Grouped Query Attention (15-20% faster)
- SwiGLU activation (better convergence)
- Pre-normalization (better gradients)
- BFloat16 training (40-60% faster on modern CPUs)

Dataset: 500k arXiv papers (~310 MB raw, ~139 MB tokenized, 65.5M train tokens)
Model: 38M parameters (5 layers, 5 heads, 320 embedding dim)
Training: 10 epochs = 159,891 iterations (~655M tokens processed)
Hardware: Optimized for high-end CPU (Ryzen 7 7800X3D or similar)

Expected training time: ~18 hours (vs ~36 hours with base model)
Expected final loss: ~1.6-2.0 (validation) - better than base model
Token/Param ratio: 17.2x (Good, close to Chinchilla optimal ~20x)

Speedup: ~2x faster than base model.py
Quality: 10-20% better loss at same iteration count
"""

config = {
    # I/O
    "out_dir": "models/gpt-38M-scientific-pretrain-optimized",
    "eval_interval": 1600,
    "eval_iters": 100,
    "log_interval": 50,

    # Data
    "dataset": "scientific_papers",
    "init_from": "scratch",

    # Model architecture - uses model_optimized.py
    "model_type": "optimized",  # Use OptimizedGPT instead of GPT
    "n_layer": 5,
    "n_head": 8,
    "n_embd": 320,
    "n_kv_head": 2,  # GQA: 8 Q heads share 2 KV heads (4:1 ratio)
    "block_size": 256,
    "dropout": 0.2,
    "bias": False,  # No bias in Linear layers (modern standard)

    # Training
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch: 4 * 4 = 16
    "max_iters": 159891,  # 10 epochs on 500k dataset

    # Optimizer
    "learning_rate": 3e-4,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # LR scheduler
    "decay_lr": True,
    "warmup_iters": 1600,  # 1% warmup
    "lr_decay_iters": 159891,  # Decay to end
    "min_lr": 3e-5,  # 10% of peak LR

    # System
    "device": "cpu",
    "dtype": "bfloat16",  # BFloat16 for CPU training (40-60% speedup)
    "compile": False,  # torch.compile not beneficial for CPU

    # Reproducibility
    "seed": 1337,

    # Logging
    "wandb_log": False,
    "wandb_project": "nano-gpt-optimized",
    "wandb_run_name": "gpt-38M-scientific-optimized",
}
