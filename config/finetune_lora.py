"""
Config for LoRA fine-tuning: TinyStories → Scientific domain

Demonstrates parameter-efficient fine-tuning using PEFT LoRA.
Load a pre-trained TinyStories model and adapt it to generate
scientific paper abstracts using low-rank adaptation.

Base model: models/gpt-38M-tinystories-pretrain/ (38M params)
Target domain: Scientific papers (arXiv)
Method: LoRA (Low-Rank Adaptation via PEFT)

Training budget: ~4,000 iterations (sufficient for domain adaptation)
Trainable params: ~2% of total (LoRA rank=16)
Expected time: 1-3 hours on CPU
"""

config = {
    # Model output directory (standardized naming)
    "out_dir": "models/gpt-38M-tinystories-to-scientific-lora",

    # Initialization: load pre-trained TinyStories model
    "init_from": "resume",
    "resume_dir": "models/gpt-38M-tinystories-pretrain",  # Base model location

    # LoRA fine-tuning mode
    "finetune_lora": True,  # Enable PEFT LoRA (only train adapter params)

    # LoRA Hyperparameters (PEFT)
    "lora_rank": 16,        # Rank (r): Higher rank for domain shift
    "lora_alpha": 32.0,     # Scaling factor (typically 2 * rank)
    "lora_dropout": 0.05,   # Dropout on LoRA path (light regularization)
    "lora_targets": ['c_attn', 'c_proj'],  # Target attention layers

    # Evaluation & Logging
    "eval_interval": 200,   # Evaluate every 200 steps
    "eval_iters": 50,       # 50 batches for validation
    "log_interval": 10,     # Log every 10 steps for detailed progress

    # Dataset: scientific papers (arXiv abstracts)
    "dataset": "scientific_papers",

    # Batch settings (same as pre-training for consistency)
    "gradient_accumulation_steps": 4,
    "batch_size": 4,  # Effective batch = 4 * 4 = 16
    "block_size": 256,  # Same as pre-training

    # Optimizer: AdamW (only LoRA parameters!)
    "learning_rate": 3e-4,  # Higher LR for LoRA domain adaptation
    "max_iters": 4000,  # More iterations for domain adaptation
    "weight_decay": 1e-2,  # Light weight decay for LoRA
    "beta1": 0.9,
    "beta2": 0.95,

    # Learning rate schedule
    "decay_lr": True,
    "warmup_iters": 100,  # Short warmup
    "lr_decay_iters": 2000,
    "min_lr": 1e-5,  # Minimum learning rate

    # System settings
    "device": "cpu",
    "dtype": "float32",
    "compile": False,
    "seed": 42,  # Different seed from pre-training

    # Model architecture (inherited from checkpoint, but specified for clarity)
    # These will be overridden by the loaded checkpoint
    "n_layer": 5,
    "n_head": 5,
    "n_embd": 320,
    "dropout": 0.0,  # No dropout during fine-tuning (LoRA has its own dropout)
}

# Notes on hyperparameters:
#
# LoRA Rank (r):
#   - Smaller (4-8): Fewer params, faster training, may underfit
#   - Medium (8-16): Good balance for most tasks
#   - Larger (16-32): More capacity, slower, may overfit on small datasets
#
# LoRA Alpha (α):
#   - Rule of thumb: α = 2 * r
#   - Controls magnitude of LoRA updates
#   - Higher α = stronger adaptation (but risk overfitting)
#
# Learning Rate:
#   - Fine-tuning typically uses 1/2 to 1/10 of pre-training LR
#   - Start conservative: 1e-4 to 2e-4
#   - Can increase if loss plateaus
#
# Iterations:
#   - LoRA fine-tunes much faster than full training
#   - 1000-3000 iters often sufficient for domain adaptation
#   - Monitor validation loss - stop when it plateaus
#
# Expected behavior:
#   - Loss should drop quickly in first 500 iters
#   - Validation loss should track training loss (no overfitting with LoRA)
#   - Generated text should shift from children's stories → scientific language
