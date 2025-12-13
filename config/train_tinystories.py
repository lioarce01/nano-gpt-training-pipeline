"""
Config for training a small GPT on TinyStories using GPT-2 BPE.
Defaults lean toward GPU; adjust device/batch if running on CPU.
"""

config = {
    "out_dir": "out-tinystories",
    "eval_interval": 800,
    "eval_iters": 100,
    "log_interval": 20,
    # data
    "dataset": "tinystories",
    "gradient_accumulation_steps": 4,
    "batch_size": 4,  # CPU-friendly; eff batch = 16 con grad accum
    "block_size": 256,
    # model (CPU-sized)
    "n_layer": 5,
    "n_head": 5,
    "n_embd": 320,
    "dropout": 0.2,
    # adamw optimizer
    "learning_rate": 3e-4,
    "max_iters": 6000,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    # lr schedule (suave)
    "decay_lr": True,
    "warmup_iters": 400,
    "lr_decay_iters": 6000,
    "min_lr": 1.5e-4,
    # system
    "device": "cpu",
    "dtype": "float32",
    "compile": False,
    "seed": 1337,
}

