"""
Config for finetuning a small GPT on Tiny Shakespeare using GPT-2 BPE.

These values intentionally mirror nanoGPT defaults but scaled down to run
quickly on a single GPU/CPU.
"""

config = {
    "out_dir": "out-shakespeare",
    "eval_interval": 200,
    "eval_iters": 80,
    "log_interval": 10,
    # data
    "dataset": "shakespeare",
    "gradient_accumulation_steps": 4,
    "batch_size": 8,  # keep modest for CPU
    "block_size": 256,
    # model (CPU-friendly, smaller but still non-toy)
    "n_layer": 5,
    "n_head": 6,
    "n_embd": 288,
    "dropout": 0.25,
    # adamw optimizer
    "learning_rate": 3e-4,
    "max_iters": 3200,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    # lr schedule
    "decay_lr": False,
    "warmup_iters": 200,
    "lr_decay_iters": 8000,
    "min_lr": 1e-4,
    # system
    "device": "cpu",  # fuerza CPU en el 7800X3D
    "dtype": "float32",  # fp32 es lo más estable en CPU
    "compile": False,  # torch.compile no aporta mucho en CPU aquí  
    "seed": 1337,
}

