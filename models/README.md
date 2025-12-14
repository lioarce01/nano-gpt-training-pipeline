# Model Directory

Standardized location for all trained models and checkpoints.

## Naming Convention

Format: `{architecture}-{size}-{dataset}-{training_type}-{version}`

### Components:

- **architecture**: Model type (e.g., `gpt`, `gpt2`)
- **size**: Parameter count (e.g., `1.8M`, `124M`) or size descriptor (e.g., `tiny`, `small`, `base`)
- **dataset**: Training dataset (e.g., `tinystories`, `scientific`, `arxiv`)
- **training_type**: Training method (e.g., `pretrain`, `lora`, `ft`)
- **version**: Optional version/date (e.g., `v1`, `20231213`)

### Examples:

```
models/
├── gpt-38M-tinystories-pretrain/          # Pretrained on TinyStories (38M params)
├── gpt-38M-scientific-pretrain/           # Pretrained on scientific papers (38M params)
├── gpt-38M-tinystories-to-scientific-lora/  # LoRA fine-tuned
└── gpt-38M-scientific-merged-ft/          # Merged LoRA weights
```

## Directory Structure

Each model directory contains:

```
model-name/
├── ckpt.pt                  # PyTorch checkpoint (always)
├── model.safetensors        # SafeTensors format (optional)
├── model_args.json          # Model architecture config
├── config.json              # Training config
├── export_meta.json         # Export metadata
└── adapters/                # LoRA adapters (if applicable)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Standard Model Configurations

### Small (38M params) - CPU Training
```python
n_layer=5, n_head=5, n_embd=320, vocab_size=50257
# Actual params: ~38M
# Use for: CPU training, quick experiments
```

### Medium (124M params) - GPU Training
```python
n_layer=12, n_head=12, n_embd=768, vocab_size=50257
# Actual params: ~124M (GPT-2 Small size)
# Use for: GPU training, better quality
```

### Large (350M+ params) - Multi-GPU
```python
n_layer=24, n_head=16, n_embd=1024, vocab_size=50257
# Actual params: ~350M (GPT-2 Medium size)
# Use for: Production models, large datasets
```

## Usage

### Training
```bash
# Set output directory in config
"out_dir": "models/gpt-38M-scientific-pretrain"

# Run training
python train.py config/train_scientific.py
```

### Loading
```python
from checkpoint import CheckpointManager

# Load for inference
model = CheckpointManager.load_model_for_inference(
    "models/gpt-38M-scientific-pretrain/ckpt.pt"
)

# Load for fine-tuning
model, metadata = CheckpointManager.load_model(
    "models/gpt-38M-scientific-pretrain/ckpt.pt",
    apply_lora=True,
    lora_config={'rank': 8, 'alpha': 16}
)
```

## Model Lifecycle

1. **Pretrain** → `models/gpt-{size}-{dataset}-pretrain/`
2. **LoRA Fine-tune** → `models/gpt-{size}-{source}-to-{target}-lora/`
3. **Merge** → `models/gpt-{size}-{source}-to-{target}-merged/`
4. **Deploy** → Copy best checkpoint to production

## Best Practices

- ✅ Use descriptive names that indicate dataset and training method
- ✅ Keep config.json with each model for reproducibility
- ✅ Version models (v1, v2) when iterating
- ✅ Document training results in a separate training_log.txt
- ✅ Clean up old/failed runs to save space
- ❌ Don't mix different model sizes in same directory
- ❌ Don't use generic names like "model1", "test"
