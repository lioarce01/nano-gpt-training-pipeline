# Scripts / Utility Tools

Auxiliary tools for working with trained GPT models.

## Overview

This directory contains scripts for:
- **Generating text** from trained models (base or LoRA fine-tuned)
- **Merging LoRA adapters** into base models for deployment
- **Comparing models** to validate fine-tuning results
- **Benchmarking performance** between base and optimized architectures

## Scripts

### `sample.py` - Text Generation

Generate text from any trained GPT model checkpoint.

**Usage:**
```bash
# Basic generation
python scripts/sample.py --checkpoint models/gpt-38M-scientific-pretrain/ckpt.pt

# Custom prompt
python scripts/sample.py \
    --checkpoint models/gpt-38M-scientific-pretrain/ckpt.pt \
    --prompt "In this paper we present" \
    --num_samples 3 \
    --max_tokens 200 \
    --temperature 0.8
```

**Key parameters:**
- `--checkpoint`: Path to model checkpoint (`.pt`)
- `--prompt`: Starting text (default: newline)
- `--num_samples`: Number of samples to generate
- `--max_tokens`: Tokens per sample
- `--temperature`: Sampling randomness (0.1=deterministic, 1.5=creative)
- `--top_k`: Top-k sampling (limits vocabulary)

### `merge_lora.py` - Merge LoRA Adapters

Combine LoRA adapters with base model to create standalone checkpoint.

**Why merge?**
- LoRA checkpoint: Small (~5-10 MB), requires base model
- Merged checkpoint: Standalone (~150 MB), no dependencies, faster inference

**Usage:**
```bash
python scripts/merge_lora.py \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt \
    --output_dir models/gpt-38M-tinystories-to-scientific-merged \
    --device cpu
```

**Workflow:**
```bash
# 1. Fine-tune with LoRA
python train.py config/finetune_lora.py

# 2. Merge for deployment
python scripts/merge_lora.py \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt \
    --output_dir models/gpt-38M-tinystories-to-scientific-merged

# 3. Use merged model
python scripts/sample.py --checkpoint models/gpt-38M-tinystories-to-scientific-merged/ckpt.pt
```

### `test_lora.py` - Compare Models

Side-by-side comparison of base model vs LoRA fine-tuned model.

**Usage:**
```bash
python scripts/test_lora.py \
    --base_checkpoint models/gpt-38M-tinystories-pretrain/ckpt.pt \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt \
    --prompt "Once upon a time" \
    --max_tokens 100
```

**Example output:**
```
BASE MODEL (TinyStories):
Once upon a time there was a little girl named Lily...

LORA FINE-TUNED MODEL (Scientific):
Once upon a time, the field of machine learning was primarily focused on...
```

Use this to validate that LoRA fine-tuning successfully adapted the model to a new domain.

### `benchmark_optimized.py` - Performance Benchmark

Compare training speed of base GPT (`model.py`) vs optimized GPT (`model_optimized.py`).

**Usage:**
```bash
# Basic benchmark (float32)
python scripts/benchmark_optimized.py

# With BFloat16 (recommended for modern CPUs)
python scripts/benchmark_optimized.py --dtype bfloat16

# Custom config
python scripts/benchmark_optimized.py \
    --batch_size 8 \
    --block_size 256 \
    --num_iters 50 \
    --dtype bfloat16
```

**Example output:**
```
Base model:      2.5 it/s
Optimized model: 5.0 it/s

Speedup: 2.0x faster
Time savings: 50%

For 160k iterations (scientific training):
  Base model:      64 hours
  Optimized model: 32 hours
  Time saved:      32 hours
```

Use this before starting large training runs to estimate actual training time.

## Common Workflows

### Complete Training Pipeline

```bash
# 1. Pretrain base model
python train.py config/train_tinystories.py

# 2. LoRA fine-tune to new domain
python train.py config/finetune_lora.py

# 3. Compare base vs fine-tuned
python scripts/test_lora.py \
    --base_checkpoint models/gpt-38M-tinystories-pretrain/ckpt.pt \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt

# 4. Merge for deployment
python scripts/merge_lora.py \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt \
    --output_dir models/gpt-38M-tinystories-to-scientific-merged

# 5. Generate samples
python scripts/sample.py \
    --checkpoint models/gpt-38M-tinystories-to-scientific-merged/ckpt.pt \
    --prompt "Abstract: In this paper we" \
    --num_samples 3
```

### Optimized Training

```bash
# 1. Benchmark speedup
python scripts/benchmark_optimized.py --dtype bfloat16

# 2. Train with optimized model (~2x faster)
python train.py config/train_scientific_optimized.py

# 3. Generate samples
python scripts/sample.py \
    --checkpoint models/gpt-38M-scientific-pretrain-optimized/ckpt.pt
```

## Technical Notes

### Path Handling

All scripts automatically add the project root to Python path:
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This allows importing from root (`model.py`, `checkpoint.py`, etc.) regardless of current directory.

### Checkpoint Compatibility

Scripts work with:
- Standard checkpoints (pretraining from scratch)
- LoRA checkpoints (PEFT fine-tuned)
- Merged checkpoints
- Formats: `.pt` (PyTorch) and `.safetensors`

### Sampling Tips

**For deterministic output:**
- `--temperature 0.5` (more focused, less random)
- `--top_k 50` (conservative vocabulary)

**For creative output:**
- `--temperature 1.0-1.2` (more variety)
- `--top_k 200` (broader vocabulary)
- `--num_samples 5` (generate multiple options)

## See Also

- `../MODEL_OPTIMIZED.md` - Detailed guide on optimized GPT architecture
- `../TRAINING_GUIDE.md` - Complete training workflow documentation
- `../models/README.md` - Model naming conventions
- `../config/` - Training configurations
