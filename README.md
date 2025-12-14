# nano-GPT: Modern GPT Training Pipeline

A lightweight, educational GPT implementation for CPU training with modern optimizations and LoRA fine-tuning support.

Based on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) with 2025 architectural improvements for efficient CPU training.

## Features

- **Two model architectures:**
  - `model.py`: Standard GPT-2 style (baseline)
  - `model_optimized.py`: Modern optimizations (RMSNorm, RoPE, GQA, SwiGLU) - **~2x faster**

- **LoRA fine-tuning:**
  - Parameter-efficient fine-tuning with HuggingFace PEFT
  - Adapt pretrained models to new domains with minimal compute

- **CPU-optimized training:**
  - BFloat16 mixed precision (40-60% speedup on modern CPUs)
  - Optimized for Ryzen 7000+ and Intel 12th gen+ processors

- **Production-ready tools:**
  - Unified checkpoint management (`.pt` and `.safetensors`)
  - Text generation, model merging, benchmarking scripts
  - Clean project structure with organized configs

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch 2.9.1+ (CPU or CUDA)
- HuggingFace PEFT, transformers, datasets
- tiktoken, numpy, safetensors

### 2. Prepare Dataset

Choose a dataset to train on:

**TinyStories (children's stories):**
```bash
python data/tinystories/prepare.py
```

**Scientific Papers (arXiv abstracts):**
```bash
# Fast download via HuggingFace (500k samples, ~310 MB)
python data/scientific_papers/prepare_hf.py --num_samples 500000

# Or smaller dataset for testing
python data/scientific_papers/prepare_hf.py --num_samples 100000
```

**Shakespeare (character-level):**
```bash
python data/shakespeare/prepare.py
```

### 3. Train a Model

**Option A: Standard GPT (baseline)**
```bash
python train.py config/train_scientific.py
```

**Option B: Optimized GPT (~2x faster)**
```bash
python train.py config/train_scientific_optimized.py
```

Training will save checkpoints to `models/` directory.

### 4. Generate Text

```bash
python scripts/sample.py \
    --checkpoint models/gpt-38M-scientific-pretrain/ckpt.pt \
    --prompt "In this paper we present" \
    --num_samples 3 \
    --max_tokens 150
```

## Project Structure

```
nano-gpt/
├── model.py                    # Base GPT-2 architecture
├── model_optimized.py          # Optimized GPT with modern improvements
├── train.py                    # Training entrypoint
├── checkpoint.py               # Unified checkpoint management
├── peft_utils.py               # LoRA utilities
│
├── config/                     # Training configurations
│   ├── train_scientific.py            # Pretrain on scientific papers (base model)
│   ├── train_scientific_optimized.py  # Pretrain on scientific papers (optimized)
│   ├── train_tinystories.py           # Pretrain on TinyStories
│   └── finetune_lora.py               # LoRA fine-tuning
│
├── data/                       # Dataset preparation scripts
│   ├── scientific_papers/
│   │   ├── prepare_hf.py       # Fast HuggingFace download
│   │   └── prepare.py          # Slow arXiv API download
│   ├── tinystories/
│   └── shakespeare/
│
├── scripts/                    # Utility tools
│   ├── sample.py               # Generate text from trained models
│   ├── merge_lora.py           # Merge LoRA adapters into base model
│   ├── test_lora.py            # Compare base vs fine-tuned models
│   └── benchmark_optimized.py  # Benchmark base vs optimized models
│
├── models/                     # Trained model checkpoints
│   ├── gpt-38M-scientific-pretrain/
│   ├── gpt-38M-scientific-pretrain-optimized/
│   ├── gpt-38M-tinystories-pretrain/
│   └── gpt-38M-tinystories-to-scientific-lora/
│
└── docs/
    ├── TRAINING_GUIDE.md       # Complete training workflow guide
    ├── MODEL_OPTIMIZED.md      # Optimized model documentation
    └── ...
```

## Common Workflows

### Workflow 1: Pretrain from Scratch

```bash
# 1. Download dataset (500k scientific papers)
python data/scientific_papers/prepare_hf.py --num_samples 500000

# 2. Benchmark speedup (optional)
python scripts/benchmark_optimized.py --dtype bfloat16

# 3. Train optimized model (~32 hours on Ryzen 7 7800X3D)
python train.py config/train_scientific_optimized.py

# 4. Generate samples
python scripts/sample.py \
    --checkpoint models/gpt-38M-scientific-pretrain-optimized/ckpt.pt \
    --prompt "Abstract: In this paper we investigate"
```

### Workflow 2: LoRA Fine-Tuning

Adapt a pretrained model to a new domain efficiently:

```bash
# 1. Pretrain on TinyStories
python train.py config/train_tinystories.py

# 2. LoRA fine-tune to scientific domain
#    Edit config/finetune_lora.py to set:
#    - resume_dir: "models/gpt-38M-tinystories-pretrain"
#    - out_dir: "models/gpt-38M-tinystories-to-scientific-lora"
python train.py config/finetune_lora.py

# 3. Compare base vs fine-tuned
python scripts/test_lora.py \
    --base_checkpoint models/gpt-38M-tinystories-pretrain/ckpt.pt \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt

# 4. Merge LoRA adapters for deployment
python scripts/merge_lora.py \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt \
    --output_dir models/gpt-38M-tinystories-to-scientific-merged
```

## Model Architectures

### Base GPT (`model.py`)

Standard GPT-2 style transformer:
- **Architecture**: MHA (Multi-Head Attention), LayerNorm, GELU, learned position embeddings
- **Parameters**: 38.4M (5 layers, 5 heads, 320 embedding dim)
- **Training speed**: ~2.5 it/s (Ryzen 7 7800X3D, CPU, float32)
- **Use case**: Baseline, educational, GPT-2 compatibility

### Optimized GPT (`model_optimized.py`)

Modern transformer with 2025 improvements:
- **Architecture**: GQA (Grouped Query Attention), RMSNorm, SwiGLU, RoPE
- **Parameters**: ~36M (slightly fewer due to no position embeddings)
- **Training speed**: ~5 it/s (Ryzen 7 7800X3D, CPU, bfloat16)
- **Speedup**: **~2x faster** than base model
- **Quality**: 10-20% better loss at same iteration count
- **Use case**: Production training, research, fast iteration

**Key improvements:**
1. **RMSNorm**: 10-15% faster than LayerNorm
2. **RoPE**: Better position encoding, no learned params
3. **GQA**: 15-20% faster, 60% less KV cache memory
4. **SwiGLU**: Better convergence, fewer iterations needed
5. **Pre-normalization**: Better gradient flow
6. **BFloat16**: 40-60% speedup on modern CPUs

See `MODEL_OPTIMIZED.md` for detailed architecture comparison.

## Training Configurations

All configs in `config/` directory:

| Config | Model | Dataset | Time (Ryzen 7800X3D) | Purpose |
|--------|-------|---------|----------------------|---------|
| `train_scientific.py` | Base GPT | 500k papers | ~64 hours | Baseline scientific pretrain |
| `train_scientific_optimized.py` | Optimized GPT | 500k papers | ~32 hours | Fast scientific pretrain |
| `train_tinystories.py` | Base GPT | TinyStories | ~20 hours | Children's stories pretrain |
| `finetune_lora.py` | Any | Any | ~2-5 hours | LoRA domain adaptation |

## Benchmarking

Compare base vs optimized model before training:

```bash
# Float32 (baseline)
python scripts/benchmark_optimized.py

# BFloat16 (recommended)
python scripts/benchmark_optimized.py --dtype bfloat16

# Custom config
python scripts/benchmark_optimized.py \
    --batch_size 8 \
    --block_size 256 \
    --num_iters 50 \
    --dtype bfloat16
```

**Expected results (Ryzen 7 7800X3D):**
- Base model (float32): 2.5 it/s
- Optimized model (float32): 3.2 it/s (1.3x faster)
- Optimized model (bfloat16): 5.0 it/s (2.0x faster)

## Hardware Requirements

**Minimum:**
- CPU: Any modern x86_64 processor
- RAM: 8 GB
- Storage: 2 GB for datasets + 500 MB per model

**Recommended:**
- CPU: Ryzen 7000+ or Intel 12th gen+ (BFloat16 support)
- RAM: 16 GB
- Storage: 10 GB

**Training time estimates (38M model, 160k iterations):**
- Entry CPU (no BFloat16): ~100 hours
- Modern CPU (with BFloat16, base model): ~64 hours
- Modern CPU (with BFloat16, optimized model): ~32 hours

**Note:** No GPU required. All training is CPU-optimized.

## Documentation

- `TRAINING_GUIDE.md` - Complete training workflow guide
- `MODEL_OPTIMIZED.md` - Optimized architecture deep dive
- `scripts/README.md` - Utility scripts documentation
- `models/README.md` - Model naming conventions

## Credits

Based on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

Modern optimizations inspired by:
- LLaMA (Meta)
- Mistral (Mistral AI)
- Gemma (Google)

## License

MIT License (same as original nanoGPT)
