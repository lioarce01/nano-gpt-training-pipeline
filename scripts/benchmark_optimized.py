"""
Benchmark: Compare base GPT vs OptimizedGPT training speed.

This script measures iterations per second for both models to demonstrate
the speedup from modern optimizations (RMSNorm, RoPE, GQA, SwiGLU).

Expected results:
- Base model: ~2-3 it/s (Ryzen 7 7800X3D, CPU, float32)
- Optimized model (float32): ~3-4 it/s (20-30% faster)
- Optimized model (bfloat16): ~5-6 it/s (80-100% faster)

Usage:
    python scripts/benchmark_optimized.py
    python scripts/benchmark_optimized.py --dtype bfloat16
    python scripts/benchmark_optimized.py --batch_size 8
"""

import sys
import os
import argparse
import time

# Add parent directory to path to import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from model import GPT, GPTConfig
from model_optimized import OptimizedGPT, OptimizedGPTConfig


def benchmark_model(model, batch_size: int, block_size: int, num_iters: int, dtype, device: str):
    """Benchmark training iterations for a model."""
    model.to(device)
    model.train()

    # Create dummy data
    x = torch.randint(0, model.config.vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, model.config.vocab_size, (batch_size, block_size), device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup (not timed)
    print("  Warming up...")
    for _ in range(5):
        with torch.amp.autocast(device_type=device, dtype=dtype):
            logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Benchmark
    print(f"  Running {num_iters} iterations...")
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()

    for _ in range(num_iters):
        with torch.amp.autocast(device_type=device, dtype=dtype):
            logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()

    elapsed = end_time - start_time
    iters_per_sec = num_iters / elapsed

    return iters_per_sec, elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark base GPT vs OptimizedGPT")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--block_size", type=int, default=256, help="Sequence length")
    parser.add_argument("--n_layer", type=int, default=5, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=5, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=320, help="Embedding dimension")
    parser.add_argument("--num_iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="Data type")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Check BFloat16 support on CPU
    if args.device == "cpu" and args.dtype == "bfloat16":
        if not torch.cpu._is_bf16_supported():
            print("WARNING: BFloat16 not supported on this CPU, falling back to float32")
            dtype = torch.float32
            args.dtype = "float32"

    print("=" * 70)
    print("BENCHMARK: Base GPT vs OptimizedGPT")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Block size: {args.block_size}")
    print(f"  Layers: {args.n_layer}")
    print(f"  Heads: {args.n_head}")
    print(f"  Embedding dim: {args.n_embd}")
    print(f"  Data type: {args.dtype}")
    print(f"  Device: {args.device}")
    print(f"  Benchmark iterations: {args.num_iters}")
    print()

    # Create base model
    print("-" * 70)
    print("1. BASE GPT MODEL")
    print("-" * 70)

    base_config = GPTConfig(
        vocab_size=50257,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
    )
    base_model = GPT(base_config)
    base_params = sum(p.numel() for p in base_model.parameters())

    print(f"  Parameters: {base_params:,} ({base_params/1e6:.1f}M)")
    print()

    base_iters_per_sec, base_elapsed = benchmark_model(
        base_model, args.batch_size, args.block_size, args.num_iters, dtype, args.device
    )

    print(f"  Results:")
    print(f"    Total time: {base_elapsed:.2f}s")
    print(f"    Iterations/sec: {base_iters_per_sec:.2f} it/s")
    print(f"    Time per iteration: {1000 / base_iters_per_sec:.1f} ms/it")
    print()

    # Create optimized model
    print("-" * 70)
    print("2. OPTIMIZED GPT MODEL (RMSNorm + RoPE + GQA + SwiGLU)")
    print("-" * 70)

    opt_config = OptimizedGPTConfig(
        vocab_size=50257,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        n_kv_head=max(1, args.n_head // 2),  # GQA with 2:1 Q:KV ratio
        bias=False,
    )
    opt_model = OptimizedGPT(opt_config)
    opt_params = sum(p.numel() for p in opt_model.parameters())

    print(f"  Parameters: {opt_params:,} ({opt_params/1e6:.1f}M)")
    print(f"  GQA: {opt_config.n_head} Q heads, {opt_config.n_kv_head} KV heads")
    print()

    opt_iters_per_sec, opt_elapsed = benchmark_model(
        opt_model, args.batch_size, args.block_size, args.num_iters, dtype, args.device
    )

    print(f"  Results:")
    print(f"    Total time: {opt_elapsed:.2f}s")
    print(f"    Iterations/sec: {opt_iters_per_sec:.2f} it/s")
    print(f"    Time per iteration: {1000 / opt_iters_per_sec:.1f} ms/it")
    print()

    # Comparison
    speedup = opt_iters_per_sec / base_iters_per_sec
    time_savings = (1 - 1/speedup) * 100

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Base model:      {base_iters_per_sec:.2f} it/s")
    print(f"  Optimized model: {opt_iters_per_sec:.2f} it/s")
    print()
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Time savings: {time_savings:.1f}%")
    print()
    print(f"  For 160k iterations (scientific training):")
    print(f"    Base model:      {160000 / base_iters_per_sec / 3600:.1f} hours")
    print(f"    Optimized model: {160000 / opt_iters_per_sec / 3600:.1f} hours")
    print(f"    Time saved:      {160000 * (1/base_iters_per_sec - 1/opt_iters_per_sec) / 3600:.1f} hours")
    print()

    # Recommendations
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if args.dtype == "float32":
        print("  [TIP] Try running with --dtype bfloat16 for additional 40-60% speedup:")
        print("        python scripts/benchmark_optimized.py --dtype bfloat16")
        print()

    if speedup < 1.3:
        print("  [NOTE] Speedup is lower than expected. This could be due to:")
        print("    - CPU-specific optimizations not enabled")
        print("    - Small model size (overhead dominates)")
        print("    - Running in debug mode or with profiling enabled")
        print()
    elif speedup >= 1.8:
        print("  [SUCCESS] Great speedup! The optimized model is significantly faster.")
        print("            Use config/train_scientific_optimized.py for full training.")
        print()


if __name__ == "__main__":
    main()
