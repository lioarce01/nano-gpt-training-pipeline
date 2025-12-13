import argparse
import importlib.util
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from safetensors.torch import save_file

from model import GPT, GPTConfig


def load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.config


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train a nanoGPT-style model.")
    parser.add_argument("config", type=str, help="Path to a config Python file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get("out_dir", "out")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg.get("seed", 1337))

    # system settings
    device = cfg.get("device", "cpu")
    device_type = "cuda" if ("cuda" in device and torch.cuda.is_available()) else "cpu"
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(cfg.get("dtype", "float32"), torch.float32)
    if device_type == "cpu" and dtype == torch.float16:
        dtype = torch.float32  # safer fallback on CPU
    compile_model = cfg.get("compile", False)

    # data
    data_dir = os.path.join("data", cfg["dataset"])
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    if not (os.path.exists(train_bin) and os.path.exists(val_bin)):
        raise FileNotFoundError(
            f"Expected preprocessed data at {train_bin} and {val_bin}. "
            f"Run the dataset prepare script first (e.g., python data/{cfg['dataset']}/prepare.py)."
        )
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    # meta info
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        import pickle

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
    else:
        vocab_size = 50257  # GPT-2 vocab size fallback

    # model init
    model_args = dict(
        vocab_size=vocab_size,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.0),
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device_type)

    optimizer = model.configure_optimizers(
        weight_decay=cfg.get("weight_decay", 0.1),
        learning_rate=cfg["learning_rate"],
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
    )

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore

    # helpers
    batch_size = cfg["batch_size"]
    block_size = cfg["block_size"]
    grad_accum_steps = cfg.get("gradient_accumulation_steps", 1)

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(device_type), y.to(device_type)
        return x, y

    def get_lr(iter_num: int):
        if not cfg.get("decay_lr", False):
            return cfg["learning_rate"]
        warmup_iters = cfg.get("warmup_iters", 0)
        lr_decay_iters = cfg.get("lr_decay_iters", cfg["max_iters"])
        min_lr = cfg.get("min_lr", 0.0)
        if iter_num < warmup_iters:
            return cfg["learning_rate"] * iter_num / max(1, warmup_iters)
        if iter_num > lr_decay_iters:
            return min_lr
        decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (cfg["learning_rate"] - min_lr)

    # GradScaler (works on CUDA; enabled=False on CPU or non-fp16)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16 and device_type == "cuda"))
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )

    best_val_loss = float("inf")
    running_loss = 0.0
    model.train()

    def save_json(obj: dict, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def sha256_file(path: str) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    for iter_num in range(cfg["max_iters"] + 1):
        # evaluation
        if iter_num % cfg["eval_interval"] == 0:
            model.eval()
            losses = {"train": 0.0, "val": 0.0}
            with torch.no_grad():
                for split in ["train", "val"]:
                    loss_accum = 0.0
                    for _ in range(cfg["eval_iters"]):
                        xb, yb = get_batch(split)
                        with ctx:
                            _, loss = model(xb, yb)
                        loss_accum += loss.item()
                    losses[split] = loss_accum / cfg["eval_iters"]
            print(f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "config": cfg,
                    "model_args": model_args,
                    "iter": iter_num,
                }
                ckpt_path = os.path.join(out_dir, "ckpt.pt")
                torch.save(ckpt, ckpt_path)

                # save safetensors weights
                st_path = os.path.join(out_dir, "model.safetensors")
                save_file(model.state_dict(), st_path)

                # save configs and metadata for export
                save_json(model_args, os.path.join(out_dir, "model_args.json"))
                save_json(cfg, os.path.join(out_dir, "config.json"))
                export_meta = {
                    "iter": iter_num,
                    "best_val_loss": best_val_loss,
                    "ckpt_pt": os.path.basename(ckpt_path),
                    "safetensors": {
                        "path": os.path.basename(st_path),
                        "sha256": sha256_file(st_path),
                    },
                    "tokenizer": {
                        "name": "gpt2",
                        "vocab_size": vocab_size,
                    },
                }
                save_json(export_meta, os.path.join(out_dir, "export_meta.json"))
            model.train()

        # learning rate schedule
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # training step with grad accumulation
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(grad_accum_steps):
            xb, yb = get_batch("train")
            with ctx:
                _, loss = model(xb, yb)
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if iter_num % cfg["log_interval"] == 0:
            avg_loss = running_loss / max(1, cfg["log_interval"])
            print(
                f"iter {iter_num:06d} | loss {avg_loss:.4f} | lr {lr:.5e} | time {time.strftime('%H:%M:%S')}"
            )
            running_loss = 0.0


if __name__ == "__main__":
    main()

