import sys
import argparse
import os
import json
import pickle

# Add parent directory to path to import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tiktoken
from safetensors.torch import load_file

from model import GPT, GPTConfig


def load_checkpoint(out_dir: str):
    st_path = os.path.join(out_dir, "model.safetensors")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    model_args_path = os.path.join(out_dir, "model_args.json")
    config_path = os.path.join(out_dir, "config.json")

    data = {}
    state_dict = None

    # Prefer safetensors + json configs
    if os.path.exists(st_path) and os.path.exists(model_args_path):
        state_dict = load_file(st_path, device="cpu")
        with open(model_args_path, "r", encoding="utf-8") as f:
            data["model_args"] = json.load(f)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data["config"] = json.load(f)
    elif os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location="cpu")
        state_dict = data["model"]
    else:
        raise FileNotFoundError(f"No checkpoint found in {out_dir}")

    return data, state_dict


def main():
    parser = argparse.ArgumentParser(description="Sample from a trained nanoGPT-style model.")
    parser.add_argument("--out_dir", type=str, default="out-shakespeare", help="Checkpoint directory")
    parser.add_argument("--start", type=str, default="ROMEO:", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0, help="0 = disabled")
    parser.add_argument("--top_p", type=float, default=1.0, help="1.0 = disabled")
    args = parser.parse_args()

    data, state_dict = load_checkpoint(args.out_dir)
    model_args = data.get("model_args", {})

    # dataset info
    cfg = data.get("config", {})
    dataset = cfg.get("dataset", "shakespeare")
    meta_path = os.path.join("data", dataset, "meta.pkl")
    vocab_size = 50257
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta.get("vocab_size", vocab_size)
    model_args["vocab_size"] = vocab_size

    config = GPTConfig(**model_args)
    model = GPT(config)
    model.load_state_dict(state_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode(args.start)
    start_ids = start_ids[-config.block_size :]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    def top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float):
        # logits: (batch, vocab)
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            thresh = v[:, -1].unsqueeze(1)
            logits = torch.where(logits < thresh, torch.tensor(float("-inf"), device=logits.device), logits)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            mask = cumulative_probs > top_p
            mask[..., 0] = False
            sorted_logits[mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, sorted_indices, sorted_logits)
        return logits

    for _ in range(args.num_samples):
        with torch.no_grad():
            generated = x
            for _ in range(args.max_new_tokens):
                idx_cond = generated[:, -config.block_size :]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / max(args.temperature, 1e-6)
                logits = top_k_top_p_filter(logits, args.top_k, args.top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
            text = enc.decode(generated[0].tolist())
        print("---- SAMPLE ----")
        print(text)


if __name__ == "__main__":
    main()

