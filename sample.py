import argparse
import os
import pickle

import torch
import tiktoken

from model import GPT, GPTConfig


def load_checkpoint(out_dir: str):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description="Sample from a trained nanoGPT-style model.")
    parser.add_argument("--out_dir", type=str, default="out-shakespeare", help="Checkpoint directory")
    parser.add_argument("--start", type=str, default="ROMEO:", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    ckpt = load_checkpoint(args.out_dir)
    model_args = ckpt["model_args"]

    dataset = ckpt.get("config", {}).get("dataset", "shakespeare")
    meta_path = os.path.join("data", dataset, "meta.pkl")
    vocab_size = 50257
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta.get("vocab_size", vocab_size)

    model_args["vocab_size"] = vocab_size
    config = GPTConfig(**model_args)
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode(args.start)
    start_ids = start_ids[-config.block_size :]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    for _ in range(args.num_samples):
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=args.max_new_tokens)[0]
        text = enc.decode(y.tolist())
        print("---- SAMPLE ----")
        print(text)


if __name__ == "__main__":
    main()

