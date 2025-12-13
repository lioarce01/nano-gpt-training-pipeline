import os
import pickle
import urllib.request

import numpy as np
import tiktoken


# Try multiple mirrors to avoid 404s
TRAIN_URLS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/TinyStories-train.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/TinyStories-train.txt?download=1",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=1",
]
VAL_URLS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/TinyStories-valid.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/TinyStories-valid.txt?download=1",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt?download=1",
]


def try_download(urls, dest: str):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last_err = None
    for url in urls:
        try:
            print(f"Downloading {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"Failed {url}: {e}")
    raise RuntimeError(f"All download attempts failed for {dest}") from last_err


def prepare():
    data_dir = os.path.dirname(__file__)
    train_txt = os.path.join(data_dir, "train.txt")
    val_txt = os.path.join(data_dir, "val.txt")

    try_download(TRAIN_URLS, train_txt)
    try_download(VAL_URLS, val_txt)

    enc = tiktoken.get_encoding("gpt2")

    def tokenize_file(path: str) -> np.ndarray:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded {path}, {len(text):,} chars")
        tokens = enc.encode_ordinary(text)
        print(f"Tokenized {len(tokens):,} tokens")
        return np.array(tokens, dtype=np.uint16)

    train_tokens = tokenize_file(train_txt)
    val_tokens = tokenize_file(val_txt)

    train_tokens.tofile(os.path.join(data_dir, "train.bin"))
    val_tokens.tofile(os.path.join(data_dir, "val.bin"))
    print(
        f"Wrote train.bin ({train_tokens.nbytes/1e6:.1f} MB) and val.bin ({val_tokens.nbytes/1e6:.1f} MB)"
    )

    meta = {"vocab_size": enc.n_vocab, "dataset": "tinystories"}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("Saved meta.pkl with vocab_size", enc.n_vocab)


if __name__ == "__main__":
    prepare()

