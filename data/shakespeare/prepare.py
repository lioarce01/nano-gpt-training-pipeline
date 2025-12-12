import os
import pickle
import urllib.request

import numpy as np
import tiktoken


def download_shakespeare(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    input_file_path = os.path.join(data_dir, "input.txt")
    if os.path.exists(input_file_path):
        return input_file_path

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, input_file_path)
    return input_file_path


def main():
    data_dir = os.path.dirname(__file__)
    input_path = download_shakespeare(data_dir)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters.")
    enc = tiktoken.get_encoding("gpt2")

    tokens = enc.encode_ordinary(text)
    print(f"Tokenized to {len(tokens):,} tokens.")
    data = np.array(tokens, dtype=np.uint16)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_data.tofile(os.path.join(data_dir, "train.bin"))
    val_data.tofile(os.path.join(data_dir, "val.bin"))
    print(f"Wrote train.bin ({train_data.nbytes/1e6:.1f} MB) and val.bin ({val_data.nbytes/1e6:.1f} MB)")

    meta = {"vocab_size": enc.n_vocab}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("Saved meta.pkl with vocab_size", enc.n_vocab)


if __name__ == "__main__":
    main()

