# nano-gpt (project scaffold)

This project starts a **nanoGPT-style** training pipeline, closely following the layout and practices of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT):

- Explicit config files under `config/`
- Simple data prep scripts under `data/`
- A lean GPT model in `model.py`
- A single entrypoint `train.py` plus a small `sample.py` for inference

## Quick start

1) Create a Python environment with PyTorch 2.x (CUDA if available) and:
   ```
   pip install -r requirements.txt
   ```

2) Prepare Tiny Shakespeare (char-level GPT-2 BPE) data:
   ```
   python data/shakespeare/prepare.py
   ```
   This downloads the dataset, tokenizes with GPT-2 BPE, and writes `data/shakespeare/train.bin` and `val.bin`.

3) Train (single GPU/CPU):
   ```
   python train.py config/train_shakespeare.py
   ```
   Adjust config values in `config/train_shakespeare.py` to change batch size, model dims, or runtime length.

4) Sample from a checkpoint:
   ```
   python sample.py --out_dir=out-shakespeare --start="ROMEO:" --max_new_tokens=80
   ```

## Notes

- The code mirrors nanoGPT practices (config-first, small readable modules, optional `torch.compile`).
- On Windows, if `torch.compile` fails, run with `--compile=False` in your config.
- Extend by adding new configs in `config/` and new dataset prep scripts in `data/<dataset>/prepare.py`.

