"""
Merge LoRA weights into base model for inference.

After LoRA fine-tuning, this script combines the LoRA adapter weights (A, B)
with the frozen base weights (W) to create a single merged model:
    W_merged = W_base + (alpha/r) * B @ A

The merged model has the same architecture as the original GPT model
(no LoRA overhead) and can be used for inference with sample.py or other scripts.
"""

import argparse
from checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default="out-lora-scientific/ckpt.pt",
        help="Path to LoRA fine-tuned checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out-lora-scientific-merged",
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)"
    )

    args = parser.parse_args()

    # Merge using unified checkpoint manager
    CheckpointManager.merge_and_save_lora(
        lora_ckpt_path=args.lora_checkpoint,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
