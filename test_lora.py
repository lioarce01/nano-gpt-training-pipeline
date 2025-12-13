"""
Test and compare LoRA fine-tuned model vs base model.

This script generates text from both the original TinyStories model and the
LoRA-finetuned scientific model to demonstrate domain adaptation.
"""

import torch
import tiktoken
from checkpoint import CheckpointManager


def generate_text(model, prompt: str, max_tokens: int = 100, temperature: float = 0.8, device: str = "cpu"):
    """Generate text from a prompt."""
    enc = tiktoken.get_encoding("gpt2")

    # Encode prompt
    tokens = enc.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop to block_size if needed
            idx_cond = input_ids if input_ids.size(1) <= model.config.block_size else input_ids[:, -model.config.block_size:]

            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Sample
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if we hit a natural break (optional)
            if next_token.item() == enc.encode(".")[0]:
                # Give it a few more tokens after period
                if input_ids.size(1) > len(tokens) + 20:
                    break

    # Decode
    generated_tokens = input_ids[0].tolist()
    text = enc.decode(generated_tokens)

    return text


def main():
    device = "cpu"

    # Load both models
    print("="*70)
    print("MODEL COMPARISON: Base TinyStories vs LoRA Scientific")
    print("="*70)

    # Load models using unified checkpoint manager
    base_model, _ = CheckpointManager.load_model("out-tinystories/ckpt.pt", device)
    base_model.eval()

    lora_model, _ = CheckpointManager.load_model("out-lora-scientific-merged/ckpt.pt", device)
    lora_model.eval()

    # Test prompts
    prompts = [
        "In this paper, we propose",
        "The experiment demonstrates",
        "Our results show that",
        "Once upon a time",
    ]

    print("\n" + "="*70)
    print("TEXT GENERATION COMPARISON")
    print("="*70)

    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"PROMPT: \"{prompt}\"")
        print(f"{'='*70}\n")

        # Generate from base model
        print("BASE MODEL (TinyStories):")
        print("-" * 70)
        base_text = generate_text(base_model, prompt, max_tokens=80, temperature=0.8, device=device)
        print(base_text)
        print()

        # Generate from LoRA model
        print("LORA MODEL (Scientific fine-tuned):")
        print("-" * 70)
        lora_text = generate_text(lora_model, prompt, max_tokens=80, temperature=0.8, device=device)
        print(lora_text)
        print()

    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
Expected behavior:
- Base model: Trained on TinyStories → generates children's story language
- LoRA model: Fine-tuned on arXiv abstracts → should adapt toward scientific language

If LoRA worked correctly:
- Scientific prompts → LoRA model should continue with technical vocabulary
- Story prompts → Base model should be more coherent for stories
- LoRA model should show domain shift (less story-like, more formal/technical)

Note: If both models generate similar text, LoRA might need more training iterations
or the validation loss might indicate the training didn't converge yet.
    """)


if __name__ == "__main__":
    main()
