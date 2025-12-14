"""
PEFT Utilities for nanoGPT

This module provides wrapper functions for HuggingFace PEFT library,
specifically for LoRA (Low-Rank Adaptation) fine-tuning.

Key functions:
    - create_lora_config: Creates LoRA configuration
    - apply_peft_lora: Applies LoRA to a model using PEFT
    - save_peft_adapters: Saves LoRA adapters to disk
    - load_peft_adapters: Loads LoRA adapters from disk
    - merge_peft_adapters: Merges LoRA weights into base model
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)


def create_lora_config(
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """
    Create a LoRA configuration for PEFT.

    Args:
        rank: LoRA rank (r), smaller = fewer params, typical: 4-32
        alpha: Scaling factor, typical: 2*rank
        dropout: Dropout probability for LoRA path
        target_modules: List of module names to apply LoRA to
                       (e.g., ['c_attn', 'c_proj', 'c_fc'])

    Returns:
        LoraConfig object ready for use with get_peft_model()
    """
    if target_modules is None:
        target_modules = ['c_attn', 'c_proj']

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    return config


def apply_peft_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> PeftModel:
    """
    Apply LoRA to a model using PEFT.

    IMPORTANT: This returns a new PeftModel wrapper around the original model.
    The returned object is a PeftModel, not the original model class.

    Args:
        model: Base model to apply LoRA to
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout
        target_modules: List of module names to target

    Returns:
        PeftModel with LoRA adapters applied
    """
    lora_config = create_lora_config(rank, alpha, dropout, target_modules)

    print(f"\nApplying LoRA via PEFT:")
    print(f"  Rank: {rank}")
    print(f"  Alpha: {alpha}")
    print(f"  Dropout: {dropout}")
    print(f"  Targets: {target_modules}")

    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameters
    peft_model.print_trainable_parameters()

    return peft_model


def save_peft_adapters(
    peft_model: PeftModel,
    adapter_dir: str,
) -> None:
    """
    Save PEFT LoRA adapters to disk.

    This creates a directory with:
        - adapter_config.json: LoRA configuration
        - adapter_model.safetensors: Adapter weights

    Args:
        peft_model: PeftModel with LoRA adapters
        adapter_dir: Directory to save adapters to
    """
    if not isinstance(peft_model, PeftModel):
        raise TypeError(f"Expected PeftModel, got {type(peft_model)}")

    os.makedirs(adapter_dir, exist_ok=True)

    # PEFT's standard save method
    peft_model.save_pretrained(adapter_dir)

    print(f"[OK] PEFT adapters saved to: {adapter_dir}")
    print(f"  - adapter_config.json")
    print(f"  - adapter_model.safetensors")


def load_peft_adapters(
    base_model: nn.Module,
    adapter_dir: str,
    device: str = 'cpu',
) -> PeftModel:
    """
    Load PEFT LoRA adapters from disk.

    Args:
        base_model: Base model (without LoRA)
        adapter_dir: Directory containing adapter files
        device: Device to load model to ('cpu', 'cuda', etc.)

    Returns:
        PeftModel with loaded adapters
    """
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    config_path = os.path.join(adapter_dir, 'adapter_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_dir}\n"
            f"Expected PEFT adapter directory structure."
        )

    # PEFT's standard load method
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=True,  # Allow further fine-tuning
    )

    # Ensure model is on correct device
    peft_model = peft_model.to(device)

    print(f"[OK] PEFT adapters loaded from: {adapter_dir}")

    return peft_model


def merge_peft_adapters(peft_model: PeftModel) -> nn.Module:
    """
    Merge LoRA adapters into base model weights.

    This creates a standard model (no PEFT wrapper) with merged weights:
        W_merged = W_base + (alpha/r) * B @ A

    IMPORTANT: This is destructive - the PeftModel wrapper is removed.
    Use this for inference/deployment, not during training.

    Args:
        peft_model: PeftModel with LoRA adapters

    Returns:
        Base model with merged weights (no LoRA overhead)
    """
    if not isinstance(peft_model, PeftModel):
        raise TypeError(f"Expected PeftModel, got {type(peft_model)}")

    print("\nMerging LoRA adapters into base model...")

    # PEFT's merge_and_unload returns a new model with merged weights
    merged_model = peft_model.merge_and_unload()

    print("[OK] LoRA adapters merged successfully")
    print("  Model is now a standard (non-PEFT) model")

    return merged_model


def get_peft_stats(peft_model: PeftModel) -> dict:
    """
    Get statistics about PEFT LoRA configuration and parameters.

    Args:
        peft_model: PeftModel with LoRA adapters

    Returns:
        Dictionary with stats (rank, alpha, trainable_params, etc.)
    """
    if not isinstance(peft_model, PeftModel):
        raise TypeError(f"Expected PeftModel, got {type(peft_model)}")

    # Get LoRA config (PEFT uses 'default' as the adapter name)
    lora_config = peft_model.peft_config.get('default')

    if lora_config is None:
        return {'error': 'No LoRA config found'}

    # Count parameters
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(
        p.numel() for p in peft_model.parameters() if p.requires_grad
    )

    return {
        'rank': lora_config.r,
        'alpha': lora_config.lora_alpha,
        'dropout': lora_config.lora_dropout,
        'target_modules': lora_config.target_modules,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_params / total_params * 100,
    }
