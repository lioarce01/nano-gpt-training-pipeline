"""
Unified checkpoint manager for GPT models.

Handles loading and saving checkpoints in different scenarios:
- Standard training (from scratch)
- LoRA fine-tuning (with adapter layers)
- Inference (standard or LoRA models)
- Merging LoRA weights

Automatically detects checkpoint type and handles LoRA structure.
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from model import GPT, GPTConfig


class CheckpointManager:
    """
    Unified checkpoint manager for loading and saving GPT models.

    Automatically handles:
    - Standard checkpoints (vanilla GPT)
    - LoRA checkpoints (GPT with LoRA adapters)
    - Metadata and configuration
    - SafeTensors export
    """

    @staticmethod
    def detect_checkpoint_type(ckpt_path: str) -> str:
        """
        Detect if checkpoint contains LoRA weights.

        Args:
            ckpt_path: Path to checkpoint file

        Returns:
            'lora' if checkpoint has LoRA structure, 'standard' otherwise
        """
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)

        # Check for LoRA keys in state dict
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower() or 'base_linear' in k.lower()]

        return 'lora' if lora_keys else 'standard'

    @staticmethod
    def load_model(
        ckpt_path: str,
        device: str = 'cpu',
        apply_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GPT, Dict[str, Any]]:
        """
        Load a GPT model from checkpoint.

        Automatically handles:
        - Standard checkpoints → loads as GPT
        - LoRA checkpoints → loads with LoRA structure
        - Can apply LoRA to standard checkpoints (for fine-tuning)

        Args:
            ckpt_path: Path to checkpoint file
            device: Device to load model on
            apply_lora: If True, apply LoRA to a standard checkpoint
            lora_config: LoRA configuration if apply_lora=True
                        {rank, alpha, dropout, targets}

        Returns:
            (model, metadata) tuple
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"\nLoading checkpoint: {ckpt_path}")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Extract model args
        model_args = checkpoint.get('model_args', {})
        if not model_args:
            raise ValueError("Checkpoint missing 'model_args'. Invalid checkpoint format.")

        # Detect checkpoint type
        ckpt_type = CheckpointManager.detect_checkpoint_type(ckpt_path)
        print(f"Checkpoint type: {ckpt_type}")

        # Create model config
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        # Load state dict based on type
        if ckpt_type == 'lora':
            # LoRA checkpoint - load with strict=False (missing/extra keys expected)
            print("Loading LoRA checkpoint (contains adapter weights)...")
            missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)

            # Verify we have LoRA keys
            lora_keys = [k for k in unexpected if 'lora' in k.lower()]
            if not lora_keys:
                print("WARNING: Detected as LoRA but no LoRA keys found in unexpected!")

            print(f"  Loaded with {len(lora_keys)} LoRA parameter groups")

        else:
            # Standard checkpoint
            print("Loading standard checkpoint...")
            model.load_state_dict(checkpoint['model'])

            # Apply LoRA if requested (for fine-tuning)
            if apply_lora:
                if lora_config is None:
                    raise ValueError("apply_lora=True but no lora_config provided!")

                print("\nApplying LoRA to loaded model...")
                model.apply_lora(
                    rank=lora_config.get('rank', 8),
                    alpha=lora_config.get('alpha', 16.0),
                    dropout=lora_config.get('dropout', 0.0),
                    target_modules=lora_config.get('targets', ['c_attn', 'c_proj']),
                )

        model.to(device)

        # Extract metadata
        metadata = {
            'checkpoint_type': ckpt_type,
            'model_args': model_args,
            'config': checkpoint.get('config', {}),
            'iter': checkpoint.get('iter', 0),
            'best_val_loss': checkpoint.get('best_val_loss', None),
        }

        total_params = model.get_num_params()
        trainable_params = model.get_num_params(trainable_only=True)

        print(f"\nModel loaded successfully:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        if trainable_params < total_params:
            print(f"  Frozen parameters: {total_params - trainable_params:,}")
            print(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")

        return model, metadata

    @staticmethod
    def save_checkpoint(
        model: GPT,
        save_dir: str,
        metadata: Dict[str, Any],
        save_safetensors: bool = True,
    ):
        """
        Save model checkpoint with metadata.

        Saves:
        - PyTorch checkpoint (ckpt.pt)
        - SafeTensors weights (model.safetensors) [optional]
        - Model args JSON (model_args.json)
        - Config JSON (config.json)
        - Export metadata (export_meta.json)

        Args:
            model: GPT model to save
            save_dir: Directory to save checkpoint in
            metadata: Dictionary with config, iter, best_val_loss, etc.
            save_safetensors: Whether to also save SafeTensors format
        """
        os.makedirs(save_dir, exist_ok=True)

        # Get state dict
        state_dict = model.state_dict()

        # Detect if this is a LoRA model
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower() or 'base_linear' in k.lower()]
        is_lora = len(lora_keys) > 0

        # Prepare checkpoint
        checkpoint = {
            'model': state_dict,
            'model_args': metadata.get('model_args', {}),
            'config': metadata.get('config', {}),
            'iter': metadata.get('iter', 0),
            'best_val_loss': metadata.get('best_val_loss', None),
            'is_lora': is_lora,
        }

        if is_lora:
            # Add LoRA config to checkpoint
            checkpoint['lora_config'] = {
                'rank': model.config.lora_rank,
                'alpha': model.config.lora_alpha,
                'dropout': model.config.lora_dropout,
                'targets': list(model.config.lora_targets),
            }

        # Save PyTorch checkpoint
        ckpt_path = os.path.join(save_dir, 'ckpt.pt')
        torch.save(checkpoint, ckpt_path)
        print(f"✓ Saved checkpoint: {ckpt_path}")

        # Save SafeTensors (optional, for compatibility)
        if save_safetensors:
            st_path = os.path.join(save_dir, 'model.safetensors')
            save_file(state_dict, st_path)
            print(f"✓ Saved SafeTensors: {st_path}")

            # Compute SHA256 for SafeTensors
            sha256_hash = CheckpointManager._sha256_file(st_path)
        else:
            sha256_hash = None

        # Save JSON metadata
        model_args_path = os.path.join(save_dir, 'model_args.json')
        with open(model_args_path, 'w') as f:
            json.dump(metadata.get('model_args', {}), f, indent=2)
        print(f"✓ Saved model args: {model_args_path}")

        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(metadata.get('config', {}), f, indent=2)
        print(f"✓ Saved config: {config_path}")

        # Export metadata
        export_meta = {
            'iter': metadata.get('iter', 0),
            'best_val_loss': metadata.get('best_val_loss', None),
            'ckpt_pt': 'ckpt.pt',
            'is_lora': is_lora,
        }

        if save_safetensors and sha256_hash:
            export_meta['safetensors'] = {
                'path': 'model.safetensors',
                'sha256': sha256_hash,
            }

        if is_lora:
            export_meta['lora_config'] = checkpoint['lora_config']

        export_meta_path = os.path.join(save_dir, 'export_meta.json')
        with open(export_meta_path, 'w') as f:
            json.dump(export_meta, f, indent=2)
        print(f"✓ Saved export metadata: {export_meta_path}")

        checkpoint_type = "LoRA checkpoint" if is_lora else "Standard checkpoint"
        print(f"\n{checkpoint_type} saved to: {save_dir}")

    @staticmethod
    def merge_and_save_lora(
        lora_ckpt_path: str,
        output_dir: str,
        device: str = 'cpu',
    ):
        """
        Load LoRA checkpoint, merge weights, and save as standard checkpoint.

        Args:
            lora_ckpt_path: Path to LoRA checkpoint
            output_dir: Where to save merged checkpoint
            device: Device to use
        """
        print(f"\n{'='*70}")
        print("MERGING LORA WEIGHTS")
        print(f"{'='*70}\n")

        # Load LoRA model
        model, metadata = CheckpointManager.load_model(lora_ckpt_path, device=device)

        # Merge LoRA weights
        print("\nMerging LoRA weights into base model...")
        model.merge_lora()
        print("✓ LoRA weights merged")

        # Update metadata
        metadata['model_args'] = {
            'vocab_size': model.config.vocab_size,
            'block_size': model.config.block_size,
            'n_layer': model.config.n_layer,
            'n_head': model.config.n_head,
            'n_embd': model.config.n_embd,
            'dropout': model.config.dropout,
            'bias': model.config.bias,
        }

        # Save as standard checkpoint
        CheckpointManager.save_checkpoint(model, output_dir, metadata)

        print(f"\n{'='*70}")
        print("MERGE COMPLETE")
        print(f"{'='*70}\n")

        return model

    @staticmethod
    def _sha256_file(path: str) -> str:
        """Compute SHA256 hash of a file."""
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()


# Convenience functions
def load_model_for_training(ckpt_path: str, device: str = 'cpu', lora_config: Optional[Dict] = None):
    """
    Load model for training/fine-tuning.

    Args:
        ckpt_path: Checkpoint path or 'scratch' for new model
        device: Device
        lora_config: If provided, apply LoRA for fine-tuning

    Returns:
        model, metadata
    """
    if ckpt_path == 'scratch':
        # Return None, caller should initialize new model
        return None, {}

    apply_lora = lora_config is not None
    return CheckpointManager.load_model(ckpt_path, device, apply_lora, lora_config)


def load_model_for_inference(ckpt_path: str, device: str = 'cpu'):
    """
    Load model for inference (generation).

    Automatically handles LoRA checkpoints.

    Args:
        ckpt_path: Checkpoint path
        device: Device

    Returns:
        model (ready for inference)
    """
    model, _ = CheckpointManager.load_model(ckpt_path, device)
    model.eval()
    return model
