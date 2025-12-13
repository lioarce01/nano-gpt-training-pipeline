"""
LoRA (Low-Rank Adaptation) implementation for fine-tuning.

LoRA adds trainable low-rank matrices to frozen pre-trained weights:
    y = W·x + (α/r)·B·A·x

Where:
    W: Frozen pre-trained weights (d_out × d_in)
    A: Trainable matrix (r × d_in), initialized randomly
    B: Trainable matrix (d_out × r), initialized to zero
    r: Rank (typically 4, 8, 16, 32)
    α: Scaling factor (typically 2r)

Benefits:
    - 100x fewer trainable parameters
    - Same performance as full fine-tuning
    - Multiple adapters can share same base model
    - Can be merged back into W for inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps an existing Linear layer.

    Adds low-rank adaptation: ΔW = B @ A
    Forward pass: output = W @ x + (alpha/r) * B @ A @ x

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (r), smaller = fewer params, typical: 4-32
        alpha: Scaling factor, typical: 2*rank
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # LoRA trainable matrices
        # A: (rank × in_features) - initialized with random Gaussian
        # B: (out_features × rank) - initialized to zero
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Scaling factor: alpha / rank
        self.scaling = alpha / rank

        # Dropout (applied to LoRA path only, not main weights)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize LoRA matrices:
        - A: Kaiming uniform (same as nn.Linear default)
        - B: Zero (ensures ΔW = 0 at start, model unchanged initially)
        """
        # A ~ Uniform(-1/√k, 1/√k) where k = in_features
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B = 0 (critical: ensures initial ΔW = B @ A = 0)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA forward pass: (alpha/r) * B @ A @ x

        Args:
            x: Input tensor (..., in_features)

        Returns:
            LoRA output (..., out_features)
        """
        # x: (batch, seq_len, in_features) or (batch, in_features)

        # Apply dropout to input of LoRA path
        x_lora = self.dropout(x)

        # Efficient computation: B @ (A @ x)
        # Step 1: A @ x  → (batch, ..., rank)
        # Step 2: B @ result → (batch, ..., out_features)
        result = F.linear(x_lora, self.lora_A)  # A @ x
        result = F.linear(result, self.lora_B)   # B @ (A @ x)

        # Scale by alpha/rank
        result = result * self.scaling

        return result


class LinearWithLoRA(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adaptation.

    Combines frozen base weights with trainable LoRA matrices:
        output = base_linear(x) + lora(x)

    Args:
        base_linear: Frozen nn.Linear layer to adapt
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store base layer (will be frozen)
        self.base_linear = base_linear

        # Freeze base layer parameters
        for param in self.base_linear.parameters():
            param.requires_grad = False

        # Add LoRA layer
        self.lora = LoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Store config for later inspection
        self.rank = rank
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base output + LoRA adaptation

        output = W @ x + (alpha/r) * B @ A @ x
        """
        # Base model forward (frozen)
        base_output = self.base_linear(x)

        # LoRA adaptation (trainable)
        lora_output = self.lora(x)

        # Combine
        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into base weights for inference.

        Creates a new Linear layer with weights:
            W_merged = W_base + (alpha/r) * B @ A

        Returns:
            New nn.Linear with merged weights (no LoRA overhead at inference)
        """
        # Compute ΔW = (alpha/r) * B @ A
        delta_W = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling

        # Create new linear layer
        merged = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias=self.base_linear.bias is not None,
        )

        # Merge weights: W_new = W_base + ΔW
        merged.weight.data = self.base_linear.weight.data + delta_W

        # Copy bias if exists
        if self.base_linear.bias is not None:
            merged.bias.data = self.base_linear.bias.data.clone()

        return merged

    def trainable_parameters(self):
        """Return only LoRA trainable parameters (for optimizer)."""
        return self.lora.parameters()

    def get_stats(self):
        """Get statistics about parameter counts."""
        base_params = sum(p.numel() for p in self.base_linear.parameters())
        lora_params = sum(p.numel() for p in self.lora.parameters())

        return {
            'base_params': base_params,
            'lora_params': lora_params,
            'total_params': base_params + lora_params,
            'trainable_params': lora_params,
            'param_reduction': f"{(1 - lora_params / base_params) * 100:.2f}%",
            'rank': self.rank,
            'alpha': self.alpha,
        }


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list[str] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> dict:
    """
    Apply LoRA to specific modules in a model.

    Args:
        model: PyTorch model to adapt
        target_modules: List of module names to apply LoRA to
                       (e.g., ['c_attn', 'c_proj'])
                       If None, applies to all Linear layers
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout

    Returns:
        Dictionary with stats about LoRA application
    """
    if target_modules is None:
        # Default: apply to all Linear layers
        target_modules = []

    replaced_count = 0
    total_base_params = 0
    total_lora_params = 0

    # Recursively replace target Linear layers with LinearWithLoRA
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            # Check if this is a target module
            should_replace = False

            if isinstance(child, nn.Linear):
                if not target_modules:
                    # Replace all Linear layers
                    should_replace = True
                else:
                    # Check if module name matches any target
                    for target in target_modules:
                        if target in child_name or child_name == target:
                            should_replace = True
                            break

            if should_replace:
                # Replace with LoRA version
                lora_layer = LinearWithLoRA(
                    base_linear=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )

                # Set the new module
                setattr(module, child_name, lora_layer)

                # Update stats
                stats = lora_layer.get_stats()
                total_base_params += stats['base_params']
                total_lora_params += stats['lora_params']
                replaced_count += 1

                print(f"Applied LoRA to {name}.{child_name}: "
                      f"{stats['base_params']:,} base params + "
                      f"{stats['lora_params']:,} LoRA params "
                      f"({stats['param_reduction']} reduction)")

    return {
        'replaced_layers': replaced_count,
        'total_base_params': total_base_params,
        'total_lora_params': total_lora_params,
        'param_reduction': f"{(1 - total_lora_params / total_base_params) * 100:.2f}%",
    }


def freeze_non_lora_parameters(model: nn.Module):
    """
    Freeze ALL parameters in the model except LoRA parameters.

    This is CRITICAL for LoRA fine-tuning - without this, embeddings,
    LayerNorms, and lm_head will still be trainable, defeating the purpose
    of parameter-efficient fine-tuning.

    Args:
        model: Model with LoRA layers applied

    Returns:
        Dictionary with statistics about frozen/trainable params
    """
    total_params = 0
    frozen_params = 0
    lora_params = 0

    # First, freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False
        total_params += param.numel()
        frozen_params += param.numel()

    # Then, unfreeze ONLY LoRA parameters
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            for param in module.lora.parameters():
                param.requires_grad = True
                lora_params += param.numel()
                frozen_params -= param.numel()

    stats = {
        'total_params': total_params,
        'lora_params': lora_params,
        'frozen_params': frozen_params,
        'trainable_ratio': lora_params / total_params * 100,
    }

    print(f"\n{'='*60}")
    print("FREEZING NON-LORA PARAMETERS")
    print(f"{'='*60}")
    print(f"Total parameters:     {stats['total_params']:>12,}")
    print(f"LoRA parameters:      {stats['lora_params']:>12,} (trainable)")
    print(f"Frozen parameters:    {stats['frozen_params']:>12,}")
    print(f"Trainable ratio:      {stats['trainable_ratio']:>12.2f}%")
    print(f"{'='*60}\n")

    return stats


def get_lora_parameters(model: nn.Module):
    """
    Extract only LoRA parameters from a model (for optimizer).

    Returns:
        List of LoRA parameters (only these will be trained)
    """
    lora_params = []

    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            lora_params.extend(module.trainable_parameters())

    return lora_params


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights back into base model for inference.

    Replaces all LinearWithLoRA modules with regular nn.Linear containing
    merged weights: W_merged = W_base + (alpha/r) * B @ A

    Args:
        model: Model with LoRA layers

    Returns:
        Model with merged weights (no LoRA overhead)
    """
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LinearWithLoRA):
                # Merge and replace
                merged = child.merge_weights()
                setattr(module, child_name, merged)
                print(f"Merged LoRA weights for {name}.{child_name}")

    return model
