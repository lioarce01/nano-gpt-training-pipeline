import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = False  # set True to add bias terms to Linear/LayerNorm

    # LoRA configuration (for fine-tuning)
    lora_rank: int = 0  # 0 = disabled, typical: 4-32
    lora_alpha: float = 16.0  # scaling factor (typically 2 * rank)
    lora_dropout: float = 0.0  # dropout for LoRA path
    lora_targets: tuple = ('c_attn', 'c_proj')  # which layers to apply LoRA to


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # register causal mask once to avoid regenerating every forward
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, time, channels
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, n_head, T, head_size)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """A minimal GPT language model inspired by nanoGPT."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError("Cannot forward, sequence length is too long.")

        pos = torch.arange(0, t, device=device, dtype=torch.long)
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas, lora_only: bool = False
    ):
        """
        Configure AdamW optimizer with weight decay.

        Args:
            weight_decay: Weight decay for regularization
            learning_rate: Learning rate
            betas: Adam beta parameters (beta1, beta2)
            lora_only: If True, only optimize LoRA parameters (for fine-tuning)
        """
        if lora_only:
            # LoRA fine-tuning mode: only optimize LoRA parameters
            from lora import LinearWithLoRA

            lora_params = []
            for module in self.modules():
                if isinstance(module, LinearWithLoRA):
                    lora_params.extend(module.trainable_parameters())

            if not lora_params:
                raise ValueError("lora_only=True but no LoRA parameters found! "
                                 "Did you forget to call apply_lora()?")

            print(f"\nLoRA fine-tuning mode: optimizing {len(lora_params)} LoRA parameter tensors")

            # No weight decay distinction for LoRA (all params are small adapters)
            optimizer = torch.optim.AdamW(lora_params, lr=learning_rate, betas=betas)

        else:
            # Standard training: optimize all trainable parameters
            decay, no_decay = [], []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue  # skip frozen parameters

                if param.dim() >= 2:
                    decay.append(param)
                else:
                    no_decay.append(param)

            optim_groups = [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer

    def apply_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: list = None,
    ):
        """
        Apply LoRA (Low-Rank Adaptation) to the model for parameter-efficient fine-tuning.

        This replaces target Linear layers with LinearWithLoRA, which adds trainable
        low-rank matrices (A, B) while freezing the original weights.

        Args:
            rank: LoRA rank (r), smaller = fewer params, typical: 4-32
            alpha: Scaling factor for LoRA updates (typically 2 * rank)
            dropout: Dropout probability for LoRA path
            target_modules: List of module names to apply LoRA to
                          (default: ['c_attn', 'c_proj'] from config)

        Returns:
            Dictionary with statistics about LoRA application
        """
        from lora import apply_lora_to_model

        if target_modules is None:
            target_modules = list(self.config.lora_targets)

        print(f"\nApplying LoRA to model:")
        print(f"  Rank: {rank}")
        print(f"  Alpha: {alpha}")
        print(f"  Dropout: {dropout}")
        print(f"  Target modules: {target_modules}")
        print()

        stats = apply_lora_to_model(
            model=self,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Update config to reflect LoRA is enabled
        self.config.lora_rank = rank
        self.config.lora_alpha = alpha
        self.config.lora_dropout = dropout

        print(f"\nLoRA applied successfully!")
        print(f"  Total LoRA parameters: {stats['total_lora_params']:,}")
        print(f"  Total base parameters: {stats['total_base_params']:,}")
        print(f"  Parameter reduction: {stats['param_reduction']}")
        print()

        return stats

    def merge_lora(self):
        """
        Merge all LoRA weights back into the base model.

        After fine-tuning with LoRA, this combines the LoRA adapters (A, B)
        with the frozen base weights (W) to create a single merged model:
            W_merged = W_base + (alpha/r) * B @ A

        The resulting model has the same architecture as the original but
        incorporates the learned adaptations.
        """
        from lora import merge_lora_weights

        print("\nMerging LoRA weights into base model...")
        merge_lora_weights(self)
        print("LoRA weights merged successfully!")
        print("Model now contains single merged weights (no LoRA overhead).")

        # Reset LoRA config
        self.config.lora_rank = 0

    def get_num_params(self, trainable_only: bool = False):
        """
        Get the number of parameters in the model.

        Args:
            trainable_only: If True, only count trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

