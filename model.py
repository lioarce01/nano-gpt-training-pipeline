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

    # PEFT compatibility attributes
    model_type: str = "gpt"  # Required by PEFT


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

        # PEFT compatibility: Add generation_config attribute
        self.generation_config = None

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

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Prepare inputs for generation (required by PEFT for causal LM).

        This method is called by HuggingFace generation utilities and PEFT.
        For our simple GPT, we just return the input_ids as-is.
        """
        return {"idx": input_ids}

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
            lora_only: If True, only optimize LoRA parameters (PEFT mode)
        """
        if lora_only:
            # PEFT LoRA fine-tuning mode: only optimize adapter parameters
            from peft import PeftModel

            if not isinstance(self, PeftModel):
                raise ValueError(
                    "lora_only=True but model is not a PeftModel!\n"
                    "Did you forget to call apply_lora()?"
                )

            print("\nPEFT LoRA mode: only adapter parameters are trainable")

            # PEFT automatically sets requires_grad for adapter params
            trainable_params = [p for p in self.parameters() if p.requires_grad]

            if not trainable_params:
                raise ValueError("No trainable parameters found in PeftModel!")

            print(f"  Found {len(trainable_params)} trainable parameter tensors")

            # No weight decay distinction for LoRA (all params are small adapters)
            optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, betas=betas)

        else:
            # Standard training: optimize all trainable parameters with weight decay
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
        Apply LoRA (Low-Rank Adaptation) via PEFT for parameter-efficient fine-tuning.

        IMPORTANT: This returns a NEW PeftModel that wraps the original model.
        You MUST reassign the returned model: model = model.apply_lora(...)

        Args:
            rank: LoRA rank (r), smaller = fewer params, typical: 4-32
            alpha: Scaling factor for LoRA updates (typically 2 * rank)
            dropout: Dropout probability for LoRA path
            target_modules: List of module names to apply LoRA to
                          (default: ['c_attn', 'c_proj'])

        Returns:
            PeftModel wrapping this model with LoRA adapters
        """
        from peft import get_peft_model
        from peft_utils import create_lora_config

        if target_modules is None:
            target_modules = ['c_attn', 'c_proj']

        print(f"\nApplying LoRA via PEFT:")
        print(f"  Rank: {rank}")
        print(f"  Alpha: {alpha}")
        print(f"  Dropout: {dropout}")
        print(f"  Target modules: {target_modules}")

        # Create LoRA config and apply via PEFT
        lora_config = create_lora_config(rank, alpha, dropout, target_modules)
        peft_model = get_peft_model(self, lora_config)

        # Print trainable parameters
        peft_model.print_trainable_parameters()

        print("\n[SUCCESS] LoRA applied successfully via PEFT!")

        return peft_model

    def merge_lora(self):
        """
        Merge all LoRA weights back into the base model via PEFT.

        After fine-tuning with LoRA, this combines the LoRA adapters (A, B)
        with the frozen base weights (W) to create a single merged model:
            W_merged = W_base + (alpha/r) * B @ A

        The resulting model has the same architecture as the original but
        incorporates the learned adaptations.

        IMPORTANT: This only works if the model is a PeftModel.
        """
        from peft import PeftModel

        if not isinstance(self, PeftModel):
            print("⚠ Warning: Model is not a PeftModel, nothing to merge")
            return

        print("\nMerging LoRA weights via PEFT...")
        merged_model = self.merge_and_unload()

        # Copy merged state into self
        self.load_state_dict(merged_model.state_dict())

        print("[SUCCESS] LoRA weights merged successfully!")
        print("  Model is now a standard (non-PEFT) model")

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

