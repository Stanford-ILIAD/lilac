"""
imitation.py

Language-Conditioned Behavioral Cloning Model from Proprioceptive + Object State; leverages precomputed language
embeddings from a BERT-style LM (pooled), and late, FiLM-based fusion to incorporate linguistic information with the
proprioceptive + object state. Supports a masked k-step history via a Small Transformer Encoder.

Note :: At "inference" --> leverages "Unnatural Language Processing" for mapping novel language inputs to "exemplars".
"""
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from annoy import AnnoyIndex
from einops import rearrange
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from lilac.models.core import TransformerBlock, get_1D_position_embeddings


class Imitation(LightningModule):
    def __init__(
        self,
        state_dim: int,
        language_dim: int,
        action_space: str,
        action_dim: int,
        horizon: int,
        max_grad_steps: int,
        run_directory: Path,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim, self.language_dim, self.horizon = state_dim, language_dim, horizon
        self.action_space, self.action_dim, self.max_grad_steps = action_space, action_dim, max_grad_steps
        self.hidden_dim, self.n_heads, self.n_blocks = hidden_dim, n_heads, n_blocks
        self.run_directory = run_directory

        # Load Annoy Index from Run Directory
        self.index = AnnoyIndex(self.language_dim, "angular")
        self.index.load(str(self.run_directory / "index.ann"))
        with open(self.run_directory / "idx2lang.json", "r") as f:
            self.idx2lang = json.load(f)

        # Build Model
        self.build_model()

    def build_model(self) -> None:
        # State Encoder =>
        #   + BatchNorm1D over the state --> ensures numerical stability & good training performance
        #   + 2 Layer MLP for each state_t into Hidden Dimension
        self.state_bn = nn.BatchNorm1d(self.state_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # We apply position encoding after state-encoding, before the Transformer Blocks...
        self.pe = nn.Parameter(torch.zeros(1, self.horizon, self.hidden_dim))
        self.pe.data.copy_(
            torch.from_numpy(get_1D_position_embeddings(self.hidden_dim, self.horizon)).float().unsqueeze(0)
        )

        # Two-Block Transformer for eating State Histories...
        self.blocks = nn.ModuleList([TransformerBlock(self.hidden_dim, self.n_heads) for _ in range(self.n_blocks)])
        self.norm = nn.LayerNorm(self.hidden, eps=1e-6)

        # Language Encoder => FiLM Generator
        #   + 2 x 2-Layer MLP with a shared "trunk" (initial layer) for producing FiLM Gamma and Beta
        self.film_gen = nn.Sequential(nn.Linear(self.language_dim, self.hidden_dim), nn.GELU())
        self.gamma, self.beta = nn.Linear(self.hidden_dim, self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim)

        # Action MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, lang: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode state --> Transformer over history --> late FiLM fusion --> action."""
        state_emb = self.state_bn(rearrange(state, "bsz seq d -> (bsz seq) d"))
        state_enc = self.state_encoder(state_emb)
        states = rearrange(state_enc, "(bsz seq) d -> bsz seq d", seq=self.horizon) + self.pe

        # Apply Transformer Blocks
        for block in self.blocks:
            states = block(states)
        states = self.norm(states)

        # Run masked average-pooling to come up with final dense representation...
        h_state = (states * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        # Language => FiLM Generator
        film = self.film_gen(lang)
        gamma, beta = self.gamma(film), self.beta(film)

        # FiLM the State Representation!
        fused = (gamma * h_state) + beta

        # Return Action
        return self.mlp(fused)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW([p for p in self.parameters() if p.requires_grad])
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(self.max_steps * 0.05), self.max_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], _: int
    ) -> torch.Tensor:
        """Unpack batch, perform a forward pass, and compute MSE loss with respect to actual actions."""
        states, lang, mask, actions = batch

        # Forward...
        predicted_actions = self.forward(states, lang, mask)

        # Compute MSE
        loss = F.mse_loss(predicted_actions, actions)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> None:
        states, lang, mask, actions = batch

        # Forward...
        predicted_actions = self.forward(states, lang, mask)

        # Compute MSE
        loss = F.mse_loss(predicted_actions, actions)
        self.log("val_loss", loss, prog_bar=True)
