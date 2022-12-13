"""
lila.py

LILA from Proprioceptive + Object State; leverages precomputed language embeddings from a BERT-style LM (pooled),
and late, FiLM-based fusion to incorporate linguistic information with the proprioceptive + object state.

Note :: At "inference" -->  leverages "Unnatural Language Processing" for mapping novel language inputs to "exemplars".
"""
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from annoy import AnnoyIndex
from pytorch_lightning import LightningModule
from torch.optim import AdamW, Optimizer


class LILA(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        language_dim: int,
        action_space: str,
        action_dim: int,
        run_directory: Path,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim, self.state_dim, self.language_dim = latent_dim, state_dim, language_dim
        self.action_space, self.action_dim, self.hidden_dim = action_space, action_dim, hidden_dim
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
        #   + 2 Layer MLP into Hidden Dimension
        self.state_bn = nn.BatchNorm1d(self.state_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Language Encoder => FiLM Generator
        #   + 2 x 2-Layer MLP with a shared "trunk" (initial layer) for producing FiLM Gamma and Beta
        self.film_gen = nn.Sequential(nn.Linear(self.language_dim, self.hidden_dim), nn.GELU())
        self.gamma, self.beta = nn.Linear(self.hidden_dim, self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim)

        # (State/Language) + Action Encoder :: Takes (h_fused, action) --> Encodes to `latent_dim`
        #   > Terminates in a `Tanh()` to squash latent actions between [-1, 1]
        self.fused_enc = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh(),
        )

        # Projection into Basis Vectors :: (Takes h_fused) --> encodes to [`latent_dim` * `action_dim`] (gets unrolled)
        self.basis_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim * self.action_dim),
        )

    @staticmethod
    def orthonormalize(x: torch.Tensor) -> torch.Tensor:
        """Given a tensor [bsz, k, dim] return orthonormalized [bsz, k, dim] via Modified Gram-Schmidt."""
        ortho = torch.zeros_like(x)
        for i in range(x.shape[1]):
            # Iterate over "columns" of x (k-dim) --> Clone to avoid autograd issues
            ortho[:, i] = F.normalize(x[:, i], dim=1)
            for j in range(1, x.shape[1]):
                # Modified Gram-Schmidt
                orthogonalized = torch.einsum("bi, bi -> b", ortho[:, 1], x[:, j].clone()).unsqueeze(-1) * ortho[:, 1]
                ortho[:, j] = x[:, j] - orthogonalized
        return x

    def decoder(self, state: torch.Tensor, lang: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decoder-only pipeline, given inputs and latent action `z` => generate high-DoF robot action."""
        state_emb = self.state_bn(state)

        # Feed through State Encoder (normalize because of orthonormalization)
        h_state = self.state_encoder(state_emb)

        # Language => FiLM Generator
        film = self.film_gen(lang)
        gamma, beta = self.gamma(film), self.beta(film)

        # FiLM the Gated Embedding!
        h_fused = (gamma * h_state) + beta

        # Basis Projection & Orthonormalization
        bases = self.basis_projection(h_fused).reshape(-1, self.latent_dim, self.action_dim)
        bases = self.orthonormalize(bases)

        # Expand and compute final action --> sum_{dim=1}([bsz, latent_dim, 1] * [bsz, latent_dim, action_dim])
        return torch.sum(bases * z[..., None], dim=1)

    def forward(self, state: torch.Tensor, lang: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Encode state, the fuse with language prior to feeding to the full latent actions encode/decode pipeline."""
        state_emb = self.state_bn(state)

        # Feed through State Encoder...
        h_state = self.state_encoder(state_emb)

        # Language => FiLM Generator
        film = self.film_gen(lang)
        gamma, beta = self.gamma(film), self.beta(film)

        # FiLM the Gated Embedding!
        h_fused = (gamma * h_state) + beta

        # Basis Projection & Orthonormalization
        bases = self.basis_projection(h_fused).reshape(-1, self.latent_dim, self.action_dim)
        bases = self.orthonormalize(bases)

        # Compute z :: encode + Tanh (range in [-1, 1])
        z = self.fused_enc(torch.cat([h_fused, action], dim=-1))

        # Expand and compute final action --> sum_{dim=1}([bsz, latent_dim, 1] * [bsz, latent_dim, action_dim])
        return torch.sum(bases * z[..., None], dim=1)

    def configure_optimizers(self) -> Optimizer:
        return AdamW([p for p in self.parameters() if p.requires_grad])

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        """Unpack batch, perform a forward pass through the LILAC encoder & decoder, and compute MSE loss."""
        state, language, action = batch

        # Get Predicted/Reconstructed Action
        predicted_action = self.forward(state, language, action)

        # Compute MSE
        loss = F.mse_loss(predicted_action, action)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> None:
        state, language, action = batch

        # Get Predicted/Reconstructed Action
        predicted_action = self.forward(state, language, action)

        # Compute MSE
        loss = F.mse_loss(predicted_action, action)
        self.log("val_loss", loss, prog_bar=True)
