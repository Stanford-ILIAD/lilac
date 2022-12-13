"""
inference.py

Defines wrappers for online evaluation of LILAC/LILA/Imitation models; handles history tracking for Imitation, and
language parsing/retrieval via unnatural language processing.
"""
import json
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from lilac.constants import CONTROL_HZ, N_OBJECTS
from lilac.preprocessing.utils import get_object_state, sentence_pool


# === Imitation Wrapper ===
class ImitationInferenceWrapper:
    def __init__(
        self, model: nn.Module, tokenizer: PreTrainedTokenizer, lm: PreTrainedModel, horizon: int, window: float
    ) -> None:
        self.model, self.tokenizer, self.lm = model, tokenizer, lm
        self.horizon, self.window = horizon, window

        # Load Object State
        self.obj_state = get_object_state("data/evaluation-state.json")

        # Robot State Trackers
        self.q, self.ee_pose, self.gripper_open, self.state, self.mask = None, None, None, None, None

        # Compute Interval & Stride for Horizon
        self.interval = int(self.window * CONTROL_HZ)

        # Returns [-9, -8, ... -1] for `horizon = 10, window = 1.0s, CONTROL_HZ = 10` for example...
        self.stride_idxs = list(range(-self.interval // horizon, -self.interval, -self.interval // horizon))[::-1]

        # Create history buffers...
        self.state_history = torch.zeros((self.interval, 7 + 7 + 1 + (3 * N_OBJECTS)))
        self.mask_history = torch.zeros(self.interval)

        # Set Preprocessing Trackers...
        self.embedding, self.exemplar = None, None

    def embed(self, utterance: str) -> None:
        print(f"\t=>> Encoding Input Utterance :: `{utterance.lower()}`")
        with torch.no_grad():
            enc = self.tokenizer(utterance, padding=True, truncation=True, max_length=32, return_tensors="pt")
            embed = self.lm(**enc)
            self.embedding = sentence_pool(embed, enc["attention_mask"])

            # Run Unnatural Language Processing => Nearest Neighbor Retrieval...
            (idx,) = self.model.index.get_nns_by_vector(self.embedding, n=1)
            self.embedding = torch.Tensor(self.model.index.get_item_vector(idx))
            self.exemplar = self.model.idx2lang[str(idx)]

    def encode_state(self, q: np.ndarray, ee_pose: np.ndarray, gripper_open: bool) -> None:
        self.q, self.ee_pose, self.gripper_open = q, ee_pose, gripper_open

        # Compute  `self.state` as a function of existing history + current
        #   > Compute current state using `q` and `ee_pose` and `gripper_open`
        #   > Grab `-stride_idxs` states + concatenate `current` as "full horizon"
        #   > Store in `self.state`
        current = torch.from_numpy(
            np.concatenate([self.q, self.ee_pose, [1.0] if gripper_open else [-1.0], self.obj_state])
        )
        self.state = torch.cat([self.state_history[self.stride_idxs], current[None, ...]], dim=0)
        self.mask = torch.cat([self.mask_history[self.stride_idxs], torch.ones(1)], dim=0)

        # Now, after updating the global `self.state` --> add current to history...
        self.state_history = torch.roll(self.state_history, -1, dims=0)
        self.mask_history = torch.roll(self.mask_history, -1, dims=0)

        # Add current state to "end" of the buffer...
        self.state_history[-1], self.mask_history[-1] = current, 1

    def act(self) -> Tuple[np.ndarray, Optional[bool]]:
        state, lang, mask = self.state[None, ...], self.embedding[None, ...], self.mask[None, ...]
        with torch.no_grad():
            action = self.model(state, lang, mask).squeeze().numpy()

        # Parse out action vs. gripper...
        return action[:-1], action[-1] > 0


# === LILA Wrapper ===
class LILAInferenceWrapper:
    def __init__(
        self, model: nn.Module, tokenizer: PreTrainedTokenizer, lm: PreTrainedModel, action_scale: float = 0.01
    ) -> None:
        self.model, self.tokenizer, self.lm = model, tokenizer, lm
        self.action_scale = action_scale

        # Load Object State
        self.obj_state = get_object_state("data/evaluation-state.json")

        # Set Robot State Trackers....
        self.q, self.ee_pose, self.state = None, None, None

        # Set Preprocessing Trackers...
        self.embedding, self.exemplar = None, None

    def embed(self, utterance: str) -> None:
        print(f"\t=>> Encoding Input Utterance :: `{utterance.lower()}`")
        with torch.no_grad():
            enc = self.tokenizer(utterance, padding=True, truncation=True, max_length=32, return_tensors="pt")
            embed = self.lm(**enc)
            self.embedding = sentence_pool(embed, enc["attention_mask"])

            # Run Unnatural Language Processing => Nearest Neighbor Retrieval...
            (idx,) = self.model.index.get_nns_by_vector(self.embedding, n=1)
            self.embedding = torch.Tensor(self.model.index.get_item_vector(idx))
            self.exemplar = self.model.idx2lang[str(idx)]

    def encode_state(self, q: np.ndarray, ee_pose: np.ndarray, _: bool) -> None:
        self.q, self.ee_pose = q, ee_pose
        self.state = torch.from_numpy(np.concatenate([self.q, self.ee_pose, self.obj_state]))

    def decode_action(self, z: np.ndarray) -> np.ndarray:
        state, lang, z = self.state[None, ...], self.embedding[None, ...], torch.from_numpy(z).float()[None, ...]
        with torch.no_grad():
            action = self.model.decoder(state, lang, z).squeeze().detach().numpy()

        return action * self.action_scale


# === LILAC Wrapper ===
class LILACInferenceWrapper:
    def __init__(
        self, model: nn.Module, tokenizer: PreTrainedTokenizer, lm: PreTrainedModel, action_scale: float = 0.01
    ) -> None:
        self.model, self.tokenizer, self.lm = model, tokenizer, lm
        self.action_scale = action_scale

        # Load Object State
        self.obj_state = get_object_state("data/evaluation-state.json")

        # Set Robot State Trackers....
        self.q, self.ee_pose, self.state = None, None, None

        # Set Preprocessing Trackers...
        self.embedding, self.exemplar, self.alpha = None, None, None

        # Load GPT-3 Alphas
        with open("data/gpt3-alphas.json", "r") as f:
            self.lang2alpha = json.load(f)

    def embed(self, utterance: str) -> None:
        print(f"\t=>> Encoding Input Utterance :: `{utterance.lower()}`")
        with torch.no_grad():
            enc = self.tokenizer(utterance, padding=True, truncation=True, max_length=32, return_tensors="pt")
            embed = self.lm(**enc)
            self.embedding = sentence_pool(embed, enc["attention_mask"])

            # Run Unnatural Language Processing => Nearest Neighbor Retrieval...
            (idx,) = self.model.index.get_nns_by_vector(self.embedding, n=1)
            self.embedding = torch.Tensor(self.model.index.get_item_vector(idx))
            self.exemplar = self.model.idx2lang[str(idx)]

            # Retrieve `alpha` as well...
            self.alpha = float(self.lang2alpha.get(self.exemplar, 1.0))

    def encode_state(self, q: np.ndarray, ee_pose: np.ndarray, _: bool) -> None:
        self.q, self.ee_pose = q, ee_pose
        self.state = torch.from_numpy(np.concatenate([self.q, self.ee_pose, self.obj_state]))

    def decode_action(self, z: np.ndarray) -> np.ndarray:
        state, lang, z = self.state[None, ...], self.embedding[None, ...], torch.from_numpy(z).float()[None, ...]
        alpha = torch.from_numpy(np.array(self.alpha)).float()[None, ...]
        with torch.no_grad():
            action = self.model.decoder(state, lang, alpha, z).squeeze().detach().numpy()

        return action * self.action_scale
