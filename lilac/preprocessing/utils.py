"""
utils.py

Utilities for preprocessing data -- includes normalizing/featurizing object positions, additional functions for
constructing language embeddings and NN-stores (via Annoy).
"""
import json
from pathlib import Path

import numpy as np
import torch

from lilac.constants import GRIPPER_TOLERANCE, N_OBJECTS


# === Sentence-Transformers/RoBERTa Language Model Utilities ===
def sentence_pool(output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average pooling for obtaining "sentence" embeddings from a BERT-style language model."""
    embeddings = output[0]
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    embedding_sum = torch.sum(embeddings * mask, dim=1)
    mask_sum = torch.clamp(mask.sum(1), min=1e-9)
    return (embedding_sum / mask_sum).squeeze()


# === Gripper Processing ===
def compute_binary_grasps(gripper_widths: np.ndarray) -> np.ndarray:
    """Compute binary gripper actions, given a serious of continuous gripper widths."""
    future_widths, past_widths = np.array(gripper_widths[1:]), np.array(gripper_widths[:-1])
    diffs = future_widths - past_widths
    diffs[np.abs(diffs) < GRIPPER_TOLERANCE] = 0

    # Note: 1 = "Open" & -1 = "Close"
    binary_grasps, curr_status = [1], 1
    for diff in diffs:
        if not math.isclose(0, diff):
            curr_status = 1 if diff > 0 else -1
        binary_grasps.append(curr_status)
    return np.asarray(binary_grasps)


# === Object State Processing ===
def get_object_state(delta_json: Path) -> np.ndarray:
    """Convert set of "delta" coordinates per-object into a flattened np.ndarray."""
    with open(delta_json, "r") as f:
        xyz_coords = json.load(f)

    # Assertion...
    assert len(xyz_coords) == N_OBJECTS, "All objects must be represented in the state-delta file!"

    # Create object state vector...
    obj_state = []
    for obj in sorted(xyz_coords.keys()):
        obj_state += map(float, xyz_coords[obj])
    return np.asarray(obj_state, dtype=np.float32)
