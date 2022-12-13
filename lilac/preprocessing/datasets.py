"""
datasets.py

Given a set of collected demonstrations and paired language instructions (following `scripts/demonstrate.py`), create a
traditional PyTorch Dataset for enabling easy training & validation.

Note :: though 2 action spaces are acceptable (`ee-euler-delta` and `joint-delta`), the default is `ee-euler-delta`!
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from annoy import AnnoyIndex
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lilac.constants import ACTION_DIMENSIONALITY, CONTROL_HZ, N_OBJECTS
from lilac.preprocessing.utils import compute_binary_grasps, get_object_state, sentence_pool
from lilac.robot.utils import quat2euler, subtract_quats


class ImitationDataset(Dataset):
    def __init__(
        self,
        demonstrations: List[Path],
        utterances: List[str],
        action_space: str = "ee-euler-delta",
        horizon: int = 10,
        window: float = 1.0,
        is_train: bool = True,
    ):
        """Create an Imitation Dataset with states, language, and actions, with the specified window of past history."""
        self.demonstrations, self.utterances = demonstrations, utterances
        self.action_space, self.action_dim, self.is_train = action_space, ACTION_DIMENSIONALITY[action_space], is_train
        self.horizon, self.window = horizon, window
        self.obj_states = [Path(str(x)[:-4] + ".json") for x in self.demonstrations]

        # State Dimensionality --> (x-y-z object positions * N_OBJECTS + joint poses + ee pose/quat)
        self.proprioceptive_dim, self.language_dim = 7 + (3 + 4), 768
        self.state_dim = (N_OBJECTS * 3) + self.proprioceptive_dim

        # Dataset Internals...
        self.states, self.language, self.actions, self.demo_idxs = [], [], [], []

        # History Management
        if self.horizon > 0:
            # `stride_idxs` is the set of "past" steps we need to retrieve to fulfill the horizon (number of states)
            #  and window (duration of the "history" we want to look at). Depends on `HZ`.
            #       =>> Example: `horizon = 5, window = 1.0` -- we want `horizon` states out of the last 1.0 * HZ = 20!
            #                     |=> stride_idxs (we'll multiply by -1 later) = [16, 12, 8, 4] & current state
            self.interval = int(self.window * CONTROL_HZ)
            self.stride = self.interval // self.horizon
            self.stride_idxs = list(range(self.stride, self.interval, self.stride))[::-1]

        # Initialize language -> embedding mapping, instantiate Sentence-Transformers RoBERTa Model
        os.makedirs("cache/paraphrase", exist_ok=True)
        self.lang2embed = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
        )
        self.lm = AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
        )

        # Preprocess Language...
        self.preprocess_language()

        # Build Examples...
        self.build_examples()

    def preprocess_language(self) -> None:
        print("[*] Preprocessing Language Instructions...")
        with torch.no_grad():
            for utterance in self.utterances:
                enc = self.tokenizer(utterance, padding=True, truncation=True, max_length=32, return_tensors="pt")
                embed = self.lm(**enc)
                self.lang2embed[utterance] = sentence_pool(embed, enc["attention_mask"])

    def build_examples(self) -> None:
        print("[*] Building Dataset of (state, language, action) tuples for task...")
        desc = "    =>> Processing demonstrations... "
        for demo_idx, demo_file in tqdm(enumerate(self.demonstrations), desc=desc, total=len(self.demonstrations)):
            demo, obj_state = np.load(demo_file), get_object_state(self.obj_states[demo_idx])
            qs, ee_poses, gripper_widths = demo["q"], demo["ee_pose"], demo["gripper_width"]

            # Discretize Gripper Widths into {-1, 1} Grasps --> {"closed", "open"}
            grasps = compute_binary_grasps(gripper_widths)

            # We have a nice "built-in" augmentation set --> the language instructions (we'll only sample 5 for now)
            utterance_set = np.random.choice(list(self.lang2embed.keys()), 5, replace=False).tolist()
            for utterance_idx, utterance in utterance_set:
                # Iterate through trajectory, assemble example tuples...
                for i in range(len(qs) - 1):
                    pre_q, post_q, pre_ee, post_ee = qs[i], qs[i + 1], ee_poses[i], ee_poses[i + 1]

                    # Compute action in *specified* action space!
                    if self.action_space == "joint-delta":
                        action = post_q - pre_q
                    elif self.action_space == "ee-euler-delta":
                        # Position delta is calculated normally, orientation needs to be subtracted in quat space...
                        pos_action = post_ee[:3] - pre_ee[:3]
                        euler_action = quat2euler(subtract_quats(post_ee[3:], pre_ee[3:]))
                        action = np.concatenate([pos_action, euler_action])
                    else:
                        raise ValueError(f"Action space `{self.action_space}` is not supported!")

                    # Add to trackers...
                    self.states.append(np.concatenate([pre_q, pre_ee, grasps[i][None, ...], obj_state]))
                    self.language.append(self.lang2embed[utterance])
                    self.actions.append(np.concatenate([action, grasps[i + 1][None, ...]]))
                    self.demo_idxs.append((demo_idx * 1000) + utterance_idx)

        # Tensorize...
        self.states, self.actions = np.asarray(self.states), np.asarray(self.actions)
        self.states = torch.from_numpy(self.states).float()
        self.language = torch.stack(self.language).float()
        self.actions = torch.from_numpy(self.actions).float()

        # Sanity Check...
        assert len(self.states) == len(self.language) == len(self.actions), "We don't have an equal number of samples!"

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        # No history --> just regular BC!
        if self.horizon == 0:
            return self.states[idx], self.language[idx], self.actions[idx]

        # Otherwise --> prepend history...
        else:
            demo_start_idx = self.demo_idxs.index(self.demo_idxs[idx])
            idxs = torch.Tensor([max(demo_start_idx, idx - s) for s in self.stride_idxs] + [idx]).long()
            mask = torch.Tensor([1 if (idxs[i + 1] - idxs[i] == self.stride) else 0 for i in range(len(idxs) - 1)] + [1])
            return self.states[idxs], self.language[idx], mask, self.actions[idx]

    def __len__(self) -> int:
        return len(self.states)


class LILADataset(Dataset):
    def __init__(
        self,
        demonstrations: List[Path],
        utterances: List[str],
        action_space: str = "ee-euler-delta",
        is_train: bool = True,
    ) -> None:
        """Create a LILA Dataset with states, actions, and language."""
        self.demonstrations, self.utterances = demonstrations, utterances
        self.action_space, self.action_dim, self.is_train = action_space, ACTION_DIMENSIONALITY[action_space], is_train
        self.obj_states = [Path(str(x)[:-4] + ".json") for x in self.demonstrations]

        # State Dimensionality --> (x-y-z object positions * N_OBJECTS + joint poses + ee pose/quat)
        self.proprioceptive_dim, self.language_dim = 7 + (3 + 4), 768
        self.state_dim = (N_OBJECTS * 3) + self.proprioceptive_dim

        # Dataset Internals...
        self.states, self.language, self.actions = [], [], []

        # Initialize language -> embedding mapping, instantiate Sentence-Transformers RoBERTa Model
        os.makedirs("cache/paraphrase", exist_ok=True)
        self.lang2embed = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
        )
        self.lm = AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
        )

        # Preprocess Language...
        self.preprocess_language()

        # Build Examples...
        self.build_examples()

    def preprocess_language(self) -> None:
        print("[*] Preprocessing Language Instructions...")
        with torch.no_grad():
            for utterance in self.utterances:
                enc = self.tokenizer(utterance, padding=True, truncation=True, max_length=32, return_tensors="pt")
                embed = self.lm(**enc)
                self.lang2embed[utterance] = sentence_pool(embed, enc["attention_mask"])

    def build_examples(self) -> None:
        print("[*] Building Dataset of (state, language, action) tuples for task...")
        desc = "    =>> Processing demonstrations... "
        for demo_idx, demo_file in tqdm(enumerate(self.demonstrations), desc=desc, total=len(self.demonstrations)):
            demo, obj_state = np.load(demo_file), get_object_state(self.obj_states[demo_idx])
            qs, ee_poses = demo["q"], demo["ee_pose"]

            # We have a nice "built-in" augmentation set --> the language instructions (we'll only sample 5 for now)
            utterance_set = np.random.choice(list(self.lang2embed.keys()), 5, replace=False).tolist()
            for utterance_idx, utterance in utterance_set:
                # Iterate through trajectory, assemble example tuples...
                for i in range(len(qs) - 1):
                    pre_q, post_q, pre_ee, post_ee = qs[i], qs[i + 1], ee_poses[i], ee_poses[i + 1]

                    # Compute action in *specified* action space!
                    if self.action_space == "joint-delta":
                        action = post_q - pre_q
                    elif self.action_space == "ee-euler-delta":
                        # Position delta is calculated normally, orientation needs to be subtracted in quat space...
                        pos_action = post_ee[:3] - pre_ee[:3]
                        euler_action = quat2euler(subtract_quats(post_ee[3:], pre_ee[3:]))
                        action = np.concatenate([pos_action, euler_action])
                    else:
                        raise ValueError(f"Action space `{self.action_space}` is not supported!")

                    # Add to trackers...
                    self.states.append(np.concatenate([pre_q, pre_ee, obj_state]))
                    self.language.append(self.lang2embed[utterance])
                    self.actions.append(action)

        # Tensorize...
        self.states, self.actions = np.asarray(self.states), np.asarray(self.actions)
        self.states = torch.from_numpy(self.states).float()
        self.language = torch.stack(self.language).float()
        self.actions = torch.from_numpy(self.actions).float()

        # Unit Normalize Actions for LILA/LILAC (because of orthonormalization in model)
        self.actions = F.normalize(self.actions, dim=1)

        # Sanity Check...
        assert len(self.states) == len(self.language) == len(self.actions), "We don't have an equal number of samples!"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.language[idx], self.actions[idx]

    def __len__(self) -> int:
        return len(self.states)


class LILACDataset(Dataset):
    def __init__(
        self,
        demonstrations: List[Path],
        utterances: List[str],
        lang2alpha: Dict[str, float],
        action_space: str = "ee-euler-delta",
        is_train: bool = True,
    ) -> None:
        """Create a LILAC Dataset with states, actions, language, and alphas."""
        self.demonstrations, self.utterances, self.lang2alpha = demonstrations, utterances, lang2alpha
        self.action_space, self.action_dim, self.is_train = action_space, ACTION_DIMENSIONALITY[action_space], is_train
        self.obj_states = [Path(str(x)[:-4] + ".json") for x in self.demonstrations]

        # State Dimensionality --> (x-y-z object positions * N_OBJECTS + joint poses + ee pose/quat)
        self.proprioceptive_dim, self.language_dim = 7 + (3 + 4), 768
        self.state_dim = (N_OBJECTS * 3) + self.proprioceptive_dim

        # Dataset Internals...
        self.states, self.language, self.alphas, self.actions = [], [], [], []

        # Initialize language -> embedding mapping, instantiate Sentence-Transformers RoBERTa Model
        os.makedirs("cache/paraphrase", exist_ok=True)
        self.lang2embed = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
        )
        self.lm = AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
        )

        # Preprocess Language...
        self.preprocess_language()

        # Build Examples...
        self.build_examples()

    def preprocess_language(self) -> None:
        print("[*] Preprocessing Language Instructions...")
        with torch.no_grad():
            for utterance in self.utterances:
                enc = self.tokenizer(utterance, padding=True, truncation=True, max_length=32, return_tensors="pt")
                embed = self.lm(**enc)
                self.lang2embed[utterance] = sentence_pool(embed, enc["attention_mask"])

    def build_examples(self) -> None:
        print("[*] Building Dataset of (state, language, alpha, action) tuples for task...")
        desc = "    =>> Processing demonstrations... "
        for demo_idx, demo_file in tqdm(enumerate(self.demonstrations), desc=desc, total=len(self.demonstrations)):
            demo, obj_state = np.load(demo_file), get_object_state(self.obj_states[demo_idx])
            qs, ee_poses = demo["q"], demo["ee_pose"]

            # We have a nice "built-in" augmentation set --> the language instructions (we'll only sample 5 for now)
            utterance_set = np.random.choice(list(self.lang2embed.keys()), 5, replace=False).tolist()
            for utterance_idx, utterance in utterance_set:
                # Iterate through trajectory, assemble example tuples...
                for i in range(len(qs) - 1):
                    pre_q, post_q, pre_ee, post_ee = qs[i], qs[i + 1], ee_poses[i], ee_poses[i + 1]

                    # Compute action in *specified* action space!
                    if self.action_space == "joint-delta":
                        action = post_q - pre_q
                    elif self.action_space == "ee-euler-delta":
                        # Position delta is calculated normally, orientation needs to be subtracted in quat space...
                        pos_action = post_ee[:3] - pre_ee[:3]
                        euler_action = quat2euler(subtract_quats(post_ee[3:], pre_ee[3:]))
                        action = np.concatenate([pos_action, euler_action])
                    else:
                        raise ValueError(f"Action space `{self.action_space}` is not supported!")

                    # Add to trackers...
                    self.states.append(np.concatenate([pre_q, pre_ee, obj_state]))
                    self.language.append(self.lang2embed[utterance])
                    self.alphas.append(self.lang2alpha[utterance])
                    self.actions.append(action)

        # Tensorize...
        self.states, self.actions = np.asarray(self.states), np.asarray(self.actions)
        self.states = torch.from_numpy(self.states).float()
        self.language = torch.stack(self.language).float()
        self.alphas = torch.from_numpy(np.array(self.alphas)).float()
        self.actions = torch.from_numpy(self.actions).float()

        # Unit Normalize Actions for LILA/LILAC (because of orthonormalization in model)
        self.actions = F.normalize(self.actions, dim=1)

        # Sanity Check...
        assert (
            len(self.states) == len(self.language) == len(self.alphas) == len(self.actions)
        ), "We don't have an equal number of samples!"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.language[idx], self.alphas[idx], self.actions[idx]

    def __len__(self) -> int:
        return len(self.states)


def get_imitation_dataset(
    run_directory: Path,
    tasks: List[str],
    action_space: str = "ee-euler-delta",
    data_directory=Path("data"),
    n_train: int = 45,
    n_val: int = 5,
    horizon: int = 10,
    window: float = 1.0,
) -> Tuple[Dataset, Dataset]:
    """Create a history-aware Imitation Learning dataset fusing state/actions and language."""
    train_datasets, val_datasets = [], []

    # Iterate through tasks, and assemble task-specific datasets
    for task in tasks:
        # Fetch Language Utterances for the given task...
        with open(data_directory / "language" / f"{task}.json", "r") as f:
            utterances = json.load(f)["utterances"]

        # Assemble Demonstration `.npz` files...
        demos = [fn for fn in (data_directory / task / "playback-final").iterdir() if ".npz" in str(fn)]

        # Create Train & Validation Datasets...
        train_datasets.append(
            ImitationDataset(demos[:n_train], utterances, action_space, horizon=horizon, window=window)
        )
        val_datasets.append(
            ImitationDataset(demos[-n_val:], utterances, action_space, horizon=horizon, window=window, is_train=False)
        )

    # Create Nearest-Neighbor Store (Annoy) over `train_dataset.lang2embed` for "Unnatural Language Processing"
    lang2idx, index = {}, AnnoyIndex(train_datasets[0].language_dim, "angular")
    for dataset in train_datasets:
        for utterance in dataset.lang2embed:
            idx = len(lang2idx)
            lang2idx[utterance] = idx
            index.add_item(idx, dataset.lang2embed[utterance])

        # Save Index to Run Directory...
        print(f"[*] Building ANN Index w/ 10 Trees & Saving to `{run_directory}`...")
        index.build(n_trees=10)
        index.save(str(run_directory / "index.ann"))
        with open(run_directory / "idx2lang.json", "w") as f:
            json.dump({idx: lang for lang, idx in lang2idx.items()}, f, indent=4)

    # Concatenate & Return Datasets...
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


def get_lila_dataset(
    run_directory: Path,
    tasks: List[str],
    action_space: str = "ee-euler-delta",
    data_directory=Path("data"),
    n_train: int = 45,
    n_val: int = 5,
) -> Tuple[Dataset, Dataset]:
    """Create a LILA Dataset fusion state/actions and language."""
    train_datasets, val_datasets = [], []

    # Iterate through tasks, and assemble task-specific datasets
    for task in tasks:
        # Fetch Language Utterances for the given task...
        with open(data_directory / "language" / f"{task}.json", "r") as f:
            utterances = json.load(f)["utterances"]

        # Assemble Demonstration `.npz` files...
        demos = [fn for fn in (data_directory / task / "playback-final").iterdir() if ".npz" in str(fn)]

        # Create Train & Validation Datasets...
        train_datasets.append(LILADataset(demos[:n_train], utterances, action_space))
        val_datasets.append(LILADataset(demos[-n_val:], utterances, action_space, is_train=False))

    # Create Nearest-Neighbor Store (Annoy) over `train_dataset.lang2embed` for "Unnatural Language Processing"
    lang2idx, index = {}, AnnoyIndex(train_datasets[0].language_dim, "angular")
    for dataset in train_datasets:
        for utterance in dataset.lang2embed:
            idx = len(lang2idx)
            lang2idx[utterance] = idx
            index.add_item(idx, dataset.lang2embed[utterance])

        # Save Index to Run Directory...
        print(f"[*] Building ANN Index w/ 10 Trees & Saving to `{run_directory}`...")
        index.build(n_trees=10)
        index.save(str(run_directory / "index.ann"))
        with open(run_directory / "idx2lang.json", "w") as f:
            json.dump({idx: lang for lang, idx in lang2idx.items()}, f, indent=4)

    # Concatenate & Return Datasets...
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


def get_lilac_dataset(
    run_directory: Path,
    tasks: List[str],
    action_space: str = "ee-euler-delta",
    data_directory: Path = Path("data"),
    n_train: int = 45,
    n_val: int = 5,
) -> Tuple[Dataset, Dataset]:
    """Create a LILAC Dataset fusing state/actions, language, and GPT-3 alphas."""
    train_datasets, val_datasets = [], []

    # Load GPT-3 Alphas from `data_directory`
    with open(data_directory / "gpt3-alphas.json", "r") as f:
        lang2alpha = json.load(f)

    # Iterate through tasks, and assemble task-specific datasets
    for task in tasks:
        # Fetch Language Utterances for the given task...
        with open(data_directory / "language" / f"{task}.json", "r") as f:
            utterances = json.load(f)["utterances"]

        # Assemble Demonstration `.npz` files...
        demos = [fn for fn in (data_directory / task / "playback-final").iterdir() if ".npz" in str(fn)]

        # Create Train & Validation Datasets...
        train_datasets.append(LILACDataset(demos[:n_train], utterances, lang2alpha, action_space))
        val_datasets.append(LILACDataset(demos[-n_val:], utterances, lang2alpha, action_space, is_train=False))

    # Create Nearest-Neighbor Store (Annoy) over `train_dataset.lang2embed` for "Unnatural Language Processing"
    lang2idx, index = {}, AnnoyIndex(train_datasets[0].language_dim, "angular")
    for dataset in train_datasets:
        for utterance in dataset.lang2embed:
            idx = len(lang2idx)
            lang2idx[utterance] = idx
            index.add_item(idx, dataset.lang2embed[utterance])

        # Save Index to Run Directory...
        print(f"[*] Building ANN Index w/ 10 Trees & Saving to `{run_directory}`...")
        index.build(n_trees=10)
        index.save(str(run_directory / "index.ann"))
        with open(run_directory / "idx2lang.json", "w") as f:
            json.dump({idx: lang for lang, idx in lang2idx.items()}, f, indent=4)

    # Concatenate & Return Datasets...
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)
