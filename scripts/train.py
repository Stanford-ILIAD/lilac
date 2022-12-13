"""
train.py

Core training script for LILAC and baselines (LILA, Language-Conditioned Imitation Learning) -- loads and preprocesses
demonstration data, instantiates Lightning Modules, and runs training. All models are assumed to be multi-task,
language-conditioned by default.

Both LILAC and LILA take a single state input, while the imitation learning policy takes in a state history
(10 uniformly spaced frames across a 1 second window).

Run with: `python scripts/train.py --model < lilac | lila | imitation > --tasks "water-plant" "right" ...`
"""
import json
import os
from pathlib import Path
from typing import List

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from tap import Tap
from torch.utils.data import DataLoader

from lilac.metrics import MetricsLogger
from lilac.models import LILA, LILAC, Imitation
from lilac.preprocessing import get_imitation_dataset, get_lila_dataset, get_lilac_dataset


# HuggingFace Book-Keeping
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ArgumentParser(Tap):
    # fmt: off
    model: str                                      # Model to train in < lilac | lila | imitation >

    # Dataset Parameters --> this should be a list of `task` IDs (can be high-level or corrections)
    #   that were fed to `scripts/demonstrate.py`; these will match the sub-directory names in `data/`
    data_directory: Path = Path("data")             # Root path to task-specific demonstrations
    tasks: List[str]                                # Example: ["water-plant", "right", "move-to-book"]

    # LILAC / LILA Parameters
    latent_dim: int = 2                             # Dimensionality of the latent actions (e.g., 2-DoF for a joystick)

    # State History Parameters
    horizon: int = 10                               # (Only applicable for imitation) Horizon for history-aware policy
    window: float = 1.0                             # (Only applicable for imitation) Window (seconds) to encode

    # Policy Parameters
    action_space: str = "ee-euler-delta"            # We use `ee-euler-delta` as our action space, but `joint-delta` is
                                                    # also implemented.

    # Training Parameters
    n_train: int = 45                               # Number of demos per task to use for training (ideally 10+)
    n_val: int = 5                                  # Number of demos per task to use for validation (ideally 2+)

    # Optimization Parameters
    bsz: int = 512                                  # Batch Size for Training -- default :: 512
    n_epochs: int = 50                              # Training Epochs to Run -- default :: 50 (selection: `val loss`)

    # Reproducibility
    seed: int = 21
    # fmt: on


def train() -> None:
    # Parse Arguments
    print("[*] LILAC Training :: Launching =>>>")
    args = ArgumentParser().parse_args()

    # Set Random Seed
    seed_everything(args.seed)

    # Create Run Directory --> FAIL/PANIC if run directory already exists (no overwrites!)
    assert args.model in {"lilac", "lila", "imitation"}, f"Model `{args.model}` not supported!"
    run_id = f"{args.model}-a={args.action_space}+n={args.n_train}-x{args.seed}"
    run_directory = Path("runs") / run_id
    os.makedirs(run_directory, exist_ok=False)

    # Build Dataset for Respective Model...
    if args.model == "imitation":
        train_ds, val_ds = get_imitation_dataset(
            run_directory,
            args.tasks,
            args.action_space,
            data_directory=args.data_directory,
            n_train=args.n_train,
            n_val=args.n_val,
            horizon=args.horizon,
            window=args.window,
        )
    elif args.model == "lila":
        train_ds, val_ds = get_lila_dataset(
            run_directory,
            args.tasks,
            args.action_space,
            data_directory=args.data_directory,
            n_train=args.n_train,
            n_val=args.n_val,
        )
    elif args.model == "lilac":
        train_ds, val_ds = get_lilac_dataset(
            run_directory,
            args.tasks,
            args.action_space,
            data_directory=args.data_directory,
            n_train=args.n_train,
            n_val=args.n_val,
        )
    else:
        raise ValueError(f"Dataset initializer for Model `{args.model}` not supported!")

    # Create DataLoaders
    print("[*] Creating DataLoaders...")
    train_loader = DataLoader(train_ds, batch_size=args.bsz, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.bsz, num_workers=4, shuffle=False)

    # Instantiate Model (as LightningModule for easy training)
    if args.model == "imitation":
        model = Imitation(
            state_dim=train_ds.datasets[0].state_dim,
            language_dim=train_ds.datasets[0].language_dim,
            action_space=args.action_space,
            action_dim=train_ds.datasets[0].action_dim,
            horizon=args.horizon,
            max_grad_steps=args.n_epochs * len(train_loader),
            run_directory=run_directory,
        )
    elif args.model == "lila":
        model = LILA(
            latent_dim=args.latent_dim,
            state_dim=train_ds.datasets[0].state_dim,
            language_dim=train_ds.datasets[0].language_dim,
            action_space=args.action_space,
            action_dim=train_ds.datasets[0].action_dim,
            run_directory=run_directory,
        )
    elif args.model == "lilac":
        model = LILAC(
            latent_dim=args.latent_dim,
            state_dim=train_ds.datasets[0].state_dim,
            language_dim=train_ds.datasets[0].language_dim,
            action_space=args.action_space,
            action_dim=train_ds.datasets[0].action_dim,
            run_directory=run_directory,
        )

    # Create Callbacks
    print("[*] Creating Callbacks & Loggers...")
    logger = MetricsLogger(name=run_id, save_dir=run_directory)
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_directory,
        filename=f"{run_id}+" + "{epoch:02d}-{train_loss:.6f}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )

    print("[*] Training...")
    trainer = Trainer(
        max_epochs=args.n_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    # After fit --> dump "self-contained" config to `run_directory`
    with open(run_directory / "config.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "action_space": args.action_space,
                "model_args": {
                    "latent_dim": args.latent_dim,
                    "state_dim": train_ds.datasets[0].state_dim,
                    "language_dim": train_ds.datasets[0].language_dim,
                    "action_space": args.action_space,
                    "action_dim": train_ds.datasets[0].action_dim,
                    "horizon": args.horizon,
                    "window": args.window,
                    "max_grad_steps": args.n_epochs * len(train_loader),
                    "run_directory": run_directory,
                },
                "checkpoint_args": {
                    "best": str(run_directory / os.path.basename(checkpoint_callback.best_model_path)),
                    "last": str(run_directory / "last.ckpt"),
                    "all": [str(run_directory / ckpt) for ckpt in checkpoint_callback.best_k_models],
                },
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    train()
