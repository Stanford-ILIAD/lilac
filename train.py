"""
train.py

Core training script -- loads and preprocesses, instantiates a Lightning Module, and runs training. Fill in with more
repository/project-specific training details!

Run with: `python train.py --config conf/config.yaml`
"""
from datetime import datetime

import hydra

from conf import Config


@hydra.main(config_path=None, config_name="config")
def train(cfg: Config) -> None:
    # Parse Hydra Configuration Dictionary...
    print("[*] LILAC :: Launching =>>>")
    run_id, seed = cfg.run_id, cfg.seed
    print("\t=>> Thunder is good, thunder is impressive; but it is Lightning that does all the work (Mark Twain)...")

    # Create Unique Run Name
    if run_id is None:
        run_id = f"lilac+{datetime.now().strftime('%Y-%m-%d-%H:%M')}"

    # Do stuff...


if __name__ == "__main__":
    train()
