"""
config.py

Base Hydra Structured Config for `lilac`. Shards configuration over datasets, preprocessing, models,
evaluation, etc. Inheritance and core paths can all be specified via new Structured Configurations, or via the
command line.
"""
from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class Config:
    run_id: str = None
    seed: int = 21


# Retrieve Singleton ConfigStore & store Config
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
