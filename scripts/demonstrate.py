"""
demonstrate.py

Flexible demonstration collection script for recording kinesthetic demonstrations (manually moving the robot arm with
gravity compensation turned on) -- states are logged at a fixed frequency (10 Hz) and then serialized to disk.

To ensure fidelity, all demonstrations are "replayed" deterministically to verify behavior – these replayed
demonstrations are used for training (and optionally, for collecting *visual* states in the future).

References:
    - https://github.com/AGI-Labs/franka_control/blob/master/record.py
    - https://github.com/AGI-Labs/franka_control/blob/master/playback.py
"""
from pathlib import Path

from tap import Tap

from lilac.robot.env import FrankaEnv
from lilac.robot.teaching import KinestheticTeachingInterface


class ArgumentParser(Tap):
    # fmt: off
    task: str                                       # Identifier for the "task type" (meta-task) we're collecting.
    resume: bool = True                             # Whether to resume demonstration collection for the given task.
    max_time_per_demo: int = 30                     # Max time (in seconds) for recording a demonstration (default: 30s)

    # Serialization Parameters
    data_directory: Path = Path("data")             # Path to parent directory for saving raw/replayed demonstrations.
    # fmt: on


def demonstrate() -> None:
    print("[*] Starting Kinesthetic Demonstration Collection...")
    args = ArgumentParser().parse_args()

    # Initialize Franka Real-Robot Environment -- both Robot & Gripper should be enabled.
    #   > Note: for data collection, we set `do_kinesthetic = True`
    env = FrankaEnv(do_kinesthetic=True)

    # Build a Kinesthetic Teaching Interface & Collect Demonstrations!
    demonstration_interface = KinestheticTeachingInterface(
        env,
        args.task,
        args.resume,
        max_time_per_demo=args.max_time_per_demo,
        data_directory=args.data_directory,
    )

    # Drop into open loop demonstration collection...
    demonstration_interface.collect()


if __name__ == "__main__":
    demonstrate()
