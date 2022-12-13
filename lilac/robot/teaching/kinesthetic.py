"""
kinesthetic.py

Core interface for collection demonstrations via kinesthetic teaching; this class is standalone for LILAC, but is
abstracted in such a way that extending to other teaching modalities (via teleoperation, virtual reality) should be
straightforward.
"""
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from lilac.constants import DIRECTIONAL_CORRECTIONS, GRIPPER_TOLERANCE, Q_DELTA_PAUSE_TOLERANCE
from lilac.robot.env import FrankaEnv


class KinestheticTeachingInterface:
    def __init__(
        self,
        env: FrankaEnv,
        task: str,
        resume: bool,
        max_time_per_demo: int = 30,
        data_directory: Path = Path("data"),
    ) -> None:
        """
        Initialize a Kinesthetic Teaching Interface with the requisite parameters for serialization & resuming.

        :param env: `FrankaEnv` robot environment (for interfacing with the physical robot).
        :param task: Identifier for the "task type" (meta-task) we're collecting.
        :param resume: Whether to resume demonstration collection for the given task (or start anew).
        :param max_time_per_demo: Max time (in seconds) for recording a demonstration (default: 30s).
        :param data_directory: Path to parent directory for saving raw/replayed demonstrations.
        """
        self.env, self.task, self.resume, self.max_time_per_demo = env, task, resume, max_time_per_demo
        self.data_directory = data_directory

        # We collect kinesthetic demonstrations via an *interleaved* process consisting of the following 3 steps:
        #   1) Manually guide the robot/gripper to complete a given task/correction
        #   2) Reset the environment (for the given *instance*)
        #   3) Ensure that the demonstration is feasible by *replaying* it, ensuring that the task is completed.
        #       > Why? The PD gains for teaching vs. control are different, leading to slightly different behavior in
        #         some cases -- we want to ensure that our demonstrations are all collected with the same PD controller
        #         used for actual rollouts!
        self.raw_dir = self.data_directory / self.task / "record-raw"
        self.playback_dir = self.data_directory / self.task / "playback-final"
        os.makedirs(self.raw_dir, exist_ok=self.resume)
        os.makedirs(self.playback_dir, exist_ok=self.resume)

        # Create Counter for Demonstrations / Handle Resume
        self.demo_index = 1 + max([0] + [int(x.split("-")[-1].split(".")[0]) for x in os.listdir(self.playback_dir)])

    def record_demo(self) -> Optional[Path]:
        print(f"[*] Starting to Record Demonstration `{self.demo_index}`...")
        demo_file = self.raw_dir / f"{self.task}-{self.demo_index}.npz"

        # Set `do_kinesthetic = True`, reset environment, and wait on user input...
        self.env.set_kinesthetic(do_kinesthetic=True)
        user_input = input(
            f"Ready to record!\n\tYou have `{self.max_time_per_demo}` secs to complete the demo, and can use"
            " `Ctrl-C` to stop anytime.\n\tPress (r) to reset, and any other key to start..."
        )

        # Reset...
        if user_input.startswith("r"):
            return None

        # Go, go, go --> start recording; for "raw" recording, we only care about joint state, ee pose, gripper widths
        print("\t=>> Started recording... go, go, go!")
        observations = []
        try:
            for _ in range(int(self.max_time_per_demo * self.env.hz) - 1):
                obs = self.env.step(action=None)
                observations.append(obs)
        except KeyboardInterrupt:
            print("\t=>> Caught KeyboardInterrupt, stopping recording...")

        # Close environment (terminate any errant controllers)
        self.env.close()

        # Extract relevant parts of the observations...
        raw_dict = {k: [] for k in ["q", "ee_pose", "gripper_width"]}
        for obs in observations:
            for k in raw_dict:
                raw_dict[k].append(obs[k])
        raw_dict = {k: np.array(v) for k, v in raw_dict.items()}
        raw_dict["rate"], raw_dict["home"] = self.env.hz, self.env.home

        # Save "raw" file...
        np.savez(str(demo_file), **raw_dict)

        return demo_file

    def playback_demo(self, demo_file: Path) -> None:
        print(f"[*] Starting Playback for Demonstration `{self.demo_index}`...")
        trajectory = np.load(str(demo_file))

        # We use a Hybrid Joint-Cartesian Impedance Controller -- for simplicity, we'll just use joint-delta control...
        q_states, gripper_widths = trajectory["q"], trajectory["gripper_width"]

        # Filter out pauses...
        q_future, q_past = q_states[1:], q_states[:-1]
        q_diff_norms = np.linalg.norm(q_future - q_past, axis=1)
        keep_idxs = (q_diff_norms > Q_DELTA_PAUSE_TOLERANCE).nonzero()[0]
        q_states, gripper_widths = q_states[keep_idxs], gripper_widths[keep_idxs]

        # Turn `gripper_widths` (continuous) into binary Grasp/Ungrasp actions...
        future_widths, past_widths = np.array(gripper_widths[1:]), np.array(gripper_widths[:-1])
        gripper_diffs = future_widths - past_widths
        gripper_diffs[np.abs(gripper_diffs) < GRIPPER_TOLERANCE] = 0
        binary_grasps = [None]
        for diff in gripper_diffs:
            if np.isclose(0, diff):
                binary_grasps.append(None)
            elif diff > 0:
                binary_grasps.append(True)
            else:
                binary_grasps.append(False)

        # Manually reset the environment, press [ENTER] => sets "regular" control, resets robot, and executes playback!
        input("\tReady to playback! Reset the environment, then get out of the way and hit any key to continue...")
        observations, actions = [self.env.set_kinesthetic(do_kinesthetic=False)], []
        for idx in range(len(q_states) - 1):
            q_state, next_q_state = q_states[idx], q_states[idx + 1]

            # Compute Delta Action (in joint space)
            action = next_q_state - q_state
            actions.append(action)

            # Call `step` with computed action...
            observations.append(self.env.step(action, open_gripper=binary_grasps[idx + 1], action_space="joint-delta"))

        # Extract full observation / trajectory data and serialize...
        playback_dict = {k: [] for k in list(observations[0].keys())}
        for obs in observations:
            for k in playback_dict.keys():
                playback_dict[k].append(obs[k])
        playback_dict = {k: np.array(v) for k, v in playback_dict.items()}
        playback_dict["actions"] = np.array(actions)
        playback_dict["gripper_actions"] = np.array(binary_grasps)
        playback_dict["rate"], playback_dict["home"] = self.env.hz, self.env.home

        # Save to `playback-final`
        np.savez(str(self.playback_dir / demo_file.name), **playback_dict)

        # Cleanup
        self.env.close()

    def collect(self) -> None:
        # Start Demonstration Collection Loop --> "interleaved" demos start with kinesthetic teaching, then playback!
        while True:
            demo_file = self.record_demo()
            self.playback_demo(demo_file)

            # Copy `original-state.json` to {self.task}-{self.demo_index}.json and ask human to record offsets (cm)!
            offset_file = self.playback_dir / f"{self.task}-{self.demo_index}.json"
            shutil.copy(Path("data/original-state.json"), offset_file)
            input(
                f"[*] Playback complete; please record object offsets in `{offset_file}`! Press any key to continue..."
            )

            # Move on?
            self.demo_index += 1
            do_quit = input("[*] Next? Press any key to continue recording demonstrations, or (q) to quit...")
            if do_quit.startswith("q"):
                break
