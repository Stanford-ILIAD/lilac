"""
evaluate.py

Rolls out a trained LILAC, LILA, or Imitation learning model in an environment. For Imitation Learning, takes an
initial user-provided language utterance, and rolls-out till success/manual stop in the environment.

For LILA, operates similarly, but produces a 2-DoF control space that the user can interface with to operate the robot
to complete their specified intent.

LILAC operates similarly to LILA, but notably allows users to provide *corrections* at *any point during execution*
which produces a new control space for the user.
"""
import json
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tap import Tap
from transformers import AutoModel, AutoTokenizer

from lilac.inference import ImitationInferenceWrapper, LILACInferenceWrapper, LILAInferenceWrapper
from lilac.models import LILA, LILAC, Imitation
from lilac.robot.env import FrankaEnv


# Suppress PyGame Import Text
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa: E402


class JoystickInterface:
    def __init__(self, n_axes: int = 2) -> None:
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()

        # Joystick input/filtering parameters
        self.n_axes, self.deadband = n_axes, 0.1

    def input(self) -> Tuple[np.ndarray, bool, bool, bool, bool, bool]:
        pygame.event.get()
        zs = []

        # Grab the "right" joystick (assumes Logitech Controller + Right-Handed User)
        for i in range(3, 3 + self.n_axes):
            z = self.gamepad.get_axis(i)
            if abs(z) < self.deadband:
                z = 0.0
            zs.append(z)

        # Button presses in {a, b, x, y, <start-btn> = stop}
        a, b, x, y, stop = [self.gamepad.get_button(idx) for idx in [0, 1, 2, 3, 7]]
        return np.array(zs), a, b, x, y, stop


class ArgumentParser(Tap):
    # fmt: off
    run_directory: Path                     # Path to run directory (e.g., "runs/lilac-a=ee-euler-delta+n=45-x21/")
    checkpoint: str = "best"                # Checkpoint to use from the specified directory in < best | last >
    # fmt: on


def evaluate() -> None:
    print("[*] Running LILAC Evaluation =>>")
    args = ArgumentParser().parse_args()
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

    # Establish Joystick Interfaces (we'll use buttons for all strategies, as just a nice way to handle control flow)
    joystick = JoystickInterface(n_axes=2)

    # Load Model from Checkpoint
    print(f"[*] Loading Model from Run Directory `{args.run_directory}`")
    with open(args.run_directory / "config.json", "w") as f:
        config = json.load(f)

    print("\t=> Loading Prerequisite Language Model...")
    os.makedirs("cache/paraphrase", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
    )
    lm = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
    )

    # Load Actual Policy/Controller Weights...
    checkpoint_path, model_args = config["checkpoint_args"][args.checkpoint], config["model_args"]
    if config["model"] == "imitation":
        print("\t=> Loading Imitation Learning Model from Checkpoint...")
        model = Imitation(
            state_dim=model_args["state_dim"],
            language_dim=model_args["language_dim"],
            action_space=model_args["action_space"],
            action_dim=model_args["action_dim"],
            horizon=model_args["horizon"],
            max_grad_steps=model_args["max_grad_steps"],
            run_directory=Path(model_args["run_directory"]),
        )
        model.load_state_dict(checkpoint_path, strict=True)
        model.eval()

        # Create `agent` --> wraps model in a nifty inference wrapper...
        agent = ImitationInferenceWrapper(model, tokenizer, lm, model_args["horizon"], model_args["window"])

    elif config["model"] == "lila":
        print("\t=> Loading LILA Model from Checkpoint...")
        model = LILA(
            latent_dim=model_args["latent_dim"],
            state_dim=model_args["state_dim"],
            language_dim=model_args["language_dim"],
            action_space=model_args["action_space"],
            action_dim=model_args["action_dim"],
            run_directory=Path(model_args["run_directory"]),
        )
        model.load_state_dict(checkpoint_path, strict=True)
        model.eval()

        # Create `agent` --> wraps model in a nifty inference wrapper...
        agent = LILAInferenceWrapper(model, tokenizer, lm)

    elif config["model"] == "lilac":
        print("\t=> Loading LILAC Model from Checkpoint...")
        model = LILAC(
            latent_dim=model_args["latent_dim"],
            state_dim=model_args["state_dim"],
            language_dim=model_args["language_dim"],
            action_space=model_args["action_space"],
            action_dim=model_args["action_dim"],
            run_directory=Path(model_args["run_directory"]),
        )
        model.load_state_dict(checkpoint_path, strict=True)
        model.eval()

        # Create `agent` --> wraps model in a nifty inference wrapper...
        agent = LILACInferenceWrapper(model, tokenizer, lm)
    else:
        raise ValueError(f"Model type `{config['model']}` is not supported!")

    # Setup Franka Environment -- Robot & Gripper should *both* be enabled...
    print("[*] Setting up Robot...")
    env = FrankaEnv(do_kinesthetic=False)

    # Drop into an Infinite Loop, keeping the policy (Imitation Learning) or control loop (LILA/LILAC) running...
    if config["model"] == "imitation":
        input(
            "[*] Starting Imitation Policy Execution\n"
            "\t=>> Press `START` on the joystick to terminate the episode...\n"
            "\t=>> Press any key to continue..."
        )
    elif config["model"] == "lila":
        input(
            "[*] Starting LILA Execution\n"
            "\t=>> Use the right joystick to maneuver the robot, and press (B) to trigger the gripper!\n"
            "\t=>> Press `START` on the joystick to terminate the episode...\n"
            "\t=>> Press any key to continue..."
        )
    elif config["model"] == "lilac":
        input(
            "[*] Starting LILAC Execution\n"
            "\t=>> Use the right joystick to maneuver the robot, and press (B) to trigger the gripper!\n"
            "\t=>> Press (A) to signal a new language correction -- press (Y) to exit correction mode!\n"
            "\t=>> Press `START` on the joystick to terminate the episode...\n"
            "\t=>> Press any key to continue..."
        )

    # Get User-Provided Language Instruction to start things off & call `agent.embed`
    utterance = input("Enter Instruction =>> ")
    agent.embed(utterance)
    utterance_stack = [utterance]

    # Let's loop!
    try:
        # Set Timers to prevent "ghost" button presses...
        press_b, press_b_t, press_y, press_y_t = False, time.time(), False, time.time()
        while True:
            # Get observation --> update agent state!
            obs = env.get_obs()
            agent.encode_state(obs["q"], obs["ee_pose"], obs["gripper_open"])

            # Poll Joystick...
            zs, a, b, x, y, stop = joystick.input()

            # Exit Condition (valid for all model/agents)...
            if stop:
                print("[*] You pressed <START>, so exiting...")
                break

            # Run the Imitation Learning Policy...
            if config["model"] == "imitation":
                action, open_gripper = agent.act()

                # Execute!
                env.step(action, open_gripper=open_gripper, action_space=model.action_space)

            # Otherwise, Latent Actions handling... handle Gripper on (B)
            elif b and config["model"] in {"lila", "lilac"}:
                # Default button handling to make sure "holding" button for a few milliseconds doesn't loop...
                if time.time() - press_b_t > 0.25:
                    press_b = False

                # Trigger Gripper...
                if not press_b:
                    press_b, press_b_t = True, time.time()
                    env.step(None, open_gripper=not env.gripper_open, action_space=model.action_space)

            # Handle (A) for LILAC => Push onto "utterance stack"
            elif a and config["model"] == "lilac":
                correction = input("[*] You pressed (A); please enter a correction: ")
                utterance_stack.append(correction)
                agent.embed(correction)

            # Handle (Y) for LILAC => Pop off of "utterance stack"
            elif y and config["model"] == "lilac":
                # Default button handling to make sure "holding" button for a few milliseconds doesn't loop...
                if time.time() - press_y_t > 0.25:
                    press_y = False

                # Actually handle "pop"
                if not press_y:
                    press_y, press_y_t = True, time.time()

                    # Pop all the latest corrections (make sure at least the initial instruction is on stack)
                    print("\t=>> Terminating Correction Handling -> Reverting to LILAC!")
                    utterance_stack = utterance_stack[:1]
                    agent.embed(utterance_stack[-1])

            # Otherwise... we're just in Latent Actions Control!
            elif zs.sum() != 0:
                action = agent.decode_action(zs)
                env.step(action, open_gripper=None, action_space=model.action_space)

    except KeyboardInterrupt:
        pass

    finally:
        print("\n[*] Terminating... that's all, folks!")


if __name__ == "__main__":
    evaluate()
