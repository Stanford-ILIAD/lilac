"""
env.py

Core abstraction over the physical Franka Panda Robot hardware, sensors, and internal robot state. Follows a standard
OpenAI Gym-like API.

Note: When developing on non-linux, the `torchcontrol` and `polymetis` imports will be broken, as wheels for Polymetis
on these platforms don't exist; don't panic! If you need IDE support, you can install the Polymetis source as a
submodule, and configure your PYTHONPATH/IDE/Editor setting such that the bindings are loaded.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
import torch
import torchcontrol as toco
from gym import Env
from polymetis import GripperInterface, RobotInterface

from lilac.constants import (
    CONTROL_HZ,
    GRIPPER_FORCE,
    GRIPPER_MAX_WIDTH,
    GRIPPER_SPEED,
    HOME_POSE,
    REC_KQ_GAINS,
    REC_KQD_GAINS,
    REC_KX_GAINS,
    REC_KXD_GAINS,
)
from lilac.robot.utils import add_euler, add_quats, euler2quat, quat2euler


# Silence OpenAI Gym Deprecation Warnings
gym.logger.setLevel(logging.ERROR)


class Rate:
    def __init__(self, frequency: float) -> None:
        """
        Maintains a constant control rate for the POMDP loop.

        :param frequency: Polling frequency, in Hz.
        """
        self.period, self.last = 1.0 / frequency, time.time()

    def sleep(self) -> None:
        current_delta = time.time() - self.last
        sleep_time = max(0.0, self.period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self.last = time.time()


class FrankaEnv(Env):
    def __init__(self, do_kinesthetic: bool = False, franka_ip: str = "172.16.0.1") -> None:
        """
        Initialize a *physical* Franka Environment, with the given home pose, polling Hz, and connection parameters.

        :param do_kinesthetic: Whether to initialize joint controller with zero PD gains for kinesthetic demonstration.
        :param franka_ip: IP address of the Franka Control box for connecting to the robot.
        """
        self.home, self.hz, self.rate, self.franka_ip = HOME_POSE, CONTROL_HZ, Rate(CONTROL_HZ), franka_ip
        self.robot, self.gripper, self.kq, self.kqd, self.kx, self.kxd = None, None, None, None, None, None
        self.do_kinesthetic = do_kinesthetic

        # Pose & Robot State Trackers
        self.current_joint_pose, self.current_ee_pose, self.current_gripper_state = None, None, None
        self.initial_ee_pose, self.initial_gripper_state, self.gripper_open, self.gripper_act = None, None, True, None

        # Expected/Desired Poses (for PD Controller Deltas)
        self.expected_q, self.expected_ee_quat, self.expected_ee_euler = None, None, None
        self.desired_pose = {"pos": None, "ori": None}

        # Initialize Robot and Cartesian Impedance Controller
        #   => Cartesian Impedance uses `HybridJointImpedanceController` so we can send `joint` or `end-effector` poses!
        self.reset()

    def start_lilac_controller(self) -> None:
        """Start a HybridJointImpedanceController with all 4 of the desired gains; Polymetis defaults don't set both."""
        torch_policy = toco.policies.HybridJointImpedanceControl(
            joint_pos_current=self.robot.get_joint_positions(),
            Kq=self.robot.Kq_default if self.kq is None else self.kq,
            Kqd=self.robot.Kqd_default if self.kqd is None else self.kqd,
            Kx=self.robot.Kx_default if self.kx is None else self.kx,
            Kxd=self.robot.Kxd_default if self.kxd is None else self.kxd,
            robot_model=self.robot.robot_model,
            ignore_gravity=self.robot.use_grav_comp,
        )
        self.robot.send_torch_policy(torch_policy=torch_policy, blocking=False)

    def set_controller(self) -> None:
        # Special handling *only* applicable for "kinesthetic teaching"
        if self.do_kinesthetic:
            self.kq, self.kqd, self.kx, self.kxd = REC_KQ_GAINS, REC_KQD_GAINS, REC_KX_GAINS, REC_KXD_GAINS
        else:
            self.kq, self.kqd, self.kx, self.kxd = None, None, None, None

        # Start a *Cartesian Impedance Controller* with the desired gains...
        #   Note: P/D values of "None" default to HybridJointImpedance PD defaults from Polymetis
        #         |-> These values are defined in the default launch_robot YAML (`robot_client/franka_hardware.yaml`)
        self.start_lilac_controller()

    def get_obs(self) -> Dict[str, np.ndarray]:
        # Fetch Current Proprioceptive State directly from Robot...
        new_joint_pose = self.robot.get_joint_positions().numpy()
        new_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        new_ee_rot = new_ee_pose[3:]
        new_gripper_state = self.gripper.get_state()

        # Assemble Observation & Update Stateful Trackers...
        obs = {
            "q": new_joint_pose,
            "qdot": self.robot.get_joint_velocities().numpy(),
            "ee_pose": new_ee_pose,
            "gripper_width": new_gripper_state.width,
            "gripper_max_width": GRIPPER_MAX_WIDTH,
            "gripper_open": self.gripper_open,
            "gripper_action": self.gripper_act,
        }
        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = new_joint_pose, new_ee_pose, new_ee_rot
        self.current_gripper_state = {
            "width": new_gripper_state.width,
            "max_width": GRIPPER_MAX_WIDTH,
            "gripper_open": self.gripper_open,
            "gripper_action": self.gripper_act,
        }
        return obs

    def set_kinesthetic(self, do_kinesthetic: bool) -> Dict[str, np.ndarray]:
        self.do_kinesthetic = do_kinesthetic
        return self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        # Initialize Robot Interface (Polymetis) & Reset to Home Pose
        self.robot = RobotInterface(ip_address=self.franka_ip, enforce_version=False)
        self.robot.set_home_pose(torch.Tensor(self.home))
        self.robot.go_home()

        # Initialize Current Joint & EE poses...
        self.current_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        self.current_ee_rot = self.current_ee_pose[3:]
        self.current_joint_pose = self.robot.get_joint_positions().numpy()

        # Set Robot Motion Controller (e.g., joint or cartesian impedance...)
        self.set_controller()

        # Initialize Gripper Interface --> Open Gripper on each `reset()`
        self.gripper = GripperInterface(ip_address=self.franka_ip)
        self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)

        # Set Gripper State...
        self.gripper_open, self.gripper_act = True, np.array(0.0)
        self.initial_gripper_state = self.current_gripper_state = {
            "width": self.gripper.get_state().width,
            "max_width": GRIPPER_MAX_WIDTH,
            "gripper_open": self.gripper_open,
            "gripper_action": self.gripper_act,
        }

        # Set `expected` and `desired_pose`
        self.expected_q, self.expected_ee_quat = self.current_joint_pose.copy(), self.current_ee_pose.copy()
        self.expected_ee_euler = np.concatenate([self.expected_ee_quat[:3], quat2euler(self.expected_ee_quat[3:])])
        self.desired_pose = {"pos": self.current_ee_pose[:3], "ori": self.current_ee_rot}

        # Return Initial Observation
        return self.get_obs()

    def step(
        self, action: Optional[np.ndarray], open_gripper: Optional[bool] = None, action_space: str = "ee-euler-delta"
    ) -> Dict[str, np.ndarray]:
        if action is not None:
            if action_space == "joint-delta":
                # Compute next joint pose & act!
                next_q = self.expected_q = self.expected_q + action
                self.robot.update_desired_joint_positions(torch.from_numpy(next_q).float())

            elif action_space == "ee-euler-delta":
                # Compute next EE pose --> add Euler angles via intermediate conversion to quaternion (via `add_euler`)
                #   > Note: Polymetis requires orientation specified as quaternion, so we do a final conversion here...
                next_pos = self.expected_ee_euler[:3] = self.expected_ee_euler[:3] + action[:3]
                next_euler = self.expected_ee_euler[3:] = add_euler(action[3:], self.expected_ee_euler[3:])
                next_quat = euler2quat(next_euler)
                self.robot.update_desired_ee_pose(
                    position=torch.from_numpy(next_pos).float(), orientation=torch.from_numpy(next_quat).float()
                )

            else:
                # Panic if the `action_space` isn't recognized!
                raise NotImplementedError(f"Support for Action Space `{action_space}` not yet implemented!")

        # Discrete (Binary) Grasping --> (Set Open/Close)
        if open_gripper is not None and (self.gripper_open ^ open_gripper):
            # True --> Open Gripper, otherwise --> Close Gripper
            self.gripper_open = open_gripper
            if open_gripper:
                self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)
            else:
                self.gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)

        # Sleep, according to control frequency
        self.rate.sleep()

        # Return observation
        return self.get_obs()

    def close(self) -> None:
        # Terminate Policy
        self.robot.terminate_current_policy()

        # Garbage collection & sleep just in case...
        del self.robot
        self.robot, self.gripper = None, None
        time.sleep(1)
