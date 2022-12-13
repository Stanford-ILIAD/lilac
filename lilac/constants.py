"""
constants.py

Centralized, top-level script defining any & all *project-wide* constants (e.g., demonstration/policy frequency).
"""
import numpy as np


# === Important Real-Robot Constants ===
CONTROL_HZ = 10

# Reset or "Home" Pose for the Robot -- we only use the Libfranka default in this work...
HOME_POSE = np.asarray([0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0])

# Libfranka Constants
#   > Ref: Gripper constants from: https://frankaemika.github.io/libfranka/grasp_object_8cpp-example.html
GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 60, 0.08570, 0.01

# Joint Delta "Pause" Tolerance --> for Filtering Out Pauses in Kinesthetic Demonstrations
Q_DELTA_PAUSE_TOLERANCE = 5e-3

# Joint Controller gains for recording demonstrations -- we want a compliant robot, so setting all gains to ~0.
REC_KQ_GAINS, REC_KQD_GAINS = [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]
REC_KX_GAINS, REC_KXD_GAINS = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]

# Valid Action Spaces & Dimensionality
ACTION_DIMENSIONALITY = {"joint-delta": 7, "ee-euler-delta": 6}

# === Traced Demonstration Primitives ===
# Stores `ee-euler-delta` index [0, 6) and corresponding "direction" (sign)

# fmt: off
DIRECTIONAL_CORRECTIONS = {
    # Position
    "right": {0: +1.0}, "left": {0, -1.0},
    "up": {2: +1.0}, "down": {2: -1.0},
    "forward": {1: +1.0}, "backward": {1: -1.0},

    # Orientation
    "twist-right": {5: -1.0}, "twist-left": {5, +1.0},
    "tilt-right": {4: +1.0}, "tilt-left": {4, -1.0},
    "tilt-up": {3: +1.0}, "tilt-down": {3, -1.0},
}
# fmt: on

# === Object State Parameters -- update this as you change the environment! ===
TRACKED_OBJECTS = sorted(
    [
        "apple",
        "banana",
        "black-mug",
        "blue-marker",
        "book",
        "crumpled-paper",
        "drawer",
        "eraser",
        "green-marker",
        "marker-holder",
        "notebook",
        "plant-holder",
        "shelf",
        "tape",
        "trash-can",
        "yellow-cup",
    ]
)
N_OBJECTS = len(TRACKED_OBJECTS)
