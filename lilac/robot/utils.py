"""
transforms.py

Utility class containing logic for converting to-from quaternions & euler angles + other rotation helpers!

Note :: Assumes all angles are specified in *radians* (not degrees)!
"""
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def vec2reordered_mat(vec: Tuple[int]) -> np.ndarray:
    """Convert from axis ordering to a matrix transform."""
    reorder = np.zeros((len(vec), len(vec)))
    for i in range(len(vec)):
        ind = int(abs(vec[i])) - 1
        reorder[i, ind] = np.sign(vec[i])
    return reorder


def quat2euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to euler angles (xyz) in radians by way of rotation matrix."""
    return R.from_quat(quat).as_euler("xyz")


def quat2euler_degrees(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to euler angles (xyz) in degrees by way of rotation matrix."""
    return R.from_quat(quat).as_euler("xyz", degrees=True)


def euler2quat(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles (xyz) in radians to quaternion by way of rotation matrix."""
    return R.from_euler("xyz", euler).as_quat()


def euler2quat_degrees(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles (xyz) in degrees to quaternion by way of rotation matrix."""
    return R.from_euler("xyz", euler, degrees=True).as_quat()


def mat2euler(mat: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to the corresponding euler angle representation."""
    return R.from_matrix(mat).as_euler("xyz")


def euler2mat(euler: np.ndarray) -> np.ndarray:
    """Convert a set of euler angles to the corresponding rotation matrix."""
    return R.from_euler("xyz", euler).as_matrix()


def mat2quat(mat: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to the corresponding quaternion representation."""
    return R.from_matrix(mat).as_quat()


def quat2mat(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to the corresponding rotation matrix."""
    return R.from_quat(quat).as_matrix()


def add_quats(delta: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Perform quaternion addition =>> delta * source."""
    return (R.from_quat(delta) * R.from_quat(source)).as_quat()


def subtract_quats(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Perform quaternion subtraction =>> target * conj(source)."""
    return (R.from_quat(target) * R.from_quat(source).inv()).as_quat()


def add_euler(delta: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Add euler angles by way of conversion to rotation matrix."""
    return (R.from_euler("xyz", delta) * R.from_euler("xyz", source)).as_euler("xyz")


def subtract_euler(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Subtract euler angles by way of conversion to rotation matrix."""
    return (R.from_euler("xyz", target) * R.from_euler("xyz", source).inv()).as_euler("xyz")
