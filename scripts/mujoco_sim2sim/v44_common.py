"""Shared constants and math for the v4.4 G1 control-policy deployment."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/rsl_rl/g1_dribbling/2026-07-29_03-58-23_v4.4/model_93000.pt"
)
DEFAULT_MOTION_BANK = REPO_ROOT / "motions/master-modified"
DEFAULT_MJCF = (
    REPO_ROOT
    / "source/whole_body_tracking/soccer/assets/unitree_description/mjcf/g1_actuator.xml"
)

OBS_DIM = 172
ACTION_DIM = 29
RNN_NUM_LAYERS = 2
RNN_HIDDEN_DIM = 128
CONTROL_DT = 0.02
PHYSICS_DT = 0.005
BALL_RADIUS = 0.11
BALL_MASS = 0.4

# Isaac/PhysX material and rigid-body settings used by the dribbling task.
ISAAC_GROUND_DYNAMIC_FRICTION = 0.95
ISAAC_BALL_DYNAMIC_FRICTION = 0.62
ISAAC_BALL_LINEAR_DAMPING = 0.18
ISAAC_BALL_ANGULAR_DAMPING = 0.18
# Robot collision materials are randomized uniformly at startup in Isaac:
# dynamic friction in [0.3, 1.2]. MuJoCo's deterministic aligned profile uses
# the expected value so repeated rollouts remain directly comparable.
ISAAC_ROBOT_DYNAMIC_FRICTION_MEAN = 0.75
ISAAC_FOOT_GROUND_EFFECTIVE_FRICTION = (
    ISAAC_GROUND_DYNAMIC_FRICTION * ISAAC_ROBOT_DYNAMIC_FRICTION_MEAN
)
ISAAC_BALL_GROUND_EFFECTIVE_FRICTION = (
    ISAAC_BALL_DYNAMIC_FRICTION * ISAAC_GROUND_DYNAMIC_FRICTION
)
ISAAC_BALL_ROBOT_EFFECTIVE_FRICTION = (
    ISAAC_BALL_DYNAMIC_FRICTION * ISAAC_ROBOT_DYNAMIC_FRICTION_MEAN
)

# The policy and motion files use Isaac Lab's articulation order. MuJoCo uses
# the kinematic-tree order in g1_actuator.xml. Always translate by name.
ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

UPPER_BODY_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ISAACLAB_TO_MUJOCO_REINDEX = np.asarray(
    [ISAACLAB_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES],
    dtype=np.int32,
)
MUJOCO_TO_ISAACLAB_REINDEX = np.asarray(
    [MUJOCO_JOINT_NAMES.index(name) for name in ISAACLAB_JOINT_NAMES],
    dtype=np.int32,
)
UPPER_BODY_ISAACLAB_IDS = np.asarray(
    [ISAACLAB_JOINT_NAMES.index(name) for name in UPPER_BODY_JOINT_NAMES],
    dtype=np.int32,
)


def _joint_map(default: float = 0.0) -> dict[str, float]:
    return {name: default for name in MUJOCO_JOINT_NAMES}


DEFAULT_JOINT_POS = _joint_map()
DEFAULT_JOINT_POS.update(
    {
        "left_hip_pitch_joint": -0.312,
        "right_hip_pitch_joint": -0.312,
        "left_knee_joint": 0.669,
        "right_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "right_ankle_pitch_joint": -0.363,
        "left_elbow_joint": 0.6,
        "right_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

JOINT_STIFFNESS = _joint_map()
JOINT_DAMPING = _joint_map()
JOINT_EFFORT_LIMIT = _joint_map()

for _name in MUJOCO_JOINT_NAMES:
    if "_hip_pitch_joint" in _name or "_hip_yaw_joint" in _name:
        JOINT_STIFFNESS[_name] = STIFFNESS_7520_14
        JOINT_DAMPING[_name] = DAMPING_7520_14
        JOINT_EFFORT_LIMIT[_name] = 88.0
    elif "_hip_roll_joint" in _name or "_knee_joint" in _name:
        JOINT_STIFFNESS[_name] = STIFFNESS_7520_22
        JOINT_DAMPING[_name] = DAMPING_7520_22
        JOINT_EFFORT_LIMIT[_name] = 139.0
    elif "_ankle_pitch_joint" in _name or "_ankle_roll_joint" in _name:
        JOINT_STIFFNESS[_name] = 2.0 * STIFFNESS_5020
        JOINT_DAMPING[_name] = 2.0 * DAMPING_5020
        JOINT_EFFORT_LIMIT[_name] = 50.0
    elif _name in {"waist_roll_joint", "waist_pitch_joint"}:
        JOINT_STIFFNESS[_name] = 2.0 * STIFFNESS_5020
        JOINT_DAMPING[_name] = 2.0 * DAMPING_5020
        JOINT_EFFORT_LIMIT[_name] = 50.0
    elif _name == "waist_yaw_joint":
        JOINT_STIFFNESS[_name] = STIFFNESS_7520_14
        JOINT_DAMPING[_name] = DAMPING_7520_14
        JOINT_EFFORT_LIMIT[_name] = 88.0
    elif "_wrist_pitch_joint" in _name or "_wrist_yaw_joint" in _name:
        JOINT_STIFFNESS[_name] = STIFFNESS_4010
        JOINT_DAMPING[_name] = DAMPING_4010
        JOINT_EFFORT_LIMIT[_name] = 5.0
    else:
        JOINT_STIFFNESS[_name] = STIFFNESS_5020
        JOINT_DAMPING[_name] = DAMPING_5020
        JOINT_EFFORT_LIMIT[_name] = 25.0


def _array_in_mujoco_order(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values[name] for name in MUJOCO_JOINT_NAMES], dtype=np.float32)


DEFAULT_JOINT_POS_MJ = _array_in_mujoco_order(DEFAULT_JOINT_POS)
JOINT_STIFFNESS_MJ = _array_in_mujoco_order(JOINT_STIFFNESS)
JOINT_DAMPING_MJ = _array_in_mujoco_order(JOINT_DAMPING)
JOINT_EFFORT_LIMIT_MJ = _array_in_mujoco_order(JOINT_EFFORT_LIMIT)
ACTION_SCALE_MJ = (
    0.25 * JOINT_EFFORT_LIMIT_MJ / np.maximum(JOINT_STIFFNESS_MJ, 1.0e-8)
).astype(np.float32)

DEFAULT_JOINT_POS_ISAAC = DEFAULT_JOINT_POS_MJ[MUJOCO_TO_ISAACLAB_REINDEX]
ACTION_SCALE_ISAAC = ACTION_SCALE_MJ[MUJOCO_TO_ISAACLAB_REINDEX]


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-9:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_apply_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    pure = np.asarray([0.0, *np.asarray(vector, dtype=np.float64)], dtype=np.float64)
    return quat_multiply(
        quat_multiply(quat_conjugate(quat), pure),
        quat,
    )[1:]


def strip_yaw(quat: np.ndarray) -> np.ndarray:
    """Match Isaac Lab's align_body_quat_yaw_to_task_forward()."""
    quat = normalize_quat(quat)
    w, x, y, z = quat
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    inv_yaw = np.asarray(
        [math.cos(-0.5 * yaw), 0.0, 0.0, math.sin(-0.5 * yaw)],
        dtype=np.float64,
    )
    return normalize_quat(quat_multiply(inv_yaw, quat))
