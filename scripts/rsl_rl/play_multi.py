"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Simulation steps to run (default: sum of motion frames for --motion_path, else motion length / 600).",
)
parser.add_argument("--dual_view", action="store_true", default=False, help="Record split-screen video (front + back view).")
parser.add_argument(
    "--cam_layout",
    type=str,
    default="task_front",
    choices=["diagonal", "task_front", "task_front_side"],
    help="Camera preset. task_front: on +X side, looks ~ -X (robot frontal); diagonal: legacy oblique.",
)
parser.add_argument(
    "--record_all_motions",
    action="store_true",
    default=False,
    help="Alias for default multi-file --motion_path behaviour (sequential playback, auto step count).",
)
parser.add_argument("--path_tracing", action="store_true", default=False, help="Use Path Tracing renderer for higher quality.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a single motion file. When specified, only this motion is played and exported.")
parser.add_argument(
    "--motion_path",
    type=str,
    default=None,
    help="Path to a motion .npz or a directory; directories are played sequentially in sorted order.",
)

parser.add_argument("--export_motion_name", type=str, default=None, help="Select one motion for exporter (required when --motion_file is used).")
parser.add_argument(
    "--disable_training_terminations",
    action="store_true",
    default=False,
    help="Disable ball_lost / ee_body_pos etc. for full-clip playback (debug only).",
)
parser.add_argument(
    "--locomotion_cmd_vx",
    type=float,
    default=None,
    help="Manual locomotion cmd: task +X linear speed (m/s). Implies manual mode at play time.",
)
parser.add_argument(
    "--locomotion_cmd_vy",
    type=float,
    default=None,
    help="Manual locomotion cmd: task +Y lateral speed (m/s).",
)
parser.add_argument(
    "--locomotion_cmd_vz",
    type=float,
    default=None,
    help="Manual locomotion cmd: vertical linear speed (m/s).",
)
parser.add_argument(
    "--locomotion_cmd_wx",
    type=float,
    default=None,
    help="Manual locomotion cmd: roll rate (rad/s).",
)
parser.add_argument(
    "--locomotion_cmd_wy",
    type=float,
    default=None,
    help="Manual locomotion cmd: pitch rate (rad/s).",
)
parser.add_argument(
    "--locomotion_cmd_speed",
    type=float,
    nargs="+",
    default=None,
    help="Polar cmd speed(s) in m/s. Multiple values = multi-segment sequence (with heading + duration).",
)
parser.add_argument(
    "--locomotion_cmd_heading",
    type=float,
    nargs="+",
    default=None,
    help="Polar cmd heading(s) rad from task +X. Must match length of --locomotion_cmd_speed.",
)
parser.add_argument(
    "--locomotion_cmd_duration",
    type=float,
    nargs="+",
    default=None,
    help="Hold duration(s) in seconds per segment. Must match length of --locomotion_cmd_speed.",
)
parser.add_argument(
    "--locomotion_task_state",
    type=str,
    nargs="+",
    choices=["idle", "dribble", "stop"],
    default=None,
    help=(
        "High-level state(s) for polar segments: idle, dribble, or stop. "
        "Must match --locomotion_cmd_speed; omitted states infer dribble for positive speed and stop for zero."
    ),
)
parser.add_argument(
    "--locomotion_cmd_wz",
    type=float,
    nargs="*",
    default=None,
    help="Optional yaw rate (rad/s) per polar segment; one value broadcasts to all segments.",
)
parser.add_argument(
    "--locomotion_cmd_loop",
    dest="locomotion_cmd_loop",
    action="store_true",
    default=True,
    help="Loop a multi-segment polar command from the final segment back to the first (default).",
)
parser.add_argument(
    "--locomotion_cmd_hold_last",
    dest="locomotion_cmd_loop",
    action="store_false",
    help="Keep the final multi-segment polar command active instead of looping.",
)
parser.add_argument(
    "--locomotion_cmd_reset_on_end",
    action="store_true",
    default=False,
    help=(
        "Reset robot, ball, and manual command sequence after the final polar segment duration. "
        "Takes precedence over --locomotion_cmd_loop/--locomotion_cmd_hold_last."
    ),
)
parser.add_argument(
    "--diagnostic",
    action="store_true",
    default=False,
    help=(
        "Save arm and waist reference/actual joints, policy actions, pelvis/torso motion, phase, "
        "command values, and the runtime-resolved reward weights to a .npz file."
    ),
)
parser.add_argument(
    "--diagnostic_stride",
    type=int,
    default=1,
    help="Record every N simulator steps with --diagnostic (default: 1).",
)
parser.add_argument(
    "--upper_body_reference_margin",
    type=float,
    default=None,
    help=(
        "Play-only counterfactual: limit each shoulder/elbow/wrist position target to this many radians "
        "from the current reference. Omit to use the checkpoint actions unchanged."
    ),
)
parser.add_argument(
    "--upper_body_constraint_group",
    type=str,
    choices=["wrists", "wrists_elbows", "upper_body"],
    default="upper_body",
    help=(
        "Joints affected by --upper_body_reference_margin: wrists; wrists_elbows; or all "
        "shoulders, elbows, and wrists (default)."
    ),
)
parser.add_argument(
    "--waist_reference_margin",
    type=float,
    default=None,
    help=(
        "Play-only counterfactual: limit waist yaw/roll/pitch position targets to this many radians "
        "from the current reference. Omit to use the checkpoint waist actions unchanged."
    ),
)
parser.add_argument(
    "--waist_roll_stiffness_scale",
    type=float,
    default=None,
    help=(
        "Play-only counterfactual: multiply waist-roll PD stiffness by this positive factor and "
        "its damping by sqrt(factor). The waist-pitch actuator remains unchanged."
    ),
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video or args_cli.dual_view:
    args_cli.enable_cameras = True
    # Allow headless video recording over SSH.
    if not hasattr(args_cli, 'headless'):
        args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import glob
import pathlib
import numpy as np
import torch

from soccer.tasks.tracking.mdp.rewards_dribbling import (
    soccer_ball_body_contact_force_magnitudes,
    soccer_ball_contact_force_magnitude,
    soccer_ball_max_link_contact_force_magnitude,
    soccer_ball_robot_contact,
)

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import soccer.tasks  # noqa: F401
from soccer.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from soccer.utils.checkpoint_loading import load_checkpoint_with_obs_expand

def motion_frame_count(motion_file: str) -> int:
    """Return the number of frames in a motion .npz file."""
    data = np.load(motion_file)
    if "joint_pos" in data:
        return int(data["joint_pos"].shape[0])
    for key in data.files:
        arr = data[key]
        if hasattr(arr, "shape") and len(arr.shape) >= 1:
            return int(arr.shape[0])
    raise ValueError(f"Could not infer frame count from {motion_file}")


def _env_step_s(env_cfg) -> float:
    """Wall-clock seconds per env step."""
    return float(env_cfg.decimation) * float(env_cfg.sim.dt)


def _disable_play_terminations(env_cfg) -> list[str]:
    """Disable failure termination terms so playback runs full clips (v1.20 behaviour)."""
    if not hasattr(env_cfg, "terminations"):
        return []

    terms = env_cfg.terminations
    disabled: list[str] = []

    for name in getattr(type(terms), "__annotations__", {}):
        if name.startswith("_"):
            continue
        setattr(terms, name, None)
        disabled.append(name)

    for name in terms.__dict__:
        if name.startswith("_") or name in disabled:
            continue
        setattr(terms, name, None)
        disabled.append(name)

    return sorted(set(disabled))


def _setup_play_episode_limit(env_cfg, motion_files: list[str], *, keep_failure_terms: bool) -> int:
    """Size episodes for playback and drop the training 10 s / 500-step cap.

    Returns the recommended total step count (single clip = its frame count;
    multi-clip sequential run = sum of frame counts).
    """
    if not motion_files:
        return 600

    step_s = _env_step_s(env_cfg)
    frame_counts = [motion_frame_count(f) for f in motion_files]
    max_frames = max(frame_counts)

    # Training uses episode_length_s=10 (500 steps). Playback needs at least one full clip.
    env_cfg.episode_length_s = max_frames * step_s + 2.0
    if keep_failure_terms:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        print("[INFO] Playback: training failure terminations active (ball_lost, ee_body_pos, …).")
    else:
        disabled_terms = _disable_play_terminations(env_cfg)
        if disabled_terms:
            print(f"[INFO] Playback: disabled terminations: {', '.join(disabled_terms)}")

    if len(motion_files) > 1 and hasattr(env_cfg.commands.motion, "sampling_strategy"):
        env_cfg.commands.motion.sampling_strategy = "sequential"

    if len(motion_files) > 1:
        return sum(frame_counts)
    return max_frames


def get_motion_files(motion_path: str) -> list[str]:
    """
    Get a list of motion files.
    
    Args:
        motion_path: File path or directory path.
        
    Returns:
        List of motion file paths.
    """
    if os.path.isfile(motion_path):
        # Single-file input.
        return [motion_path]
    elif os.path.isdir(motion_path):
        # Directory input: collect all .npz files.
        motion_files = glob.glob(os.path.join(motion_path, "*.npz"))
        if not motion_files:
            raise ValueError(f"No .npz files found in directory: {motion_path}")
        motion_files.sort()
        print(f"Found {len(motion_files)} motion files in {motion_path}")
        for file in motion_files:
            print(f"  - {os.path.basename(file)}")
        return motion_files
    else:
        raise ValueError(f"Invalid path: {motion_path}. Must be a file or directory.")


_BALL_SENSOR_NAME = "soccer_ball_contact"
_CONTACT_FORCE_THRESHOLD = 1.0
_LAST_TERM_REASON: str = "-"

_ARM_DIAGNOSTIC_JOINT_NAMES = [
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

# The diagnostic archive keeps arm and trunk arrays separate so analysis scripts
# can distinguish their control paths without relying on array slices.
_TRUNK_DIAGNOSTIC_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
_WAIST_REFERENCE_CONSTRAINT_GROUP = "waist"
_TRUNK_DIAGNOSTIC_BODY_NAMES = ["pelvis", "torso_link"]
# These match the G1 actuator ``effort_limit_sim`` values in ``soccer/robots/g1.py``.
# They make saturation visible in an offline diagnostic without depending on a
# simulator-internal actuator API.
_TRUNK_EFFORT_LIMITS = [88.0, 50.0, 50.0]

_UPPER_BODY_CONSTRAINT_JOINT_GROUPS = {
    "wrists": [
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "wrists_elbows": [
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "upper_body": _ARM_DIAGNOSTIC_JOINT_NAMES,
}

_FOOT_DIAGNOSTIC_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _update_last_termination_reason(base_env, env_idx: int = 0) -> None:
    """Track the latest failure termination from the env's termination manager."""
    global _LAST_TERM_REASON
    tm = getattr(base_env, "termination_manager", None)
    if tm is None:
        return
    active: list[str] = []
    for name in tm.active_terms:
        try:
            if tm.get_term_cfg(name).time_out:
                continue
            if bool(tm.get_term(name)[env_idx].item()):
                active.append(name)
        except Exception:
            continue
    if active:
        _LAST_TERM_REASON = ", ".join(active)


def _active_failure_termination_reason(base_env, env_idx: int = 0) -> str:
    """Return all currently active non-timeout termination terms for one env."""
    tm = getattr(base_env, "termination_manager", None)
    if tm is None:
        return ""
    active: list[str] = []
    for name in tm.active_terms:
        try:
            if tm.get_term_cfg(name).time_out:
                continue
            if bool(tm.get_term(name)[env_idx].item()):
                active.append(name)
        except Exception:
            continue
    return ", ".join(active)


def _reward_term_values(base_env, env_idx: int = 0) -> np.ndarray:
    """Read the reward manager's per-term contribution for the most recent step."""
    reward_manager = getattr(base_env, "reward_manager", None)
    values = getattr(reward_manager, "_step_reward", None)
    if values is None:
        return np.empty(0, dtype=np.float32)
    return values[env_idx].detach().cpu().numpy().copy()


def _reward_term_weights(base_env) -> np.ndarray:
    """Snapshot the final runtime reward weights in ``_term_names`` order."""
    reward_manager = getattr(base_env, "reward_manager", None)
    term_names = list(getattr(reward_manager, "_term_names", []))
    term_cfgs = list(getattr(reward_manager, "_term_cfgs", []))
    weights: list[float] = []
    for index, name in enumerate(term_names):
        try:
            term_cfg = reward_manager.get_term_cfg(name)
        except (AttributeError, KeyError):
            term_cfg = term_cfgs[index] if index < len(term_cfgs) else None
        weight = getattr(term_cfg, "weight", np.nan)
        if isinstance(weight, torch.Tensor):
            weight = weight.detach().cpu().item()
        weights.append(float(weight))
    return np.asarray(weights, dtype=np.float32)


def _resolve_base_env(env):
    """Unwrap gym / RSL-RL wrappers to the underlying Isaac Lab env."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    if hasattr(base, "unwrapped"):
        base = base.unwrapped
    return base


def _get_joint_position_action_term(base_env):
    """Return the position-action term and its action-index → robot-joint mapping."""
    action_term = base_env.action_manager.get_term("joint_pos")
    action_joint_ids = getattr(action_term, "_joint_ids", None)
    if action_joint_ids is None:
        action_joint_ids = getattr(action_term, "joint_ids", None)
    if action_joint_ids is None:
        raise RuntimeError("The joint_pos action term does not expose its controlled joint ids.")
    if isinstance(action_joint_ids, slice):
        action_joint_ids = torch.arange(base_env.scene["robot"].num_joints, device=base_env.device)
    return action_term, torch.as_tensor(action_joint_ids, dtype=torch.long, device=base_env.device)


def _action_ids_for_robot_joint_ids(action_joint_ids: torch.Tensor, robot_joint_ids) -> list[int]:
    """Map robot joint ids to their positions in the policy action vector."""
    robot_to_action = {int(robot_id): action_id for action_id, robot_id in enumerate(action_joint_ids.tolist())}
    try:
        return [robot_to_action[int(robot_id)] for robot_id in robot_joint_ids]
    except KeyError as exc:
        raise RuntimeError(
            f"joint_pos action does not control robot joint id {exc.args[0]}."
        ) from exc


def _world_quat_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    """Convert Isaac Lab scalar-first quaternions to roll/pitch/yaw in radians."""
    w, x, y, z = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), min=-1.0, max=1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    return torch.stack((roll, pitch, yaw), dim=-1)


def _create_joint_reference_constraint(env, margin: float, group: str, joint_names: list[str]) -> dict:
    """Prepare a play-only joint-target clamp around the current reference."""
    if margin < 0.0:
        raise ValueError(f"{group} reference margin must be non-negative.")

    base_env = _resolve_base_env(env)
    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene[command.cfg.asset_name]
    action_term, action_joint_ids = _get_joint_position_action_term(base_env)
    if getattr(action_term, "uses_direct_upper_body_latent", False):
        raise ValueError(
            "Play-only joint reference clamps are incompatible with the direct upper-body latent interface. "
            "Use the environment's reference envelope and turn-aware trunk limits instead."
        )
    robot_joint_ids, found_names = robot.find_joints(joint_names, preserve_order=True)
    if len(robot_joint_ids) != len(joint_names):
        raise RuntimeError(f"Could not resolve all {group} constraint joints; found {found_names}.")

    action_ids = _action_ids_for_robot_joint_ids(action_joint_ids, robot_joint_ids)

    scale = getattr(action_term, "_scale", None)
    offset = getattr(action_term, "_offset", None)
    if scale is None or offset is None:
        raise RuntimeError("The joint_pos action term does not expose scale/offset required for target clamping.")

    return {
        "group": group,
        "joint_names": np.asarray(joint_names),
        "margin": float(margin),
        "robot_joint_ids": torch.as_tensor(robot_joint_ids, dtype=torch.long, device=base_env.device),
        "action_ids": torch.as_tensor(action_ids, dtype=torch.long, device=base_env.device),
        "scale": scale,
        "offset": offset,
    }


def _create_upper_body_reference_constraint(env, margin: float, group: str) -> dict:
    """Prepare a play-only shoulder/elbow/wrist target clamp."""
    return _create_joint_reference_constraint(
        env, margin, group, _UPPER_BODY_CONSTRAINT_JOINT_GROUPS[group]
    )


def _create_waist_reference_constraint(env, margin: float) -> dict:
    """Prepare a play-only waist yaw/roll/pitch target clamp."""
    return _create_joint_reference_constraint(
        env, margin, _WAIST_REFERENCE_CONSTRAINT_GROUP, _TRUNK_DIAGNOSTIC_JOINT_NAMES
    )


def _constrain_reference_actions(env, actions: torch.Tensor, constraint: dict) -> torch.Tensor:
    """Keep each commanded upper-body target within ``q_ref ± margin``.

    This is a counterfactual evaluation intervention only: it alters actions
    after policy inference and never changes the checkpoint or training MDP.
    """
    base_env = _resolve_base_env(env)
    command = base_env.command_manager.get_term("motion")
    action_ids = constraint["action_ids"]
    robot_joint_ids = constraint["robot_joint_ids"]
    scale = constraint["scale"][:, action_ids]
    offset = constraint["offset"][:, action_ids]
    if torch.any(torch.abs(scale) < 1.0e-8):
        raise RuntimeError(f"Cannot constrain {constraint['group']} actions with a zero action scale.")

    target = actions[:, action_ids] * scale + offset
    reference = command.joint_pos[:, robot_joint_ids]
    constrained_target = torch.clamp(
        target,
        min=reference - constraint["margin"],
        max=reference + constraint["margin"],
    )
    constrained_actions = actions.clone()
    constrained_actions[:, action_ids] = (constrained_target - offset) / scale
    return constrained_actions


def _single_joint_actuator_parameter(value, joint_name: str, scale: float):
    """Return a one-joint actuator parameter, preserving scalar config values."""
    if not isinstance(value, dict):
        return float(value) * scale
    if joint_name in value:
        return {joint_name: float(value[joint_name]) * scale}
    if len(value) == 1:
        return {joint_name: float(next(iter(value.values()))) * scale}
    raise ValueError(f"Cannot select {joint_name} from multi-joint actuator parameter {value}.")


def _apply_play_waist_roll_stiffness_scale(env_cfg, scale: float | None) -> float:
    """Split the waist actuator and strengthen only waist roll for one playback."""
    if scale is not None and scale <= 0.0:
        raise ValueError("--waist_roll_stiffness_scale must be positive.")

    actuators = dict(env_cfg.scene.robot.actuators)
    waist_cfg = actuators.pop("waist", None)
    if waist_cfg is None:
        if "waist_roll_control" in actuators:
            if scale is not None:
                raise ValueError(
                    "The control task already uses the training waist-roll PD scale (2.0); "
                    "do not also pass --waist_roll_stiffness_scale."
                )
            print("[INFO] Control task uses the training waist-roll PD scale: stiffness x2.000, damping x1.414.")
            return 2.0
        raise RuntimeError("Play-only waist-roll override requires the robot 'waist' actuator config.")
    if scale is None:
        return 1.0

    damping_scale = float(scale) ** 0.5
    actuators["waist_roll_play"] = waist_cfg.replace(
        joint_names_expr=["waist_roll_joint"],
        stiffness=_single_joint_actuator_parameter(waist_cfg.stiffness, "waist_roll_joint", scale),
        damping=_single_joint_actuator_parameter(waist_cfg.damping, "waist_roll_joint", damping_scale),
    )
    actuators["waist_pitch_play"] = waist_cfg.replace(
        joint_names_expr=["waist_pitch_joint"],
        stiffness=_single_joint_actuator_parameter(waist_cfg.stiffness, "waist_pitch_joint", 1.0),
        damping=_single_joint_actuator_parameter(waist_cfg.damping, "waist_pitch_joint", 1.0),
    )
    env_cfg.scene.robot.actuators = actuators
    print(
        "[INFO] Play-only waist-roll PD override: "
        f"stiffness x{scale:.3f}, damping x{damping_scale:.3f}; waist pitch unchanged."
    )
    return float(scale)


def _create_diagnostic(
    env,
    log_dir: str,
    stride: int,
    constraints: list[dict] | None = None,
    waist_roll_stiffness_scale: float = 1.0,
) -> dict:
    """Prepare a per-step arm, waist, and trunk-motion trace for one playback env."""
    if stride <= 0:
        raise ValueError("--diagnostic_stride must be positive.")
    constraints = [] if constraints is None else constraints

    base_env = _resolve_base_env(env)
    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene[command.cfg.asset_name]
    joint_ids, found_names = robot.find_joints(_ARM_DIAGNOSTIC_JOINT_NAMES, preserve_order=True)
    if len(joint_ids) != len(_ARM_DIAGNOSTIC_JOINT_NAMES):
        raise RuntimeError(
            "Could not resolve all arm joints for the diagnostic. "
            f"Expected {_ARM_DIAGNOSTIC_JOINT_NAMES}, found {found_names}."
        )
    trunk_joint_ids, trunk_found_names = robot.find_joints(
        _TRUNK_DIAGNOSTIC_JOINT_NAMES, preserve_order=True
    )
    if len(trunk_joint_ids) != len(_TRUNK_DIAGNOSTIC_JOINT_NAMES):
        raise RuntimeError(
            "Could not resolve all waist diagnostic joints. "
            f"Expected {_TRUNK_DIAGNOSTIC_JOINT_NAMES}, found {trunk_found_names}."
        )
    trunk_body_ids = [robot.body_names.index(name) for name in _TRUNK_DIAGNOSTIC_BODY_NAMES]
    try:
        trunk_reference_body_ids = [command.cfg.body_names.index(name) for name in _TRUNK_DIAGNOSTIC_BODY_NAMES]
    except ValueError as exc:
        raise RuntimeError(
            f"Motion command does not expose every trunk diagnostic body: {_TRUNK_DIAGNOSTIC_BODY_NAMES}."
        ) from exc
    action_term, action_joint_ids = _get_joint_position_action_term(base_env)
    arm_action_ids = _action_ids_for_robot_joint_ids(action_joint_ids, joint_ids)
    direct_upper_latent = bool(getattr(action_term, "uses_direct_upper_body_latent", False))
    if direct_upper_latent:
        trunk_action_ids = action_term.policy_action_ids_for_robot_joint_ids(trunk_joint_ids)
    else:
        trunk_action_ids = _action_ids_for_robot_joint_ids(action_joint_ids, trunk_joint_ids)

    output_dir = os.path.join(log_dir, "diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"diagnostic_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    )
    reward_manager = getattr(base_env, "reward_manager", None)
    reward_term_names = np.asarray(getattr(reward_manager, "_term_names", []))
    reward_term_weights = _reward_term_weights(base_env)
    reward_step_dt = float(getattr(base_env, "step_dt", _env_step_s(base_env.cfg)))
    _body_force, ball_contact_body_names, ball_contact_filter_available = (
        soccer_ball_body_contact_force_magnitudes(base_env, _BALL_SENSOR_NAME)
    )
    constraint_groups = np.asarray([constraint["group"] for constraint in constraints])
    constraint_margins = np.asarray([constraint["margin"] for constraint in constraints], dtype=np.float32)
    constraint_joint_names = (
        np.concatenate([constraint["joint_names"] for constraint in constraints])
        if constraints else np.asarray([])
    )
    return {
        "path": output_path,
        "stride": int(stride),
        "joint_ids": torch.as_tensor(joint_ids, dtype=torch.long, device=base_env.device),
        "joint_names": np.asarray(_ARM_DIAGNOSTIC_JOINT_NAMES),
        "action_ids": torch.as_tensor(arm_action_ids, dtype=torch.long, device=base_env.device),
        "direct_upper_body_latent": direct_upper_latent,
        "trunk_joint_ids": torch.as_tensor(trunk_joint_ids, dtype=torch.long, device=base_env.device),
        "trunk_joint_names": np.asarray(_TRUNK_DIAGNOSTIC_JOINT_NAMES),
        "trunk_action_ids": torch.as_tensor(trunk_action_ids, dtype=torch.long, device=base_env.device),
        "trunk_body_ids": torch.as_tensor(trunk_body_ids, dtype=torch.long, device=base_env.device),
        "trunk_reference_body_ids": torch.as_tensor(
            trunk_reference_body_ids, dtype=torch.long, device=base_env.device
        ),
        "trunk_body_names": np.asarray(_TRUNK_DIAGNOSTIC_BODY_NAMES),
        "reward_term_names": reward_term_names,
        "reward_term_weights": reward_term_weights,
        "reward_term_step_weights": reward_term_weights * reward_step_dt,
        "reward_step_dt": reward_step_dt,
        "task_state_names": np.asarray(["idle", "dribble", "stop"]),
        "ball_contact_body_names": np.asarray(ball_contact_body_names),
        "ball_contact_filter_available": bool(ball_contact_filter_available),
        "constraint_group": "none" if not constraints else "+".join(constraint_groups.tolist()),
        "constraint_margin": float(constraint_margins[0]) if len(constraints) == 1 else np.nan,
        "constraint_joint_names": constraint_joint_names,
        "constraint_groups": constraint_groups,
        "constraint_margins": constraint_margins,
        "waist_roll_stiffness_scale": float(waist_roll_stiffness_scale),
        "waist_roll_damping_scale": float(waist_roll_stiffness_scale) ** 0.5,
        "step": [],
        "motion_idx": [],
        "style_phase": [],
        "segment_idx": [],
        "task_state": [],
        "command_heading": [],
        "effective_command_speed": [],
        "effective_command_heading": [],
        "pelvis_yaw": [],
        "reference_joint_pos": [],
        "reference_joint_vel": [],
        "actual_joint_pos": [],
        "actual_joint_vel": [],
        "policy_action": [],
        "applied_action": [],
        "upper_policy_latent": [],
        "trunk_reference_joint_pos": [],
        "trunk_reference_joint_vel": [],
        "trunk_actual_joint_pos": [],
        "trunk_actual_joint_vel": [],
        "trunk_policy_action": [],
        "trunk_applied_action": [],
        "trunk_processed_joint_target": [],
        "trunk_target_minus_reference": [],
        "trunk_post_step_actual_joint_pos": [],
        "trunk_post_step_target_error": [],
        "trunk_soft_joint_pos_limits": [],
        "trunk_actual_limit_margin": [],
        "trunk_target_limit_margin": [],
        "trunk_computed_torque": [],
        "trunk_applied_torque": [],
        "trunk_effort_limit": [],
        "trunk_computed_effort_utilization": [],
        "trunk_effort_utilization": [],
        "trunk_effort_saturated": [],
        "pelvis_rpy": [],
        "torso_rpy": [],
        "torso_minus_pelvis_rpy": [],
        "reference_pelvis_rpy": [],
        "reference_torso_rpy": [],
        "reference_torso_minus_pelvis_rpy": [],
        "torso_minus_pelvis_rpy_error": [],
        "pelvis_ang_vel_w": [],
        "torso_ang_vel_w": [],
        "torso_minus_pelvis_ang_vel_w": [],
        "ball_pelvis_xy_distance": [],
        "ball_command_forward_offset": [],
        "ball_command_lateral_offset": [],
        "ball_position_z": [],
        "ball_vertical_speed": [],
        "ball_contact": [],
        "ball_net_contact": [],
        "ball_contact_force": [],
        "ball_link_contact": [],
        "ball_max_link_contact_force": [],
        "ball_contact_steps_since_link": [],
        "ball_contact_body_force_magnitudes": [],
        "ball_contact_body_index": [],
        "ball_undesired_body_contact": [],
        "ball_contact_duty_ema": [],
        "ball_contact_duty_penalty": [],
        "ball_too_close_penalty": [],
        "cg_label_available": [],
        "cg_expected_contact": [],
        "cg_expected_foot": [],
        "cg_contact_window_active": [],
        "cg_contact_window_hit": [],
        "cg_contact_event_score": [],
        "cg_premature_contact": [],
        "cg_missing_contact": [],
        "cg_wrong_foot_contact": [],
        "ball_xy_speed": [],
        "ball_command_forward_speed": [],
        "pelvis_xy_speed": [],
        "command_target_speed": [],
        "command_target_heading": [],
        "foot_reference_position_error": [],
        "heading_error": [],
        "no_contact_count": [],
        "no_contact_task_active": [],
        "no_contact_recovery_active": [],
        "no_contact_proximity_recovery_active": [],
        "no_contact_relative_speed": [],
        "idle_active": [],
        "idle_pelvis_speed": [],
        "idle_pelvis_angular_speed": [],
        "stop_active": [],
        "stop_settled": [],
        "stop_success": [],
        "stop_settle_elapsed_s": [],
        "stop_pelvis_speed": [],
        "stop_pelvis_angular_speed": [],
        "stop_ball_speed": [],
        "stop_forward_offset": [],
        "stop_lateral_offset": [],
        "stop_position_score": [],
        "stop_speed_score": [],
        "manifold_raw_upper_target": [],
        "manifold_reference_upper_target": [],
        "manifold_constrained_upper_target": [],
        "manifold_projected_upper_target": [],
        "manifold_latent": [],
        "manifold_projection_error": [],
        "manifold_projection_error_after_reference_constraint": [],
        "manifold_nullspace_residual": [],
        "manifold_latent_clip_fraction": [],
        "manifold_reference_overflow": [],
        "manifold_reference_clamp_fraction": [],
        "trunk_pitch_raw_target": [],
        "trunk_pitch_reference_target": [],
        "trunk_pitch_soft_target": [],
        "trunk_pitch_filtered_target": [],
        "trunk_pitch_reference_overflow": [],
        "trunk_pitch_turn_relaxation": [],
        "trunk_pitch_active_lower_deviation": [],
        "trunk_pitch_active_upper_deviation": [],
        "trunk_pitch_active_cutoff_frequency_hz": [],
        "step_reward": [],
        "reward_terms": [],
        "done": [],
        "termination_reason": [],
    }


def _append_diagnostic(
    diagnostic: dict,
    env,
    policy_actions: torch.Tensor,
    applied_actions: torch.Tensor,
    step: int,
) -> bool:
    """Record the pre-step reference/state plus policy and applied actions for env 0."""
    if step % diagnostic["stride"] != 0:
        return False

    base_env = _resolve_base_env(env)
    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene[command.cfg.asset_name]
    ids = diagnostic["joint_ids"]
    action_ids = diagnostic["action_ids"]
    trunk_ids = diagnostic["trunk_joint_ids"]
    trunk_action_ids = diagnostic["trunk_action_ids"]
    trunk_body_ids = diagnostic["trunk_body_ids"]
    trunk_reference_body_ids = diagnostic["trunk_reference_body_ids"]

    def _cpu(value: torch.Tensor) -> np.ndarray:
        return value[0].detach().cpu().numpy().copy()

    diagnostic["step"].append(int(step))
    diagnostic["motion_idx"].append(int(command.motion_idx[0].item()))
    diagnostic["style_phase"].append(int(command.style_phase_steps[0].item()))
    diagnostic["segment_idx"].append(int(command._locomotion_segment_idx[0].item()))
    task_state = getattr(command, "locomotion_task_state", None)
    diagnostic["task_state"].append(1 if task_state is None else int(task_state[0].item()))
    diagnostic["command_heading"].append(float(command.locomotion_cmd_heading[0].item()))
    target_speed = getattr(command, "locomotion_cmd_target_speed", command.locomotion_cmd_speed)
    target_heading = getattr(command, "locomotion_cmd_target_heading", command.locomotion_cmd_heading)
    diagnostic["command_target_speed"].append(float(target_speed[0].item()))
    diagnostic["command_target_heading"].append(float(target_heading[0].item()))
    diagnostic["effective_command_speed"].append(float(command.locomotion_cmd_speed[0].item()))
    diagnostic["effective_command_heading"].append(float(command.locomotion_cmd_heading[0].item()))

    pelvis_id = robot.body_names.index("pelvis")
    pelvis_quat = robot.data.body_quat_w[0, pelvis_id]
    diagnostic["pelvis_yaw"].append(float(torch.atan2(
        2.0 * (pelvis_quat[0] * pelvis_quat[3] + pelvis_quat[1] * pelvis_quat[2]),
        1.0 - 2.0 * (pelvis_quat[2].square() + pelvis_quat[3].square()),
    ).item()))
    diagnostic["reference_joint_pos"].append(_cpu(command.joint_pos[:, ids]))
    diagnostic["reference_joint_vel"].append(_cpu(command.joint_vel[:, ids]))
    diagnostic["actual_joint_pos"].append(_cpu(robot.data.joint_pos[:, ids]))
    diagnostic["actual_joint_vel"].append(_cpu(robot.data.joint_vel[:, ids]))
    action_term = base_env.action_manager.get_term("joint_pos")
    if diagnostic["direct_upper_body_latent"]:
        # A direct-latent policy has no per-arm input actions.  Store the
        # decoded pre-envelope target alongside the physically effective arm
        # action so the diagnostic remains comparable with legacy traces.
        diagnostic["policy_action"].append(_cpu(action_term.manifold_raw_upper_target))
        diagnostic["applied_action"].append(_cpu(action_term.effective_raw_actions[:, action_ids]))
        diagnostic["upper_policy_latent"].append(_cpu(action_term.manifold_policy_latent))
    else:
        diagnostic["policy_action"].append(_cpu(policy_actions[:, action_ids]))
        diagnostic["applied_action"].append(_cpu(applied_actions[:, action_ids]))
        diagnostic["upper_policy_latent"].append(np.empty(0, dtype=np.float32))
    diagnostic["trunk_reference_joint_pos"].append(_cpu(command.joint_pos[:, trunk_ids]))
    diagnostic["trunk_reference_joint_vel"].append(_cpu(command.joint_vel[:, trunk_ids]))
    diagnostic["trunk_actual_joint_pos"].append(_cpu(robot.data.joint_pos[:, trunk_ids]))
    diagnostic["trunk_actual_joint_vel"].append(_cpu(robot.data.joint_vel[:, trunk_ids]))
    diagnostic["trunk_policy_action"].append(_cpu(policy_actions[:, trunk_action_ids]))
    diagnostic["trunk_applied_action"].append(_cpu(applied_actions[:, trunk_action_ids]))

    trunk_quat = robot.data.body_quat_w[0, trunk_body_ids]
    trunk_rpy = _world_quat_to_rpy(trunk_quat)
    torso_minus_pelvis_rpy = torch.atan2(
        torch.sin(trunk_rpy[1] - trunk_rpy[0]),
        torch.cos(trunk_rpy[1] - trunk_rpy[0]),
    )
    reference_trunk_rpy = _world_quat_to_rpy(
        command.body_quat_relative_w[0, trunk_reference_body_ids]
    )
    reference_torso_minus_pelvis_rpy = torch.atan2(
        torch.sin(reference_trunk_rpy[1] - reference_trunk_rpy[0]),
        torch.cos(reference_trunk_rpy[1] - reference_trunk_rpy[0]),
    )
    torso_minus_pelvis_rpy_error = torch.atan2(
        torch.sin(torso_minus_pelvis_rpy - reference_torso_minus_pelvis_rpy),
        torch.cos(torso_minus_pelvis_rpy - reference_torso_minus_pelvis_rpy),
    )
    trunk_ang_vel_w = robot.data.body_ang_vel_w[0, trunk_body_ids]
    diagnostic["pelvis_rpy"].append(_cpu(trunk_rpy[None, 0]))
    diagnostic["torso_rpy"].append(_cpu(trunk_rpy[None, 1]))
    diagnostic["torso_minus_pelvis_rpy"].append(_cpu(torso_minus_pelvis_rpy[None]))
    diagnostic["reference_pelvis_rpy"].append(_cpu(reference_trunk_rpy[None, 0]))
    diagnostic["reference_torso_rpy"].append(_cpu(reference_trunk_rpy[None, 1]))
    diagnostic["reference_torso_minus_pelvis_rpy"].append(
        _cpu(reference_torso_minus_pelvis_rpy[None])
    )
    diagnostic["torso_minus_pelvis_rpy_error"].append(_cpu(torso_minus_pelvis_rpy_error[None]))
    diagnostic["pelvis_ang_vel_w"].append(_cpu(trunk_ang_vel_w[None, 0]))
    diagnostic["torso_ang_vel_w"].append(_cpu(trunk_ang_vel_w[None, 1]))
    diagnostic["torso_minus_pelvis_ang_vel_w"].append(_cpu((trunk_ang_vel_w[1] - trunk_ang_vel_w[0])[None]))
    soccer_ball = base_env.scene["soccer_ball"]
    ball_delta_xy = soccer_ball.data.root_pos_w[0, :2] - robot.data.body_pos_w[0, pelvis_id, :2]
    diagnostic["ball_pelvis_xy_distance"].append(float(torch.norm(ball_delta_xy).item()))
    contact_force = soccer_ball_contact_force_magnitude(base_env, _BALL_SENSOR_NAME)[0]
    net_contact = bool(contact_force.item() > _CONTACT_FORCE_THRESHOLD)
    robot_contact_tensor = soccer_ball_robot_contact(
        base_env,
        _BALL_SENSOR_NAME,
        contact_force_threshold=_CONTACT_FORCE_THRESHOLD,
        hold_steps=2,
    )
    robot_contact = bool(robot_contact_tensor[0].item())
    diagnostic["ball_contact"].append(robot_contact)
    diagnostic["ball_net_contact"].append(net_contact)
    diagnostic["ball_contact_force"].append(float(contact_force.item()))
    link_contact_force = soccer_ball_max_link_contact_force_magnitude(
        base_env, _BALL_SENSOR_NAME
    )[0]
    link_contact = bool(link_contact_force.item() > _CONTACT_FORCE_THRESHOLD)
    diagnostic["ball_link_contact"].append(link_contact)
    diagnostic["ball_max_link_contact_force"].append(float(link_contact_force.item()))
    steps_since_link = getattr(base_env, "_soccer_ball_steps_since_link_contact", None)
    diagnostic["ball_contact_steps_since_link"].append(
        -1 if steps_since_link is None else int(steps_since_link[0].item())
    )
    ball_vel_xy = soccer_ball.data.root_lin_vel_w[0, :2]
    diagnostic["ball_xy_speed"].append(float(torch.norm(ball_vel_xy).item()))
    command_dir = torch.stack((
        torch.cos(command.locomotion_cmd_heading[0]),
        torch.sin(command.locomotion_cmd_heading[0]),
    ))
    command_lateral_dir = torch.stack((-command_dir[1], command_dir[0]))
    forward_offset = torch.dot(ball_delta_xy, command_dir)
    lateral_offset = torch.dot(ball_delta_xy, command_lateral_dir)
    diagnostic["ball_command_forward_offset"].append(float(forward_offset.item()))
    diagnostic["ball_command_lateral_offset"].append(float(lateral_offset.item()))
    diagnostic["ball_position_z"].append(float(soccer_ball.data.root_pos_w[0, 2].item()))
    diagnostic["ball_vertical_speed"].append(float(soccer_ball.data.root_lin_vel_w[0, 2].item()))
    diagnostic["ball_command_forward_speed"].append(float(torch.dot(ball_vel_xy, command_dir).item()))
    diagnostic["pelvis_xy_speed"].append(float(torch.norm(robot.data.body_lin_vel_w[0, pelvis_id, :2]).item()))

    body_force_magnitudes, body_names, filtered_available = (
        soccer_ball_body_contact_force_magnitudes(
            base_env,
            _BALL_SENSOR_NAME,
            tuple(diagnostic["ball_contact_body_names"].tolist()),
        )
    )
    body_force_env0 = body_force_magnitudes[0]
    if filtered_available and body_force_env0.numel() > 0:
        actual_body_force, actual_body_index_tensor = body_force_env0.max(dim=0)
        actual_body_index = (
            int(actual_body_index_tensor.item())
            if actual_body_force.item() > _CONTACT_FORCE_THRESHOLD
            else -1
        )
    else:
        actual_body_index = -1
    diagnostic["ball_contact_body_force_magnitudes"].append(
        body_force_env0.detach().cpu().numpy().copy()
    )
    diagnostic["ball_contact_body_index"].append(actual_body_index)
    ankle_names = {"left_ankle_roll_link", "right_ankle_roll_link"}
    actual_body_name = body_names[actual_body_index] if actual_body_index >= 0 else ""
    diagnostic["ball_undesired_body_contact"].append(
        bool(link_contact and actual_body_name and actual_body_name not in ankle_names)
    )

    duty_ema = getattr(base_env, "_dribbling_contact_duty_ema", None)
    duty_penalty = getattr(base_env, "_dribbling_contact_duty_penalty", None)
    too_close_penalty = getattr(base_env, "_dribbling_ball_too_close_penalty", None)
    diagnostic["ball_contact_duty_ema"].append(
        np.nan if duty_ema is None else float(duty_ema[0].item())
    )
    diagnostic["ball_contact_duty_penalty"].append(
        np.nan if duty_penalty is None else float(duty_penalty[0].item())
    )
    diagnostic["ball_too_close_penalty"].append(
        np.nan if too_close_penalty is None else float(too_close_penalty[0].item())
    )

    cg_labeled = bool(
        hasattr(command, "motion_has_dribble_cg_label")
        and command.motion_has_dribble_cg_label[0].item()
    )
    cg_expected_contact = bool(command.dribble_cg_contact_ref[0].item()) if cg_labeled else False
    cg_expected_foot = int(command.dribble_cg_foot_ref[0].item()) if cg_labeled else -1
    expected_body_name = (
        "left_ankle_roll_link" if cg_expected_foot == 0
        else "right_ankle_roll_link" if cg_expected_foot == 1
        else ""
    )
    diagnostic["cg_label_available"].append(cg_labeled)
    diagnostic["cg_expected_contact"].append(cg_expected_contact)
    diagnostic["cg_expected_foot"].append(cg_expected_foot)
    cg_window_tensor = getattr(base_env, "_dribbling_cg_contact_window_active", None)
    cg_window_hit_tensor = getattr(base_env, "_dribbling_cg_contact_window_hit", None)
    cg_premature_tensor = getattr(base_env, "_dribbling_cg_premature_contact", None)
    cg_event_score_tensor = getattr(base_env, "_dribbling_cg_contact_event_score", None)
    cg_window_active = (
        cg_expected_contact
        if cg_window_tensor is None
        else bool(cg_window_tensor[0].item())
    )
    cg_window_hit = (
        cg_window_active and robot_contact
        if cg_window_hit_tensor is None
        else bool(cg_window_hit_tensor[0].item())
    )
    cg_premature = (
        cg_labeled and link_contact and not cg_window_active
        if cg_premature_tensor is None
        else bool(cg_premature_tensor[0].item())
    )
    cg_event_score = (
        float(cg_window_hit) - float(cg_premature)
        if cg_event_score_tensor is None
        else float(cg_event_score_tensor[0].item())
    )
    diagnostic["cg_contact_window_active"].append(cg_window_active)
    diagnostic["cg_contact_window_hit"].append(cg_window_hit)
    diagnostic["cg_contact_event_score"].append(cg_event_score)
    diagnostic["cg_premature_contact"].append(cg_premature)
    diagnostic["cg_missing_contact"].append(cg_labeled and cg_window_active and not cg_window_hit)
    diagnostic["cg_wrong_foot_contact"].append(
        bool(
            cg_labeled
            and cg_window_active
            and link_contact
            and filtered_available
            and actual_body_index >= 0
            and expected_body_name
            and actual_body_name != expected_body_name
        )
    )
    foot_ref_ids = [command.cfg.body_names.index(name) for name in _FOOT_DIAGNOSTIC_BODY_NAMES]
    foot_error = torch.norm(
        command.robot_body_pos_w[0, foot_ref_ids] - command.body_pos_relative_w[0, foot_ref_ids], dim=-1
    ).mean()
    diagnostic["foot_reference_position_error"].append(float(foot_error.item()))
    heading_error = torch.atan2(
        torch.sin(command.locomotion_cmd_heading[0] - diagnostic["pelvis_yaw"][-1]),
        torch.cos(command.locomotion_cmd_heading[0] - diagnostic["pelvis_yaw"][-1]),
    )
    diagnostic["heading_error"].append(float(heading_error.item()))
    no_contact_count = getattr(base_env, "_dribbling_no_contact_count", None)
    diagnostic["no_contact_count"].append(
        np.nan if no_contact_count is None else float(no_contact_count[0].item())
    )
    no_contact_task_active = getattr(base_env, "_dribbling_no_contact_task_active", None)
    diagnostic["no_contact_task_active"].append(
        False if no_contact_task_active is None else bool(no_contact_task_active[0].item())
    )
    no_contact_recovery = getattr(base_env, "_dribbling_no_contact_recovery_active", None)
    diagnostic["no_contact_recovery_active"].append(
        False if no_contact_recovery is None else bool(no_contact_recovery[0].item())
    )
    proximity_recovery = getattr(base_env, "_dribbling_no_contact_proximity_recovery_active", None)
    diagnostic["no_contact_proximity_recovery_active"].append(
        False if proximity_recovery is None else bool(proximity_recovery[0].item())
    )
    relative_speed = getattr(base_env, "_dribbling_no_contact_relative_speed", None)
    diagnostic["no_contact_relative_speed"].append(
        np.nan if relative_speed is None else float(relative_speed[0].item())
    )
    for key, attr in (
        ("idle_active", "_dribbling_idle_active"),
        ("idle_pelvis_speed", "_dribbling_idle_pelvis_speed"),
        ("idle_pelvis_angular_speed", "_dribbling_idle_pelvis_angular_speed"),
        ("stop_active", "_dribbling_stop_active"),
        ("stop_settled", "_dribbling_stop_settled"),
        ("stop_success", "_dribbling_stop_success"),
        ("stop_settle_elapsed_s", "_dribbling_stop_settle_elapsed_s"),
        ("stop_pelvis_speed", "_dribbling_stop_pelvis_speed"),
        ("stop_pelvis_angular_speed", "_dribbling_stop_pelvis_angular_speed"),
        ("stop_ball_speed", "_dribbling_stop_ball_speed"),
        ("stop_forward_offset", "_dribbling_stop_forward_offset"),
        ("stop_lateral_offset", "_dribbling_stop_lateral_offset"),
        ("stop_position_score", "_dribbling_stop_position_score"),
        ("stop_speed_score", "_dribbling_stop_speed_score"),
    ):
        value = getattr(base_env, attr, None)
        if value is None:
            diagnostic[key].append(False if key.endswith(("active", "settled", "success")) else np.nan)
        elif key.endswith(("active", "settled", "success")):
            diagnostic[key].append(bool(value[0].item()))
        else:
            diagnostic[key].append(float(value[0].item()))
    # Filled with the reward returned by the immediately following env.step().
    diagnostic["step_reward"].append(np.nan)
    return True


def _append_upper_body_manifold_diagnostic(diagnostic: dict, env) -> None:
    """Record post-step arm projection and trunk target/actuator telemetry."""
    base_env = _resolve_base_env(env)
    action_term = base_env.action_manager.get_term("joint_pos")

    def _env0(name: str, fallback_shape: tuple[int, ...]) -> np.ndarray:
        value = getattr(action_term, name, None)
        if not isinstance(value, torch.Tensor):
            return np.full(fallback_shape, np.nan, dtype=np.float32)
        return value[0].detach().cpu().numpy().copy()

    upper_dim = len(_ARM_DIAGNOSTIC_JOINT_NAMES)
    diagnostic["manifold_raw_upper_target"].append(
        _env0("manifold_raw_upper_target", (upper_dim,))
    )
    diagnostic["manifold_reference_upper_target"].append(
        _env0("manifold_reference_upper_target", (upper_dim,))
    )
    diagnostic["manifold_constrained_upper_target"].append(
        _env0("manifold_constrained_upper_target", (upper_dim,))
    )
    diagnostic["manifold_projected_upper_target"].append(
        _env0("manifold_projected_upper_target", (upper_dim,))
    )
    diagnostic["manifold_latent"].append(_env0("manifold_latent", (0,)))
    diagnostic["manifold_projection_error"].append(
        float(_env0("manifold_projection_error", ()).item())
    )
    diagnostic["manifold_projection_error_after_reference_constraint"].append(
        float(_env0("manifold_projection_error_after_reference_constraint", ()).item())
    )
    diagnostic["manifold_nullspace_residual"].append(
        float(_env0("manifold_nullspace_residual", ()).item())
    )
    diagnostic["manifold_latent_clip_fraction"].append(
        float(_env0("manifold_latent_clip_fraction", ()).item())
    )
    diagnostic["manifold_reference_overflow"].append(
        _env0("manifold_reference_overflow", (upper_dim,))
    )
    diagnostic["manifold_reference_clamp_fraction"].append(
        float(_env0("manifold_reference_clamp_fraction", ()).item())
    )
    diagnostic["trunk_pitch_raw_target"].append(
        float(_env0("trunk_pitch_raw_target", ()).item())
    )
    diagnostic["trunk_pitch_reference_target"].append(
        float(_env0("trunk_pitch_reference_target", ()).item())
    )
    diagnostic["trunk_pitch_soft_target"].append(
        float(_env0("trunk_pitch_soft_target", ()).item())
    )
    diagnostic["trunk_pitch_filtered_target"].append(
        float(_env0("trunk_pitch_filtered_target", ()).item())
    )
    diagnostic["trunk_pitch_reference_overflow"].append(
        float(_env0("trunk_pitch_reference_overflow", ()).item())
    )
    diagnostic["trunk_pitch_turn_relaxation"].append(
        float(_env0("trunk_pitch_turn_relaxation", ()).item())
    )
    diagnostic["trunk_pitch_active_lower_deviation"].append(
        float(_env0("trunk_pitch_active_lower_deviation", ()).item())
    )
    diagnostic["trunk_pitch_active_upper_deviation"].append(
        float(_env0("trunk_pitch_active_upper_deviation", ()).item())
    )
    diagnostic["trunk_pitch_active_cutoff_frequency_hz"].append(
        float(_env0("trunk_pitch_active_cutoff_frequency_hz", ()).item())
    )

    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene[command.cfg.asset_name]
    trunk_ids = diagnostic["trunk_joint_ids"]
    trunk_action_ids = diagnostic["trunk_action_ids"]
    processed_actions = getattr(action_term, "processed_actions", None)
    if processed_actions is None:
        processed_actions = getattr(action_term, "_processed_actions", None)

    if isinstance(processed_actions, torch.Tensor):
        trunk_target = processed_actions[0, trunk_action_ids]
    else:
        trunk_target = torch.full(
            (len(_TRUNK_DIAGNOSTIC_JOINT_NAMES),),
            float("nan"),
            dtype=robot.data.joint_pos.dtype,
            device=base_env.device,
        )
    trunk_post_step_pos = robot.data.joint_pos[0, trunk_ids]
    trunk_reference = command.joint_pos[0, trunk_ids]
    soft_limits = robot.data.soft_joint_pos_limits[0, trunk_ids]
    trunk_actual_limit_margin = torch.minimum(
        trunk_post_step_pos - soft_limits[:, 0], soft_limits[:, 1] - trunk_post_step_pos
    )
    trunk_target_limit_margin = torch.minimum(
        trunk_target - soft_limits[:, 0], soft_limits[:, 1] - trunk_target
    )
    computed_torque = getattr(robot.data, "computed_torque", None)
    if isinstance(computed_torque, torch.Tensor):
        trunk_computed_torque = computed_torque[0, trunk_ids]
    else:
        trunk_computed_torque = torch.full_like(trunk_target, float("nan"))
    applied_torque = getattr(robot.data, "applied_torque", None)
    if isinstance(applied_torque, torch.Tensor):
        trunk_applied_torque = applied_torque[0, trunk_ids]
    else:
        trunk_applied_torque = torch.full_like(trunk_target, float("nan"))
    effort_limit = torch.as_tensor(_TRUNK_EFFORT_LIMITS, dtype=trunk_target.dtype, device=base_env.device)
    computed_effort_utilization = torch.abs(trunk_computed_torque) / effort_limit
    effort_utilization = torch.abs(trunk_applied_torque) / effort_limit

    def _trunk_cpu(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().copy()

    diagnostic["trunk_processed_joint_target"].append(_trunk_cpu(trunk_target))
    diagnostic["trunk_target_minus_reference"].append(_trunk_cpu(trunk_target - trunk_reference))
    diagnostic["trunk_post_step_actual_joint_pos"].append(_trunk_cpu(trunk_post_step_pos))
    diagnostic["trunk_post_step_target_error"].append(_trunk_cpu(trunk_post_step_pos - trunk_target))
    diagnostic["trunk_soft_joint_pos_limits"].append(_trunk_cpu(soft_limits))
    diagnostic["trunk_actual_limit_margin"].append(_trunk_cpu(trunk_actual_limit_margin))
    diagnostic["trunk_target_limit_margin"].append(_trunk_cpu(trunk_target_limit_margin))
    diagnostic["trunk_computed_torque"].append(_trunk_cpu(trunk_computed_torque))
    diagnostic["trunk_applied_torque"].append(_trunk_cpu(trunk_applied_torque))
    diagnostic["trunk_effort_limit"].append(_trunk_cpu(effort_limit))
    diagnostic["trunk_computed_effort_utilization"].append(_trunk_cpu(computed_effort_utilization))
    diagnostic["trunk_effort_utilization"].append(_trunk_cpu(effort_utilization))
    diagnostic["trunk_effort_saturated"].append(_trunk_cpu(effort_utilization >= 0.98))


def _save_diagnostic(diagnostic: dict) -> None:
    """Persist the trace in a self-describing NumPy archive."""
    if not diagnostic["step"]:
        print("[WARN] Diagnostic requested but no samples were recorded.")
        return
    metadata_keys = {
        "path", "stride", "joint_ids", "joint_names", "action_ids", "reward_term_names",
        "reward_term_weights", "reward_term_step_weights", "reward_step_dt", "task_state_names",
        "ball_contact_body_names", "ball_contact_filter_available",
        "trunk_joint_ids", "trunk_joint_names", "trunk_action_ids", "trunk_body_ids",
        "trunk_reference_body_ids", "trunk_body_names",
        "constraint_group", "constraint_margin", "constraint_joint_names", "constraint_groups",
        "constraint_margins", "waist_roll_stiffness_scale", "waist_roll_damping_scale",
        "direct_upper_body_latent",
    }
    arrays = {
        key: np.asarray(value)
        for key, value in diagnostic.items()
        if key not in metadata_keys
    }
    arrays["joint_names"] = diagnostic["joint_names"]
    arrays["trunk_joint_names"] = diagnostic["trunk_joint_names"]
    arrays["trunk_body_names"] = diagnostic["trunk_body_names"]
    arrays["reward_term_names"] = diagnostic["reward_term_names"]
    arrays["reward_term_weights"] = diagnostic["reward_term_weights"]
    arrays["reward_term_step_weights"] = diagnostic["reward_term_step_weights"]
    arrays["reward_step_dt"] = np.asarray(diagnostic["reward_step_dt"])
    arrays["task_state_names"] = diagnostic["task_state_names"]
    arrays["ball_contact_body_names"] = diagnostic["ball_contact_body_names"]
    arrays["ball_contact_filter_available"] = np.asarray(
        diagnostic["ball_contact_filter_available"]
    )
    arrays["constraint_group"] = np.asarray(diagnostic["constraint_group"])
    arrays["constraint_margin"] = np.asarray(diagnostic["constraint_margin"])
    arrays["constraint_joint_names"] = diagnostic["constraint_joint_names"]
    arrays["constraint_groups"] = diagnostic["constraint_groups"]
    arrays["constraint_margins"] = diagnostic["constraint_margins"]
    arrays["waist_roll_stiffness_scale"] = np.asarray(diagnostic["waist_roll_stiffness_scale"])
    arrays["waist_roll_damping_scale"] = np.asarray(diagnostic["waist_roll_damping_scale"])
    arrays["direct_upper_body_latent"] = np.asarray(diagnostic["direct_upper_body_latent"])
    np.savez_compressed(diagnostic["path"], **arrays)
    contact_rate = float(np.mean(arrays["ball_contact"]))
    net_contact_rate = float(np.mean(arrays["ball_net_contact"]))
    link_contact_rate = float(np.mean(arrays["ball_link_contact"]))
    ball_distance = float(np.mean(arrays["ball_pelvis_xy_distance"]))
    ball_speed = float(np.mean(arrays["ball_xy_speed"]))
    ball_forward_speed = float(np.mean(arrays["ball_command_forward_speed"]))
    pelvis_speed = float(np.mean(arrays["pelvis_xy_speed"]))
    foot_error = float(np.mean(arrays["foot_reference_position_error"]))
    heading_error = float(np.mean(np.abs(arrays["heading_error"])))
    too_close_rate = float(np.mean(arrays["ball_command_forward_offset"] < 0.28))
    finite_contact_duty = arrays["ball_contact_duty_ema"][
        np.isfinite(arrays["ball_contact_duty_ema"])
    ]
    contact_duty = (
        float(np.mean(finite_contact_duty)) if finite_contact_duty.size else np.nan
    )
    labeled_mask = arrays["cg_label_available"].astype(bool)
    window_mask = labeled_mask & arrays["cg_contact_window_active"].astype(bool)
    outside_window_mask = labeled_mask & ~window_mask
    window_end_mask = window_mask.copy()
    if window_end_mask.size > 1:
        window_end_mask[:-1] &= (
            ~window_mask[1:]
            | arrays["done"][:-1].astype(bool)
            | (arrays["motion_idx"][:-1] != arrays["motion_idx"][1:])
        )
    # Do not count a trace that stops halfway through a contact window as a
    # failed event.  The last sample is complete only when the episode ended.
    window_end_mask[-1] &= arrays["done"][-1].astype(bool)
    observed_window_contact_mask = window_mask & arrays["ball_link_contact"].astype(bool)
    premature_rate = (
        float(np.mean(arrays["cg_premature_contact"][outside_window_mask]))
        if np.any(outside_window_mask) else np.nan
    )
    window_hit_rate = (
        float(np.mean(arrays["cg_contact_window_hit"][window_end_mask]))
        if np.any(window_end_mask) else np.nan
    )
    missing_rate = 1.0 - window_hit_rate if np.isfinite(window_hit_rate) else np.nan
    wrong_foot_rate = (
        float(np.mean(arrays["cg_wrong_foot_contact"][observed_window_contact_mask]))
        if np.any(observed_window_contact_mask) else np.nan
    )
    task_state_names = np.asarray(["idle", "dribble", "stop"])
    task_state_counts = np.bincount(arrays["task_state"].astype(np.int64), minlength=3)[:3]
    task_state_summary = ", ".join(
        f"{name}={count}" for name, count in zip(task_state_names, task_state_counts)
    )
    stop_mask = arrays["stop_active"].astype(bool)
    stop_settle_rate = (
        float(np.mean(arrays["stop_settled"][stop_mask])) if np.any(stop_mask) else np.nan
    )
    stop_successes = int(np.count_nonzero(arrays["stop_success"].astype(bool)))
    stop_ball_speed = (
        float(np.mean(arrays["stop_ball_speed"][stop_mask])) if np.any(stop_mask) else np.nan
    )
    stop_pelvis_speed = (
        float(np.mean(arrays["stop_pelvis_speed"][stop_mask])) if np.any(stop_mask) else np.nan
    )
    stop_pelvis_angular_speed = (
        float(np.mean(arrays["stop_pelvis_angular_speed"][stop_mask])) if np.any(stop_mask) else np.nan
    )
    arm_error = float(np.mean(np.abs(arrays["actual_joint_pos"] - arrays["reference_joint_pos"])))
    trunk_error = float(
        np.mean(np.abs(arrays["trunk_actual_joint_pos"] - arrays["trunk_reference_joint_pos"]))
    )
    trunk_action_delta = np.diff(arrays["trunk_applied_action"], axis=0)
    trunk_action_step = (
        float(np.mean(np.linalg.norm(trunk_action_delta, axis=1))) if trunk_action_delta.size else np.nan
    )
    torso_rel_tilt = float(
        np.mean(np.linalg.norm(arrays["torso_minus_pelvis_rpy"][:, :2], axis=1))
    )
    torso_rel_tilt_error = float(
        np.mean(np.linalg.norm(arrays["torso_minus_pelvis_rpy_error"][:, :2], axis=1))
    )
    torso_rel_ang_vel = float(
        np.mean(np.linalg.norm(arrays["torso_minus_pelvis_ang_vel_w"], axis=1))
    )
    trunk_target_error = float(np.nanmean(np.abs(arrays["trunk_post_step_target_error"])))
    trunk_target_reference_offset = float(np.nanmean(np.abs(arrays["trunk_target_minus_reference"])))
    trunk_pitch_filter_values = np.abs(
        arrays["trunk_pitch_filtered_target"] - arrays["trunk_pitch_raw_target"]
    )
    finite_trunk_pitch_filter_values = trunk_pitch_filter_values[
        np.isfinite(trunk_pitch_filter_values)
    ]
    trunk_pitch_filter_delta = (
        float(np.mean(finite_trunk_pitch_filter_values))
        if finite_trunk_pitch_filter_values.size
        else np.nan
    )
    finite_trunk_pitch_overflow = arrays["trunk_pitch_reference_overflow"][
        np.isfinite(arrays["trunk_pitch_reference_overflow"])
    ]
    trunk_pitch_overflow = (
        float(np.mean(finite_trunk_pitch_overflow)) if finite_trunk_pitch_overflow.size else np.nan
    )
    trunk_effort_utilization = arrays["trunk_effort_utilization"]
    finite_effort_utilization = trunk_effort_utilization[np.isfinite(trunk_effort_utilization)]
    effort_utilization_p95 = (
        float(np.percentile(finite_effort_utilization, 95)) if finite_effort_utilization.size else np.nan
    )
    computed_effort_utilization = arrays["trunk_computed_effort_utilization"]
    finite_computed_effort_utilization = computed_effort_utilization[
        np.isfinite(computed_effort_utilization)
    ]
    computed_effort_utilization_p95 = (
        float(np.percentile(finite_computed_effort_utilization, 95))
        if finite_computed_effort_utilization.size else np.nan
    )
    effort_saturation_fraction = float(np.nanmean(arrays["trunk_effort_saturated"]))
    trunk_limit_margin = arrays["trunk_actual_limit_margin"]
    finite_limit_margin = trunk_limit_margin[np.isfinite(trunk_limit_margin)]
    limit_margin_p05 = (
        float(np.percentile(finite_limit_margin, 5)) if finite_limit_margin.size else np.nan
    )
    terminations = int(np.sum(arrays["done"]))
    finite_projection = arrays["manifold_projection_error"][
        np.isfinite(arrays["manifold_projection_error"])
    ]
    finite_nullspace = arrays["manifold_nullspace_residual"][
        np.isfinite(arrays["manifold_nullspace_residual"])
    ]
    finite_clip = arrays["manifold_latent_clip_fraction"][
        np.isfinite(arrays["manifold_latent_clip_fraction"])
    ]
    projection_error = float(np.mean(finite_projection)) if finite_projection.size else np.nan
    nullspace_residual = float(np.mean(finite_nullspace)) if finite_nullspace.size else np.nan
    latent_clip = float(np.mean(finite_clip)) if finite_clip.size else np.nan
    term_reasons = arrays["termination_reason"][arrays["termination_reason"] != ""]
    unique_reasons, reason_counts = np.unique(term_reasons, return_counts=True)
    reason_summary = ", ".join(f"{name}={count}" for name, count in zip(unique_reasons, reason_counts)) or "none"
    print(f"[INFO] Diagnostic ({len(diagnostic['step'])} samples) → {diagnostic['path']}")
    print(
        "[INFO] Counterfactual metrics: "
        f"contact_rate={contact_rate:.3f}  raw_link_contact_rate={link_contact_rate:.3f}  "
        f"net_contact_rate={net_contact_rate:.3f}  "
        f"ball_pelvis_xy={ball_distance:.3f} m  "
        f"too_close_rate={too_close_rate:.3f}  contact_duty={contact_duty:.3f}  "
        f"cg_window_hit={window_hit_rate:.3f}  cg_premature={premature_rate:.3f}  "
        f"cg_missing_window={missing_rate:.3f}  "
        f"cg_wrong_foot={wrong_foot_rate:.3f}  "
        f"ball_xy_speed={ball_speed:.3f} m/s  ball_cmd_speed={ball_forward_speed:.3f} m/s  "
        f"pelvis_xy_speed={pelvis_speed:.3f} m/s  "
        f"task_states=({task_state_summary})  stop_settled={stop_settle_rate:.3f}  "
        f"stop_successes={stop_successes}  "
        f"stop_ball_speed={stop_ball_speed:.3f} m/s  stop_pelvis_speed={stop_pelvis_speed:.3f} m/s  "
        f"stop_pelvis_w={stop_pelvis_angular_speed:.3f} rad/s  "
        f"foot_ref_err={foot_error:.3f} m  mean_abs_heading_err={heading_error:.3f} rad  "
        f"arm_joint_err={arm_error:.3f} rad  waist_joint_err={trunk_error:.3f} rad  "
        f"waist_action_step={trunk_action_step:.3f}  torso_rel_tilt={torso_rel_tilt:.3f} rad  "
        f"torso_rel_tilt_err={torso_rel_tilt_error:.3f} rad  "
        f"torso_rel_ang_vel={torso_rel_ang_vel:.3f} rad/s  manifold_projection={projection_error:.3f} rad  "
        f"manifold_nullspace={nullspace_residual:.3f} rad  "
        f"waist_target_err={trunk_target_error:.3f} rad  "
        f"waist_target_ref_offset={trunk_target_reference_offset:.3f} rad  "
        f"waist_pitch_filter_delta={trunk_pitch_filter_delta:.3f} rad  "
        f"waist_pitch_overflow={trunk_pitch_overflow:.3f} rad  "
        f"waist_computed_effort_util_p95={computed_effort_utilization_p95:.3f}  "
        f"waist_effort_util_p95={effort_utilization_p95:.3f}  "
        f"waist_effort_sat={effort_saturation_fraction:.3f}  "
        f"waist_limit_margin_p05={limit_margin_p05:.3f} rad  "
        f"latent_clip={latent_clip:.3f}  terminations={terminations} ({reason_summary})"
    )


def _apply_play_locomotion_command(env, args_cli) -> bool:
    """Apply CLI locomotion: multi-segment polar sequence or legacy vx/vy/wz."""
    polar_set = (
        args_cli.locomotion_cmd_speed is not None
        or args_cli.locomotion_cmd_heading is not None
        or args_cli.locomotion_cmd_duration is not None
        or args_cli.locomotion_task_state is not None
    )
    cartesian_fields = (
        args_cli.locomotion_cmd_vx,
        args_cli.locomotion_cmd_vy,
        args_cli.locomotion_cmd_vz,
        args_cli.locomotion_cmd_wx,
        args_cli.locomotion_cmd_wy,
    )
    if not polar_set and all(v is None for v in cartesian_fields):
        return False

    base_env = _resolve_base_env(env)
    cmd = base_env.command_manager.get_term("motion")
    if not hasattr(cmd, "set_locomotion_manual_command"):
        print("[WARN] Task motion command has no manual locomotion API; --locomotion_cmd_* ignored.")
        return False

    if polar_set and hasattr(cmd, "set_locomotion_polar_command"):
        speeds = args_cli.locomotion_cmd_speed
        headings = args_cli.locomotion_cmd_heading
        durations = args_cli.locomotion_cmd_duration
        task_states = args_cli.locomotion_task_state
        if task_states is not None and not bool(
            getattr(cmd.cfg, "locomotion_task_state_enabled", False)
        ):
            print(
                "[WARN] This task does not expose the high-level locomotion state to the policy or "
                "enable state-gated rewards. --locomotion_task_state values are diagnostic labels only; "
                "IDLE and STOP both reduce to the supplied zero-speed command."
            )

        if speeds is None:
            if task_states is not None:
                if len(task_states) != 1:
                    raise ValueError("State-only polar control accepts exactly one --locomotion_task_state.")
                speeds = [0.0 if task_states[0] in {"idle", "stop"} else float(
                    torch.norm(cmd.locomotion_manual_lin_vel[0, :2]).item()
                )]
            else:
                speeds = [float(torch.norm(cmd.locomotion_manual_lin_vel[0, :2]).item())]
        if headings is None:
            headings = [
                float(
                    torch.atan2(cmd.locomotion_manual_lin_vel[0, 1], cmd.locomotion_manual_lin_vel[0, 0]).item()
                )
            ]
        if durations is None:
            durations = [2.0]

        n = len(speeds)
        if len(headings) != n or len(durations) != n:
            raise ValueError(
                f"--locomotion_cmd_speed ({n}), --locomotion_cmd_heading ({len(headings)}), "
                f"and --locomotion_cmd_duration ({len(durations)}) must have the same length."
            )
        if task_states is not None and len(task_states) != n:
            raise ValueError(
                f"--locomotion_task_state ({len(task_states)}) must match --locomotion_cmd_speed ({n})."
            )
        if task_states is None:
            task_states = ["dribble" if float(speed) > 0.05 else "stop" for speed in speeds]

        wz_raw = args_cli.locomotion_cmd_wz
        if wz_raw is None or len(wz_raw) == 0:
            wz_list = [0.0] * n
        elif len(wz_raw) == 1:
            wz_list = [float(wz_raw[0])] * n
        elif len(wz_raw) == n:
            wz_list = [float(w) for w in wz_raw]
        else:
            raise ValueError(
                f"--locomotion_cmd_wz must be omitted, length 1 (broadcast), or match speed count ({n})."
            )

        if n > 1 and hasattr(cmd, "set_locomotion_polar_sequence"):
            segments = [(speeds[i], headings[i], durations[i], wz_list[i], task_states[i]) for i in range(n)]
            cmd.set_locomotion_polar_sequence(
                segments,
                hold_last=not args_cli.locomotion_cmd_loop,
                reset_on_end=args_cli.locomotion_cmd_reset_on_end,
            )
            print(f"[INFO] Locomotion sequence ({n} segments):")
            for i, (sp, hd, dur, wz, state) in enumerate(segments):
                print(
                    f"  [{i + 1}] state={state.upper():7s}  speed={sp:.3f} m/s  "
                    f"heading={hd:.3f} rad  duration={dur:.2f} s  wz={wz:.3f}"
                )
            if args_cli.locomotion_cmd_reset_on_end:
                print("  final segment -> environment reset -> segment 1")
            elif args_cli.locomotion_cmd_loop:
                print("  looping: final segment -> first segment")
            else:
                print("  holding: final segment remains active")
            return True

        cmd.set_locomotion_command_mode("manual")
        cmd.set_locomotion_polar_command(
            speed=speeds[0],
            heading=headings[0],
            duration_s=durations[0],
            wz=wz_list[0],
            task_state=task_states[0],
        )
        print(
            f"[INFO] Locomotion polar cmd: state={task_states[0].upper()}  speed={speeds[0]:.3f} m/s  "
            f"heading={headings[0]:.3f} rad  "
            f"duration={durations[0]:.2f} s  wz={wz_list[0]:.3f} rad/s"
        )
        return True

    cmd.set_locomotion_command_mode("manual")

    cur_lin = cmd.locomotion_manual_lin_vel[0].tolist()
    cur_ang = cmd.locomotion_manual_ang_vel[0].tolist()
    lin = [
        cur_lin[0] if args_cli.locomotion_cmd_vx is None else args_cli.locomotion_cmd_vx,
        cur_lin[1] if args_cli.locomotion_cmd_vy is None else args_cli.locomotion_cmd_vy,
        cur_lin[2] if args_cli.locomotion_cmd_vz is None else args_cli.locomotion_cmd_vz,
    ]
    ang = [
        cur_ang[0] if args_cli.locomotion_cmd_wx is None else args_cli.locomotion_cmd_wx,
        cur_ang[1] if args_cli.locomotion_cmd_wy is None else args_cli.locomotion_cmd_wy,
        cur_ang[2],
    ]
    cartesian_task_state = None
    if args_cli.locomotion_task_state is not None:
        if len(args_cli.locomotion_task_state) != 1:
            raise ValueError("Cartesian manual control accepts exactly one --locomotion_task_state.")
        cartesian_task_state = args_cli.locomotion_task_state[0]
    cmd.set_locomotion_manual_command(lin_vel=lin, ang_vel=ang, task_state=cartesian_task_state)
    print(
        f"[INFO] Locomotion manual cmd: lin_vel={lin} m/s  ang_vel={ang} rad/s  (task +X/+Y/+Z)"
    )
    return True


def _get_play_overlay(env) -> str:
    """HUD for dual-view video: speeds, distances, contact, and CG labels."""
    lines: list[str] = []
    try:
        base_env = _resolve_base_env(env)
        cmd = base_env.command_manager.get_term("motion")
        i = 0

        if hasattr(cmd, "motion_idx") and hasattr(cmd, "motion") and hasattr(cmd.motion, "motion_name"):
            mi = int(cmd.motion_idx[i].item())
            names = cmd.motion.motion_name
            n = len(names)
            ref = names[mi] if 0 <= mi < n else f"motion_{mi}"
            lines.append(f"Ref: {ref}  ({mi + 1}/{n})")

        t = int(cmd.time_steps[i].item())
        motion_len = int(cmd.motion_length[i].item())
        lines.append(f"Motion frame: {t}/{max(motion_len - 1, 0)}")
        if hasattr(cmd, "style_phase_wrap_count"):
            wraps = int(cmd.style_phase_wrap_count[i].item())
            lines.append(f"Style wraps: {wraps}")

        pelvis_vel = cmd.robot_anchor_lin_vel_w[i].detach().cpu().numpy()
        pelvis_pos = cmd.robot_pelvis_pos_w[i].detach().cpu().numpy()
        pelvis_sp_xy = float(np.linalg.norm(pelvis_vel[:2]))

        soccer_ball = base_env.scene["soccer_ball"]
        ball_vel = soccer_ball.data.root_lin_vel_w[i].detach().cpu().numpy()
        ball_pos = soccer_ball.data.root_pos_w[i].detach().cpu().numpy()
        ball_sp_xy = float(np.linalg.norm(ball_vel[:2]))

        pelvis_ball_xy = float(np.linalg.norm(ball_pos[:2] - pelvis_pos[:2]))
        pelvis_ball_3d = float(np.linalg.norm(ball_pos - pelvis_pos))

        lines.append(
            f"Pelvis v_xy: {pelvis_sp_xy:.2f} m/s  |  Ball v_xy: {ball_sp_xy:.2f} m/s"
        )

        if hasattr(cmd, "locomotion_lin_vel_command_w"):
            mode = getattr(cmd, "locomotion_command_mode", "?")
            lcmd = cmd.locomotion_lin_vel_command_w()[i].detach().cpu().numpy()
            task_state = getattr(cmd, "locomotion_task_state", None)
            task_state_names = ("IDLE", "DRIBBLE", "STOP")
            task_state_name = "DRIBBLE"
            if isinstance(task_state, torch.Tensor):
                task_state_idx = int(task_state[i].item())
                if 0 <= task_state_idx < len(task_state_names):
                    task_state_name = task_state_names[task_state_idx]

            # heading / arrow helpers — ASCII-only for safe video font rendering
            _ARROWS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
            def _arrow(rad):
                idx = int(((-float(rad)) / (2 * np.pi) * 8 + 0.5) % 8)
                return _ARROWS[idx]
            def _deg(rad):
                return float(rad) * 180.0 / np.pi

            # For reference (follow) mode locomotion_cmd_speed/heading are not updated;
            # derive them from the velocity command vector instead.
            if mode == "reference":
                cmd_spd = float(np.linalg.norm(lcmd[:2]))
                cmd_hdg = float(np.arctan2(lcmd[1], lcmd[0]))
            else:
                cmd_spd = cmd.locomotion_cmd_speed[i].item()
                cmd_hdg = cmd.locomotion_cmd_heading[i].item()

            # tracking quality: cosine similarity of xy velocity vectors
            cmd_xy = lcmd[:2]
            rob_xy = pelvis_vel[:2]
            cmd_norm = float(np.linalg.norm(cmd_xy))
            rob_norm = float(np.linalg.norm(rob_xy))
            if cmd_norm > 0.05 and rob_norm > 0.05:
                cos_sim = float(np.dot(cmd_xy, rob_xy) / (cmd_norm * rob_norm))
                track_str = f"  track={max(0.0, cos_sim) * 100:3.0f}%"
            else:
                track_str = "  track= n/a"

            # segment / hold info
            seg_note = ""
            if hasattr(cmd, "_locomotion_segment_plans") and cmd._locomotion_segment_plans[i]:
                seg_idx = int(cmd._locomotion_segment_idx[i].item()) + 1
                seg_note = f" seg {seg_idx}/{len(cmd._locomotion_segment_plans[i])}"
            hold_str = ""
            if mode != "reference" and hasattr(cmd, "_locomotion_cmd_hold_steps_remaining"):
                hold_str = f"  hold={int(cmd._locomotion_cmd_hold_steps_remaining[i].item()):4d}"

            # fixed-width columns so digits don't shift layout
            # col widths: spd 4, deg 4, dir 2, track 7, hold 9
            lines.append(
                f"Loco cmd ({mode}{seg_note}):"
                f"  {cmd_spd:4.2f} m/s"
                f"  {_deg(cmd_hdg):+4.0f}deg {_arrow(cmd_hdg):<2s}"
                f"{track_str}"
                f"{hold_str}"
            )
            lines.append(f"Task state         : {task_state_name}")
            # Control smooths a requested endpoint into the effective command
            # above.  Show both during a turn so speed reduction is observable.
            target_speed_buf = getattr(cmd, "locomotion_cmd_target_speed", None)
            target_heading_buf = getattr(cmd, "locomotion_cmd_target_heading", None)
            if target_speed_buf is not None and target_heading_buf is not None:
                target_spd = float(target_speed_buf[i].item())
                target_hdg = float(target_heading_buf[i].item())
                heading_gap = float(np.arctan2(np.sin(target_hdg - cmd_hdg), np.cos(target_hdg - cmd_hdg)))
                if abs(target_spd - cmd_spd) > 0.01 or abs(heading_gap) > 0.01:
                    lines.append(
                        f"Requested endpoint :"
                        f"  {target_spd:4.2f} m/s"
                        f"  {_deg(target_hdg):+4.0f}deg {_arrow(target_hdg):<2s}"
                    )

            # actual robot velocity vs command
            actual_hdg = float(np.arctan2(pelvis_vel[1], pelvis_vel[0]))
            hdg_err_deg = float((_deg(actual_hdg) - _deg(cmd_hdg) + 180) % 360 - 180)
            vx_err = float(pelvis_vel[0] - lcmd[0])
            vy_err = float(pelvis_vel[1] - lcmd[1])

            lines.append(
                f"Robot actual        :"
                f"  {pelvis_sp_xy:4.2f} m/s"
                f"  {_deg(actual_hdg):+4.0f}deg {_arrow(actual_hdg):<2s}"
                f"  hdg_err={hdg_err_deg:+4.0f}deg"
                f"  vx_err={vx_err:+5.2f}"
                f"  vy_err={vy_err:+5.2f}"
            )

        lines.append(
            f"Pelvis-Ball: {pelvis_ball_xy:.2f} m (xy)  |  {pelvis_ball_3d:.2f} m (3D)"
        )

        net_force_xy = float(
            soccer_ball_contact_force_magnitude(base_env, _BALL_SENSOR_NAME)[i].item()
        )
        sim_touch = bool(
            soccer_ball_robot_contact(
                base_env,
                _BALL_SENSOR_NAME,
                contact_force_threshold=_CONTACT_FORCE_THRESHOLD,
                hold_steps=2,
            )[i].item()
        )

        ankle_parts = [
            f"contact={'YES' if sim_touch else 'NO'}",
            f"net_xy={net_force_xy:.1f}N",
        ]
        if hasattr(cmd, "motion_has_dribble_cg_label") and bool(cmd.motion_has_dribble_cg_label[i].item()):
            cg_window_tensor = getattr(base_env, "_dribbling_cg_contact_window_active", None)
            cg_window_hit_tensor = getattr(base_env, "_dribbling_cg_contact_window_hit", None)
            cg_window = (
                bool(cmd.dribble_cg_contact_ref[i].item())
                if cg_window_tensor is None
                else bool(cg_window_tensor[i].item())
            )
            cg_window_hit = (
                cg_window and sim_touch
                if cg_window_hit_tensor is None
                else bool(cg_window_hit_tensor[i].item())
            )
            ankle_parts.append(f"cg_window={'YES' if cg_window else 'NO'}")
            if cg_window:
                ankle_parts.append(f"hit={'YES' if cg_window_hit else 'WAIT'}")
        lines.append(f"Ankle-Ball: {'  |  '.join(ankle_parts)}")

        if hasattr(base_env, "_dribbling_no_contact_count"):
            no_contact_count = float(base_env._dribbling_no_contact_count[i].item())
            command_recovery = bool(base_env._dribbling_no_contact_recovery_active[i].item())
            proximity_recovery_tensor = getattr(
                base_env, "_dribbling_no_contact_proximity_recovery_active", None
            )
            proximity_recovery = (
                False if proximity_recovery_tensor is None else bool(proximity_recovery_tensor[i].item())
            )
            closing = float(base_env._dribbling_no_contact_closing_speed[i].item())
            relative_speed_tensor = getattr(base_env, "_dribbling_no_contact_relative_speed", None)
            relative_speed = 0.0 if relative_speed_tensor is None else float(relative_speed_tensor[i].item())
            recovery = "CMD" if command_recovery else "PROX" if proximity_recovery else "NO"
            lines.append(
                f"No-contact: count={no_contact_count:5.1f}"
                f"  recovery={recovery}"
                f"  closing={closing:+.2f} m/s  relative={relative_speed:.2f} m/s"
            )

        _update_last_termination_reason(base_env, env_idx=i)
        lines.append(f"Last term: {_LAST_TERM_REASON}")

    except Exception as e:
        lines.append(f"HUD error: {e}")

    return "\n".join(lines)

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.viewer.origin_type = None
    env_cfg.viewer.asset_name = None

    # For video recording: set a wide-angle camera that follows the robot.
    if args_cli.video:
        env_cfg.viewer.eye = (5.0, 5.0, 3.0)       # 5m back + 5m side + 3m up
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)     # look at robot's waist height
        env_cfg.viewer.origin_type = "asset_root"    # camera follows the robot
        env_cfg.viewer.asset_name = "robot"           # track the robot asset
        env_cfg.viewer.env_index = 0

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    motion_files: list[str] = []

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
            motion_files = [args_cli.motion_file]

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        # Select single-motion or multi-motion mode from CLI args.
        if args_cli.motion_file is not None:
            # Single-motion mode: play and export.
            motion_files = [args_cli.motion_file]
            print(f"[INFO]: Using single motion file: {args_cli.motion_file}")
        elif args_cli.motion_path is not None:
            motion_files = get_motion_files(args_cli.motion_path)
        else:
            raise ValueError("Either --motion_file or --motion_path must be specified.")

        # For state-machine environments: auto-split approach/strike files.
        approach_files = [f for f in motion_files if f.endswith("_approach.npz")]
        strike_files = [f for f in motion_files if f.endswith("_strike.npz")]

        if approach_files and strike_files:
            env_cfg.commands.motion.motion_files = approach_files
            if hasattr(env_cfg.commands.motion, "strike_motion_files"):
                env_cfg.commands.motion.strike_motion_files = strike_files
                print(f"[INFO] State-machine mode: {len(approach_files)} approach + {len(strike_files)} strike files")
            else:
                env_cfg.commands.motion.motion_files = motion_files
        else:
            env_cfg.commands.motion.motion_files = motion_files
            if hasattr(env_cfg.commands.motion, "strike_motion_files"):
                env_cfg.commands.motion.strike_motion_files = motion_files

        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    total_play_steps = _setup_play_episode_limit(
        env_cfg,
        motion_files,
        keep_failure_terms=not args_cli.disable_training_terminations,
    )
    term_note = ", terminations off" if args_cli.disable_training_terminations else ""
    if len(motion_files) > 1:
        print(
            f"[INFO] Sequential playback: {len(motion_files)} references, "
            f"{total_play_steps} steps total (episode cap {env_cfg.episode_length_s:.1f}s{term_note})"
        )
    elif motion_files:
        print(
            f"[INFO] Playback episode cap {env_cfg.episode_length_s:.1f}s "
            f"({total_play_steps} frames{term_note})"
        )

    if args_cli.record_all_motions and len(motion_files) <= 1:
        raise ValueError("--record_all_motions requires --motion_path with multiple .npz files.")

    play_steps = args_cli.video_length
    if play_steps is None:
        if len(motion_files) > 1 or args_cli.video or args_cli.dual_view:
            play_steps = total_play_steps
        else:
            play_steps = None
    elif args_cli.record_all_motions and len(motion_files) > 1:
        print(f"[INFO] --record_all_motions: using explicit --video_length={play_steps}")

    waist_roll_stiffness_scale = _apply_play_waist_roll_stiffness_scale(
        env_cfg, args_cli.waist_roll_stiffness_scale
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    log_dir = os.path.dirname(resume_path)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    _apply_play_locomotion_command(env, args_cli)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_checkpoint_with_obs_expand(ppo_runner, resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    reference_constraints: list[dict] = []
    if args_cli.upper_body_reference_margin is not None:
        upper_body_constraint = _create_upper_body_reference_constraint(
            env, args_cli.upper_body_reference_margin, args_cli.upper_body_constraint_group
        )
        reference_constraints.append(upper_body_constraint)
        print(
            "[INFO] Play-only upper-body reference constraint enabled: "
            f"group={upper_body_constraint['group']}  "
            f"q_target ∈ q_ref ± {upper_body_constraint['margin']:.3f} rad"
        )

    if args_cli.waist_reference_margin is not None:
        waist_constraint = _create_waist_reference_constraint(env, args_cli.waist_reference_margin)
        reference_constraints.append(waist_constraint)
        print(
            "[INFO] Play-only waist reference constraint enabled: "
            f"q_target within q_ref +/- {waist_constraint['margin']:.3f} rad"
        )

    diagnostic = None
    if args_cli.diagnostic:
        diagnostic = _create_diagnostic(
            env,
            log_dir,
            args_cli.diagnostic_stride,
            reference_constraints,
            waist_roll_stiffness_scale,
        )
        print("[INFO] Diagnostic scope: arms + waist + pelvis/torso motion.")
        print(f"[INFO] Diagnostic enabled (stride={diagnostic['stride']}) → {diagnostic['path']}")

    # export policy to onnx/jit
    export_targets: list[tuple[str, str]] = []

    if args_cli.motion_file is not None:
        # Single-file mode: export directly using the requested name or file name.
        export_name = args_cli.export_motion_name or os.path.basename(args_cli.motion_file)
        export_targets.append((args_cli.motion_file, export_name))
    elif args_cli.motion_path is not None and args_cli.export_motion_name is not None:
        # Directory mode: export by matching names from export_motion_name.
        if args_cli.export_motion_name.strip().lower() == "all":
            export_targets = [(mf, os.path.basename(mf)) for mf in motion_files]
        else:
            requested_names = [n.strip() for n in args_cli.export_motion_name.split(",") if n.strip()]
            for name in requested_names:
                match = next(
                    (
                        mf
                        for mf in motion_files
                        if os.path.splitext(os.path.basename(mf))[0] == os.path.splitext(name)[0]
                        or os.path.basename(mf) == name
                    ),
                    None,
                )
                if match is None:
                    raise ValueError(f"Requested export motion '{name}' not found in {args_cli.motion_path}.")
                export_targets.append((match, name))

    if export_targets:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        ckpt = args_cli.checkpoint.split('_')[1].split('.')[0]

        for motion_file, export_name in export_targets:
            export_stem = os.path.splitext(export_name)[0]
            filename = f"policy_{ckpt}_{export_stem}.onnx"
            export_motion_policy_as_onnx(
                env.unwrapped,
                ppo_runner.alg.policy,
                normalizer=ppo_runner.obs_normalizer,
                path=export_model_dir,
                filename=filename,
                motion_name=export_name,
            )
            attach_onnx_metadata(
                env.unwrapped,
                args_cli.wandb_path if args_cli.wandb_path else "none",
                export_model_dir,
                filename=filename,
            )
            print(f"[INFO]: Exported policy for {export_name} to: {os.path.join(export_model_dir, filename)}")
    else:
        print("[INFO]: Skipping policy export (set --export_motion_name to enable export).")
    
    # --- Video recorder with HUD overlay (--video and/or --dual_view) ---
    video_recorder = None
    if args_cli.video or args_cli.dual_view:
        from dual_view_recorder import DualViewRecorder, resolve_camera_offsets

        tag = "dual" if args_cli.dual_view else "play"
        video_dir = os.path.join(
            log_dir, "videos", f"{tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        front_offset, back_offset = resolve_camera_offsets(layout=args_cli.cam_layout)
        video_recorder = DualViewRecorder(
            env=env.unwrapped if hasattr(env, "unwrapped") else env,
            output_dir=video_dir,
            resolution=(960, 540),
            front_offset=front_offset,
            back_offset=back_offset,
            lookat_offset=0.5,
            fps=30,
            path_tracing=args_cli.path_tracing,
            dual=args_cli.dual_view,
        )
        video_recorder.setup()
        for _ in range(5):
            env.unwrapped.sim.render()
        mode = "dual-view" if args_cli.dual_view else "single-view"
        steps_label = play_steps if play_steps is not None else "unlimited"
        print(f"[INFO] {mode} recording ({steps_label} steps) → {video_dir}")
        print("[INFO] HUD: Ref name, speeds, distances, ankle-ball contact, last term.")
    elif play_steps is not None:
        print(f"[INFO] Running {play_steps} steps (sequential references)")

    # reset environment
    global _LAST_TERM_REASON
    _LAST_TERM_REASON = "-"
    obs, _ = env.get_observations()
    base_env = _resolve_base_env(env)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            policy_actions = policy(obs)
            actions = policy_actions
            for reference_constraint in reference_constraints:
                actions = _constrain_reference_actions(env, actions, reference_constraint)
            recorded_diagnostic_sample = False
            if diagnostic is not None:
                recorded_diagnostic_sample = _append_diagnostic(
                    diagnostic, env, policy_actions, actions, timestep
                )
            # env stepping
            obs, reward, dones, _ = env.step(actions)
            if recorded_diagnostic_sample:
                _append_upper_body_manifold_diagnostic(diagnostic, env)
                diagnostic["step_reward"][-1] = float(reward[0].item())
                diagnostic["done"].append(bool(dones[0].item()))
                diagnostic["reward_terms"].append(_reward_term_values(base_env))
                diagnostic["termination_reason"].append(
                    _active_failure_termination_reason(base_env) if bool(dones[0].item()) else ""
                )

        if video_recorder is not None:
            overlay = _get_play_overlay(env)
            video_recorder.capture(overlay_text=overlay)

        timestep += 1
        if play_steps is not None and timestep >= play_steps:
            break

    if video_recorder is not None:
        video_recorder.save()

    if diagnostic is not None:
        _save_diagnostic(diagnostic)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
