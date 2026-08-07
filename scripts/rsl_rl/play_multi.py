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
    help="Manual locomotion cmd: forward linear speed (m/s); pelvis-local for local-twist tasks.",
)
parser.add_argument(
    "--locomotion_cmd_vy",
    type=float,
    default=None,
    help="Manual locomotion cmd: lateral speed (m/s); pelvis-local for local-twist tasks.",
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
    "--locomotion_cmd_plan",
    type=str,
    default=None,
    help=(
        "JSON local-twist playback plan. Each segment supplies pelvis-local "
        "vx, vy, wz, and duration_s; incompatible with --locomotion_cmd_speed/heading/state."
    ),
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
        "Save action-layer, command, pelvis/torso, contact-region, phase, and reward telemetry to a .npz file."
    ),
)
parser.add_argument(
    "--diagnostic_stride",
    type=int,
    default=1,
    help="Record every N simulator steps with --diagnostic (default: 1).",
)
parser.add_argument(
    "--show_s2_contact_regions",
    action="store_true",
    default=False,
    help=(
        "Show the active S2 reference-foot frame and the inner/outer boundaries "
        "of its expected left/right ball region."
    ),
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
import json
import pathlib
import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, yaw_quat

from soccer.tasks.tracking.mdp.rewards_dribbling import (
    dribbling_contact_telemetry,
    soccer_ball_contact_force_magnitude,
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
_DIAGNOSTIC_SCHEMA_VERSION = "dribble-v6"

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


def _diagnostic_reward_params(base_env, reward_name: str) -> dict:
    """Return a reward's effective params, unwrapping task-state gates."""
    rewards_cfg = getattr(getattr(base_env, "cfg", None), "rewards", None)
    term_cfg = getattr(rewards_cfg, reward_name, None)
    params = getattr(term_cfg, "params", None)
    if not isinstance(params, dict):
        return {}
    nested = params.get("reward_params")
    return dict(nested) if isinstance(nested, dict) else dict(params)


def _diagnostic_contact_settings(base_env) -> dict:
    """Resolve the exact legal-touch geometry configured for this task."""
    params = _diagnostic_reward_params(base_env, "dribbling_legal_foot_touch")
    if not params:
        params = _diagnostic_reward_params(base_env, "s2_new_touch")
    all_body_cfg = params.get("all_body_cfg")
    return {
        "command_name": str(params.get("command_name", "motion")),
        "ball_sensor_name": str(params.get("ball_sensor_name", _BALL_SENSOR_NAME)),
        "all_body_cfg": all_body_cfg,
        "num_ankle_links": int(params.get("num_ankle_links", 2)),
        "contact_surface": str(params.get("contact_surface", "any")),
        "medial_y_min": float(params.get("medial_y_min", 0.018)),
        "contact_force_threshold": float(
            params.get("force_threshold", params.get("max_touch_force", 14.0))
        ),
        "cg_gated": bool(params.get("cg_gated", False)),
        "cg_surface_gated": bool(params.get("cg_surface_gated", False)),
        "body_names": tuple(getattr(all_body_cfg, "body_names", ()) or ()),
    }


def _active_locomotion_command(command) -> dict[str, torch.Tensor | str | bool]:
    """Describe the velocity command actually active for this policy step.

    Reference-mode tasks derive their command from the yaw-aligned motion
    anchor instead of the resampled/manual command buffers. This helper keeps
    diagnostic fields correct for every mode.
    """
    mode = str(getattr(command, "locomotion_command_mode", "reference"))
    if hasattr(command, "locomotion_lin_vel_command_w"):
        lin_vel_w = command.locomotion_lin_vel_command_w()
    else:
        lin_vel_w = command.anchor_lin_vel_w
    if hasattr(command, "locomotion_ang_vel_command_w"):
        ang_vel_w = command.locomotion_ang_vel_command_w()
    else:
        ang_vel_w = command.anchor_ang_vel_w

    is_local_twist = str(getattr(command.cfg, "locomotion_command_frame", "world")) == "pelvis_local"
    if is_local_twist and hasattr(command, "locomotion_twist_command_b"):
        twist_b = command.locomotion_twist_command_b()
        reference_twist_b = command.reference_locomotion_twist_b()
        blend_alpha = float(command.locomotion_twist_blend_alpha())
    else:
        # Legacy-only fallback.  These fields are retained for a uniform
        # archive schema but are not a local-frame ground-truth comparison.
        twist_b = torch.stack((lin_vel_w[:, 0], lin_vel_w[:, 1], ang_vel_w[:, 2]), dim=-1)
        reference_twist_b = twist_b
        blend_alpha = 1.0

    speed = torch.norm(lin_vel_w[:, :2], dim=-1)
    heading = torch.atan2(lin_vel_w[:, 1], lin_vel_w[:, 0])
    heading_valid = speed > 1.0e-6
    if mode in {"manual", "resampled"}:
        target_speed = getattr(command, "locomotion_cmd_target_speed", command.locomotion_cmd_speed)
        target_heading = getattr(command, "locomotion_cmd_target_heading", command.locomotion_cmd_heading)
    else:
        # A reference command has no independent requested endpoint.
        target_speed = speed
        target_heading = heading
    return {
        "mode": mode,
        "lin_vel_w": lin_vel_w,
        "ang_vel_w": ang_vel_w,
        "speed": speed,
        "heading": heading,
        "heading_valid": heading_valid,
        "target_speed": target_speed,
        "target_heading": target_heading,
        "twist_b": twist_b,
        "reference_twist_b": reference_twist_b,
        "blend_alpha": blend_alpha,
    }


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
    if hasattr(action_term, "diagnostic_snapshot_enabled"):
        action_term.diagnostic_snapshot_enabled = True
    arm_action_ids = _action_ids_for_robot_joint_ids(action_joint_ids, joint_ids)
    trunk_full_action_ids = _action_ids_for_robot_joint_ids(action_joint_ids, trunk_joint_ids)
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
    contact_settings = _diagnostic_contact_settings(base_env)
    constraint_groups = np.asarray([constraint["group"] for constraint in constraints])
    constraint_margins = np.asarray([constraint["margin"] for constraint in constraints], dtype=np.float32)
    constraint_joint_names = (
        np.concatenate([constraint["joint_names"] for constraint in constraints])
        if constraints else np.asarray([])
    )
    return {
        "path": output_path,
        "stride": int(stride),
        "schema_version": _DIAGNOSTIC_SCHEMA_VERSION,
        "sample_timing": (
            "state/reference/policy/submitted/contact are pre-step; reward/done/executed_action "
            "are the resulting transition"
        ),
        "action_value_semantics": np.asarray(
            [
                "policy_action: raw policy output before playback constraints",
                "submitted_action: action passed to env.step after playback constraints",
                "effective_action: action after action-layer projection, limits, and filtering",
            ]
        ),
        "joint_ids": torch.as_tensor(joint_ids, dtype=torch.long, device=base_env.device),
        "joint_names": np.asarray(_ARM_DIAGNOSTIC_JOINT_NAMES),
        "action_ids": torch.as_tensor(arm_action_ids, dtype=torch.long, device=base_env.device),
        "direct_upper_body_latent": direct_upper_latent,
        "trunk_joint_ids": torch.as_tensor(trunk_joint_ids, dtype=torch.long, device=base_env.device),
        "trunk_joint_names": np.asarray(_TRUNK_DIAGNOSTIC_JOINT_NAMES),
        "trunk_action_ids": torch.as_tensor(trunk_action_ids, dtype=torch.long, device=base_env.device),
        "trunk_full_action_ids": torch.as_tensor(
            trunk_full_action_ids, dtype=torch.long, device=base_env.device
        ),
        "trunk_body_ids": torch.as_tensor(trunk_body_ids, dtype=torch.long, device=base_env.device),
        "trunk_reference_body_ids": torch.as_tensor(
            trunk_reference_body_ids, dtype=torch.long, device=base_env.device
        ),
        "trunk_body_names": np.asarray(_TRUNK_DIAGNOSTIC_BODY_NAMES),
        "reward_term_names": reward_term_names,
        "task_state_names": np.asarray(["idle", "dribble", "stop"]),
        "ball_spawn_source_names": np.asarray(["fallback_pelvis_local_front", "reference_first_contact"]),
        "constraint_group": "none" if not constraints else "+".join(constraint_groups.tolist()),
        "constraint_margin": float(constraint_margins[0]) if len(constraints) == 1 else np.nan,
        "constraint_joint_names": constraint_joint_names,
        "constraint_groups": constraint_groups,
        "constraint_margins": constraint_margins,
        "waist_roll_stiffness_scale": float(waist_roll_stiffness_scale),
        "waist_roll_damping_scale": float(waist_roll_stiffness_scale) ** 0.5,
        "contact_settings": contact_settings,
        "contact_region_frame": "foot_yaw",
        "locomotion_command_frame": str(
            getattr(command.cfg, "locomotion_command_frame", "world")
        ),
        "contact_body_names": np.asarray(contact_settings["body_names"]),
        "contact_surface": contact_settings["contact_surface"],
        "contact_force_threshold": contact_settings["contact_force_threshold"],
        "contact_cg_gated": contact_settings["cg_gated"],
        "contact_cg_surface_gated": contact_settings["cg_surface_gated"],
        "step": [],
        "motion_idx": [],
        "ball_spawn_source": [],
        "ball_spawn_reference_contact_frame": [],
        "ball_spawn_reference_local": [],
        "style_phase": [],
        "style_cycle_length": [],
        "style_source_first_frame": [],
        "style_source_second_frame": [],
        "style_seam_blend": [],
        "style_in_seam_bridge": [],
        "segment_idx": [],
        "task_state": [],
        "command_mode": [],
        "command_heading": [],
        "effective_command_speed": [],
        "effective_command_heading": [],
        "active_command_lin_vel_w": [],
        "active_command_ang_vel_w": [],
        "reference_twist_local": [],
        "active_twist_local": [],
        "actual_twist_local": [],
        "twist_local_error": [],
        "twist_blend_alpha": [],
        "command_heading_valid": [],
        "pelvis_yaw": [],
        "reference_joint_pos": [],
        "reference_joint_vel": [],
        "actual_joint_pos": [],
        "actual_joint_vel": [],
        "policy_action": [],
        "submitted_action": [],
        "effective_action": [],
        "applied_action": [],
        "action_snapshot_available": [],
        "post_step_state_valid": [],
        "upper_policy_latent": [],
        "trunk_reference_joint_pos": [],
        "trunk_reference_joint_vel": [],
        "trunk_actual_joint_pos": [],
        "trunk_actual_joint_vel": [],
        "trunk_policy_action": [],
        "trunk_submitted_action": [],
        "trunk_effective_action": [],
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
        "ball_contact": [],
        "contact_force_magnitude": [],
        "contact_body_idx": [],
        "contact_foot": [],
        "contact_legal_ankle": [],
        "contact_generic_instep": [],
        "contact_inside_instep": [],
        "contact_outside_instep": [],
        "contact_requested_surface_match": [],
        "contact_gentle": [],
        "contact_legal_touch": [],
        "contact_cg_gate_pass": [],
        "contact_cg_eligible": [],
        "contact_ball_offset_foot_yaw": [],
        "contact_medial_offset": [],
        "cg_label_available": [],
        "cg_reference_contact": [],
        "cg_reference_foot": [],
        "cg_reference_surface": [],
        "cg_contact_foot_match": [],
        "s2_contact_window": [],
        "s2_contact_event_id": [],
        "s2_contact_event_frame": [],
        "s2_expected_foot": [],
        "s2_expected_side": [],
        "s2_reference_foot_pos_w": [],
        "s2_reference_foot_yaw": [],
        "s2_ball_offset_reference_foot": [],
        "s2_target_region_distance": [],
        "cg_flow_label_available": [],
        "cg_flow_valid": [],
        "cg_flow_anchor_frame": [],
        "cg_flow_direction_local": [],
        "cg_flow_distance": [],
        "cg_flow_duration": [],
        "cg_flow_nominal_speed": [],
        "cg_flow_latched": [],
        "cg_flow_release_active": [],
        "cg_flow_target_direction_world": [],
        "cg_flow_parallel_speed": [],
        "cg_flow_lateral_speed": [],
        "cg_flow_release_reward": [],
        "cg_flow_progress": [],
        "cg_flow_progress_rate": [],
        "cg_flow_lateral_offset": [],
        "cg_flow_progress_reward": [],
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
        "manifold_joint_limited_upper_target": [],
        "manifold_executed_upper_target": [],
        "manifold_latent": [],
        "manifold_projection_error": [],
        "manifold_projection_error_after_reference_constraint": [],
        "manifold_nullspace_residual": [],
        "manifold_latent_clip_fraction": [],
        "manifold_reference_overflow": [],
        "manifold_reference_clamp_fraction": [],
        "manifold_joint_limit_clamp_fraction": [],
        "manifold_filter_lag": [],
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
    if hasattr(command, "ball_spawn_reference_info"):
        spawn_info = command.ball_spawn_reference_info()
        diagnostic["ball_spawn_source"].append(int(spawn_info["source"][0].item()))
        diagnostic["ball_spawn_reference_contact_frame"].append(
            int(spawn_info["reference_contact_frame"][0].item())
        )
        diagnostic["ball_spawn_reference_local"].append(_cpu(spawn_info["reference_local"]))
    else:
        diagnostic["ball_spawn_source"].append(-1)
        diagnostic["ball_spawn_reference_contact_frame"].append(-1)
        diagnostic["ball_spawn_reference_local"].append(np.full(3, np.nan, dtype=np.float32))
    diagnostic["style_phase"].append(int(command.style_phase_steps[0].item()))
    if hasattr(command, "style_phase_reference_info"):
        style_info = command.style_phase_reference_info()
        diagnostic["style_cycle_length"].append(int(style_info["cycle_length"][0].item()))
        diagnostic["style_source_first_frame"].append(int(style_info["source_first_frame"][0].item()))
        diagnostic["style_source_second_frame"].append(int(style_info["source_second_frame"][0].item()))
        diagnostic["style_seam_blend"].append(float(style_info["seam_blend"][0].item()))
        diagnostic["style_in_seam_bridge"].append(bool(style_info["in_seam_bridge"][0].item()))
    else:
        diagnostic["style_cycle_length"].append(int(command.motion_length[0].item()))
        diagnostic["style_source_first_frame"].append(int(command.style_phase_steps[0].item()))
        diagnostic["style_source_second_frame"].append(int(command.style_phase_steps[0].item()))
        diagnostic["style_seam_blend"].append(0.0)
        diagnostic["style_in_seam_bridge"].append(False)
    diagnostic["segment_idx"].append(int(command._locomotion_segment_idx[0].item()))
    task_state = getattr(command, "locomotion_task_state", None)
    diagnostic["task_state"].append(1 if task_state is None else int(task_state[0].item()))
    active_command = _active_locomotion_command(command)
    active_speed = active_command["speed"]
    active_heading = active_command["heading"]
    heading_valid = active_command["heading_valid"]
    diagnostic["command_mode"].append(str(active_command["mode"]))
    diagnostic["command_heading"].append(float(active_heading[0].item()))
    diagnostic["command_target_speed"].append(float(active_command["target_speed"][0].item()))
    diagnostic["command_target_heading"].append(float(active_command["target_heading"][0].item()))
    diagnostic["effective_command_speed"].append(float(active_speed[0].item()))
    diagnostic["effective_command_heading"].append(float(active_heading[0].item()))
    diagnostic["active_command_lin_vel_w"].append(_cpu(active_command["lin_vel_w"]))
    diagnostic["active_command_ang_vel_w"].append(_cpu(active_command["ang_vel_w"]))
    diagnostic["reference_twist_local"].append(_cpu(active_command["reference_twist_b"]))
    diagnostic["active_twist_local"].append(_cpu(active_command["twist_b"]))
    diagnostic["twist_blend_alpha"].append(float(active_command["blend_alpha"]))
    diagnostic["command_heading_valid"].append(bool(heading_valid[0].item()))

    pelvis_id = robot.body_names.index("pelvis")
    pelvis_quat = robot.data.body_quat_w[0, pelvis_id]
    pelvis_yaw_inv = quat_inv(yaw_quat(robot.data.body_quat_w[:, pelvis_id]))
    pelvis_lin_vel_b = quat_apply(pelvis_yaw_inv, robot.data.body_lin_vel_w[:, pelvis_id])
    pelvis_ang_vel_b = quat_apply(pelvis_yaw_inv, robot.data.body_ang_vel_w[:, pelvis_id])
    actual_twist_b = torch.stack(
        (pelvis_lin_vel_b[:, 0], pelvis_lin_vel_b[:, 1], pelvis_ang_vel_b[:, 2]), dim=-1
    )
    diagnostic["actual_twist_local"].append(_cpu(actual_twist_b))
    diagnostic["twist_local_error"].append(_cpu(active_command["twist_b"] - actual_twist_b))
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
        # A direct-latent policy has no one-to-one arm action. Keep arm-shaped
        # entries NaN and record the actual latent policy coordinates instead.
        arm_nan = torch.full(
            (policy_actions.shape[0], len(action_ids)), torch.nan,
            dtype=policy_actions.dtype, device=policy_actions.device,
        )
        latent_ids = getattr(action_term, "_latent_policy_action_ids", None)
        latent = (
            policy_actions[:, latent_ids]
            if isinstance(latent_ids, torch.Tensor)
            else torch.full((policy_actions.shape[0], 0), torch.nan, device=policy_actions.device)
        )
        diagnostic["policy_action"].append(_cpu(arm_nan))
        diagnostic["submitted_action"].append(_cpu(arm_nan))
        diagnostic["applied_action"].append(_cpu(arm_nan))
        diagnostic["upper_policy_latent"].append(_cpu(latent))
    else:
        diagnostic["policy_action"].append(_cpu(policy_actions[:, action_ids]))
        diagnostic["submitted_action"].append(_cpu(applied_actions[:, action_ids]))
        # Kept as a compatibility alias; use submitted_action/effective_action
        # in new analysis code.
        diagnostic["applied_action"].append(_cpu(applied_actions[:, action_ids]))
        diagnostic["upper_policy_latent"].append(np.empty(0, dtype=np.float32))
    diagnostic["trunk_reference_joint_pos"].append(_cpu(command.joint_pos[:, trunk_ids]))
    diagnostic["trunk_reference_joint_vel"].append(_cpu(command.joint_vel[:, trunk_ids]))
    diagnostic["trunk_actual_joint_pos"].append(_cpu(robot.data.joint_pos[:, trunk_ids]))
    diagnostic["trunk_actual_joint_vel"].append(_cpu(robot.data.joint_vel[:, trunk_ids]))
    diagnostic["trunk_policy_action"].append(_cpu(policy_actions[:, trunk_action_ids]))
    diagnostic["trunk_submitted_action"].append(_cpu(applied_actions[:, trunk_action_ids]))
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
    contact_settings = diagnostic["contact_settings"]
    contact = dribbling_contact_telemetry(
        base_env,
        command_name=contact_settings["command_name"],
        ball_sensor_name=contact_settings["ball_sensor_name"],
        all_body_cfg=contact_settings["all_body_cfg"],
        num_ankle_links=contact_settings["num_ankle_links"],
        contact_surface=contact_settings["contact_surface"],
        medial_y_min=contact_settings["medial_y_min"],
        cg_surface_gated=contact_settings["cg_surface_gated"],
        contact_force_threshold=contact_settings["contact_force_threshold"],
    )
    diagnostic["ball_contact"].append(bool(contact["has_contact"][0].item()))
    diagnostic["contact_force_magnitude"].append(float(contact["force_magnitude"][0].item()))
    diagnostic["contact_body_idx"].append(int(contact["contact_body_idx"][0].item()))
    diagnostic["contact_foot"].append(int(contact["contact_foot"][0].item()))
    diagnostic["contact_legal_ankle"].append(bool(contact["legal_ankle"][0].item()))
    diagnostic["contact_generic_instep"].append(bool(contact["generic_instep"][0].item()))
    diagnostic["contact_inside_instep"].append(bool(contact["inside_instep"][0].item()))
    diagnostic["contact_outside_instep"].append(bool(contact["outside_instep"][0].item()))
    diagnostic["contact_requested_surface_match"].append(bool(contact["requested_surface_match"][0].item()))
    diagnostic["contact_gentle"].append(bool(contact["gentle"][0].item()))
    diagnostic["contact_legal_touch"].append(bool(contact["legal_touch"][0].item()))
    diagnostic["contact_ball_offset_foot_yaw"].append(_cpu(contact["ball_offset_foot_yaw"]))
    diagnostic["contact_medial_offset"].append(float(contact["medial_offset"][0].item()))
    cg_labeled = getattr(command, "motion_has_dribble_cg_label", None)
    cg_ref_contact = getattr(command, "dribble_cg_contact_ref", None)
    cg_ref_foot = getattr(command, "dribble_cg_foot_ref", None)
    cg_ref_surface = getattr(command, "dribble_cg_surface_ref", None)
    has_cg_label = isinstance(cg_labeled, torch.Tensor) and bool(cg_labeled[0].item())
    ref_contact = bool(cg_ref_contact[0].item()) if isinstance(cg_ref_contact, torch.Tensor) else False
    ref_foot = int(cg_ref_foot[0].item()) if isinstance(cg_ref_foot, torch.Tensor) else -1
    ref_surface = int(cg_ref_surface[0].item()) if isinstance(cg_ref_surface, torch.Tensor) else -1
    actual_foot = int(contact["contact_foot"][0].item())
    diagnostic["cg_label_available"].append(has_cg_label)
    diagnostic["cg_reference_contact"].append(ref_contact)
    diagnostic["cg_reference_foot"].append(ref_foot)
    diagnostic["cg_reference_surface"].append(ref_surface)
    diagnostic["cg_contact_foot_match"].append(
        bool(contact["has_contact"][0].item()) and has_cg_label and ref_contact and actual_foot == ref_foot
    )
    s2_event_id = getattr(command, "s2_contact_event_id_ref", None)
    s2_event_frame = getattr(command, "s2_contact_event_frame_ref", None)
    s2_expected_foot = getattr(command, "s2_contact_event_foot_ref", None)
    s2_expected_side = getattr(command, "s2_contact_event_side_ref", None)
    s2_window = getattr(command, "s2_contact_window_ref", None)
    if isinstance(s2_event_id, torch.Tensor) and hasattr(command, "s2_contact_reference_foot_pose_w"):
        reference_foot_pos, reference_foot_yaw = command.s2_contact_reference_foot_pose_w()
        ball_offset_reference_foot = quat_apply_inverse(
            reference_foot_yaw,
            soccer_ball.data.root_pos_w[:, :3] - reference_foot_pos,
        )
        side_deadzone = 0.04
        side_max = 0.16
        forward = ball_offset_reference_foot[:, 0]
        lateral = ball_offset_reference_foot[:, 1]
        forward_error = torch.relu(-0.06 - forward) + torch.relu(forward - 0.14)
        left_error = torch.relu(side_deadzone - lateral) + torch.relu(lateral - side_max)
        right_error = torch.relu(-side_max - lateral) + torch.relu(lateral + side_deadzone)
        side_error = torch.where(s2_expected_side == 0, left_error, right_error)
        region_distance = torch.sqrt(forward_error.square() + side_error.square())
        diagnostic["s2_contact_window"].append(bool(s2_window[0].item()))
        diagnostic["s2_contact_event_id"].append(int(s2_event_id[0].item()))
        diagnostic["s2_contact_event_frame"].append(int(s2_event_frame[0].item()))
        diagnostic["s2_expected_foot"].append(int(s2_expected_foot[0].item()))
        diagnostic["s2_expected_side"].append(int(s2_expected_side[0].item()))
        diagnostic["s2_reference_foot_pos_w"].append(_cpu(reference_foot_pos))
        reference_yaw = _world_quat_to_rpy(reference_foot_yaw)[:, 2]
        diagnostic["s2_reference_foot_yaw"].append(float(reference_yaw[0].item()))
        diagnostic["s2_ball_offset_reference_foot"].append(_cpu(ball_offset_reference_foot))
        diagnostic["s2_target_region_distance"].append(float(region_distance[0].item()))
    else:
        diagnostic["s2_contact_window"].append(False)
        diagnostic["s2_contact_event_id"].append(-1)
        diagnostic["s2_contact_event_frame"].append(-1)
        diagnostic["s2_expected_foot"].append(-1)
        diagnostic["s2_expected_side"].append(-1)
        diagnostic["s2_reference_foot_pos_w"].append(np.full(3, np.nan, dtype=np.float32))
        diagnostic["s2_reference_foot_yaw"].append(np.nan)
        diagnostic["s2_ball_offset_reference_foot"].append(np.full(3, np.nan, dtype=np.float32))
        diagnostic["s2_target_region_distance"].append(np.nan)
    flow_available = getattr(command, "motion_has_dribble_cg_flow_label", None)
    flow_valid = getattr(command, "dribble_cg_flow_valid_ref", None)
    flow_anchor = getattr(command, "dribble_cg_flow_anchor_frame_ref", None)
    flow_direction = getattr(command, "dribble_cg_flow_dir_local_ref", None)
    flow_distance = getattr(command, "dribble_cg_flow_distance_ref", None)
    flow_duration = getattr(command, "dribble_cg_flow_duration_ref", None)
    flow_available_0 = isinstance(flow_available, torch.Tensor) and bool(flow_available[0].item())
    flow_valid_0 = isinstance(flow_valid, torch.Tensor) and bool(flow_valid[0].item())
    flow_distance_0 = float(flow_distance[0].item()) if isinstance(flow_distance, torch.Tensor) else -1.0
    flow_duration_0 = float(flow_duration[0].item()) if isinstance(flow_duration, torch.Tensor) else -1.0
    diagnostic["cg_flow_label_available"].append(flow_available_0)
    diagnostic["cg_flow_valid"].append(flow_valid_0)
    diagnostic["cg_flow_anchor_frame"].append(
        int(flow_anchor[0].item()) if isinstance(flow_anchor, torch.Tensor) else -1
    )
    diagnostic["cg_flow_direction_local"].append(
        _cpu(flow_direction) if isinstance(flow_direction, torch.Tensor) else np.full(2, np.nan, dtype=np.float32)
    )
    diagnostic["cg_flow_distance"].append(flow_distance_0)
    diagnostic["cg_flow_duration"].append(flow_duration_0)
    diagnostic["cg_flow_nominal_speed"].append(
        flow_distance_0 / flow_duration_0 if flow_valid_0 and flow_duration_0 > 0.0 else np.nan
    )
    diagnostic["cg_flow_latched"].append(False)
    diagnostic["cg_flow_release_active"].append(False)
    diagnostic["cg_flow_target_direction_world"].append(np.full(2, np.nan, dtype=np.float32))
    diagnostic["cg_flow_parallel_speed"].append(np.nan)
    diagnostic["cg_flow_lateral_speed"].append(np.nan)
    diagnostic["cg_flow_release_reward"].append(np.nan)
    diagnostic["cg_flow_progress"].append(np.nan)
    diagnostic["cg_flow_progress_rate"].append(np.nan)
    diagnostic["cg_flow_lateral_offset"].append(np.nan)
    diagnostic["cg_flow_progress_reward"].append(np.nan)
    cg_gate_pass = not (contact_settings["cg_gated"] and has_cg_label) or ref_contact
    diagnostic["contact_cg_gate_pass"].append(cg_gate_pass)
    diagnostic["contact_cg_eligible"].append(
        bool(contact["legal_touch"][0].item()) and cg_gate_pass
    )
    ball_vel_xy = soccer_ball.data.root_lin_vel_w[0, :2]
    diagnostic["ball_xy_speed"].append(float(torch.norm(ball_vel_xy).item()))
    if bool(heading_valid[0].item()):
        command_dir = torch.stack((torch.cos(active_heading[0]), torch.sin(active_heading[0])))
        ball_forward_speed = float(torch.dot(ball_vel_xy, command_dir).item())
    else:
        ball_forward_speed = np.nan
    diagnostic["ball_command_forward_speed"].append(ball_forward_speed)
    diagnostic["pelvis_xy_speed"].append(float(torch.norm(robot.data.body_lin_vel_w[0, pelvis_id, :2]).item()))
    foot_ref_ids = [command.cfg.body_names.index(name) for name in _FOOT_DIAGNOSTIC_BODY_NAMES]
    foot_error = torch.norm(
        command.robot_body_pos_w[0, foot_ref_ids] - command.body_pos_relative_w[0, foot_ref_ids], dim=-1
    ).mean()
    diagnostic["foot_reference_position_error"].append(float(foot_error.item()))
    is_local_twist = str(getattr(command.cfg, "locomotion_command_frame", "world")) == "pelvis_local"
    if bool(heading_valid[0].item()) and not is_local_twist:
        heading_error = torch.atan2(
            torch.sin(active_heading[0] - diagnostic["pelvis_yaw"][-1]),
            torch.cos(active_heading[0] - diagnostic["pelvis_yaw"][-1]),
        )
        diagnostic["heading_error"].append(float(heading_error.item()))
    else:
        diagnostic["heading_error"].append(np.nan)
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


def _append_upper_body_manifold_diagnostic(diagnostic: dict, env, dones: torch.Tensor) -> None:
    """Record post-step arm projection and trunk target/actuator telemetry."""
    base_env = _resolve_base_env(env)
    action_term = base_env.action_manager.get_term("joint_pos")
    snapshot = getattr(action_term, "diagnostic_snapshot", {})
    snapshot_available = isinstance(snapshot, dict) and bool(snapshot)
    post_step_state_valid = not bool(dones[0].item())

    def _env0(name: str, fallback_shape: tuple[int, ...]) -> np.ndarray:
        value = snapshot.get(name, getattr(action_term, name, None)) if snapshot_available else getattr(action_term, name, None)
        if not isinstance(value, torch.Tensor):
            return np.full(fallback_shape, np.nan, dtype=np.float32)
        return value[0].detach().cpu().numpy().copy()

    def _tensor(name: str):
        if snapshot_available and isinstance(snapshot.get(name), torch.Tensor):
            return snapshot[name]
        return getattr(action_term, name, None)

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
    diagnostic["manifold_joint_limited_upper_target"].append(
        _env0("manifold_joint_limited_upper_target", (upper_dim,))
    )
    diagnostic["manifold_executed_upper_target"].append(
        _env0("manifold_executed_upper_target", (upper_dim,))
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
    diagnostic["manifold_joint_limit_clamp_fraction"].append(
        float(_env0("manifold_joint_limit_clamp_fraction", ()).item())
    )
    diagnostic["manifold_filter_lag"].append(
        float(_env0("manifold_filter_lag", ()).item())
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
    trunk_full_action_ids = diagnostic["trunk_full_action_ids"]
    effective_raw_actions = _tensor("effective_raw_actions")
    if isinstance(effective_raw_actions, torch.Tensor):
        diagnostic["effective_action"].append(
            effective_raw_actions[0, diagnostic["action_ids"]].detach().cpu().numpy().copy()
        )
        diagnostic["trunk_effective_action"].append(
            effective_raw_actions[0, trunk_full_action_ids].detach().cpu().numpy().copy()
        )
    else:
        diagnostic["effective_action"].append(
            np.full((len(diagnostic["action_ids"]),), np.nan, dtype=np.float32)
        )
        diagnostic["trunk_effective_action"].append(
            np.full((len(_TRUNK_DIAGNOSTIC_JOINT_NAMES),), np.nan, dtype=np.float32)
        )
    diagnostic["action_snapshot_available"].append(snapshot_available)
    diagnostic["post_step_state_valid"].append(post_step_state_valid)
    processed_actions = _tensor("executed_joint_targets")
    if processed_actions is None:
        processed_actions = getattr(action_term, "_processed_actions", None)

    if isinstance(processed_actions, torch.Tensor):
        trunk_target = processed_actions[0, trunk_full_action_ids]
    else:
        trunk_target = torch.full(
            (len(_TRUNK_DIAGNOSTIC_JOINT_NAMES),),
            float("nan"),
            dtype=robot.data.joint_pos.dtype,
            device=base_env.device,
    )
    trunk_post_step_pos = robot.data.joint_pos[0, trunk_ids]
    # This row's target was formed from the reference recorded before
    # env.step(). The command style phase may already have advanced now.
    trunk_reference = torch.as_tensor(
        diagnostic["trunk_reference_joint_pos"][-1],
        dtype=trunk_target.dtype,
        device=base_env.device,
    )
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
    if not post_step_state_valid:
        # Isaac Lab may already have reset the articulation. The saved action
        # target still belongs to this transition, but live body/joint/torque
        # data now belongs to the next episode and must not be mixed in.
        trunk_post_step_pos = torch.full_like(trunk_post_step_pos, torch.nan)
        trunk_actual_limit_margin = torch.full_like(trunk_actual_limit_margin, torch.nan)
        trunk_computed_torque = torch.full_like(trunk_computed_torque, torch.nan)
        trunk_applied_torque = torch.full_like(trunk_applied_torque, torch.nan)
        computed_effort_utilization = torch.full_like(computed_effort_utilization, torch.nan)
        effort_utilization = torch.full_like(effort_utilization, torch.nan)

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
    effort_saturated = (
        effort_utilization >= 0.98
        if post_step_state_valid
        else torch.full_like(effort_utilization, torch.nan)
    )
    diagnostic["trunk_effort_saturated"].append(_trunk_cpu(effort_saturated))


def _complete_cg_flow_diagnostic(diagnostic: dict, env) -> None:
    """Attach flow-reward telemetry produced by the just-completed transition."""
    base_env = _resolve_base_env(env)
    telemetry = getattr(base_env, "_dribbling_cg_flow_telemetry", None)
    if not isinstance(telemetry, dict):
        return

    def _scalar(name: str) -> float:
        value = telemetry.get(name)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return np.nan
        return float(value[0].item())

    def _boolean(name: str) -> bool:
        value = telemetry.get(name)
        return isinstance(value, torch.Tensor) and value.numel() > 0 and bool(value[0].item())

    direction = telemetry.get("target_direction_world")
    diagnostic["cg_flow_latched"][-1] = _boolean("latched")
    diagnostic["cg_flow_release_active"][-1] = _boolean("release_active")
    if isinstance(direction, torch.Tensor) and direction.numel() >= 2:
        diagnostic["cg_flow_target_direction_world"][-1] = (
            direction[0].detach().cpu().numpy().copy()
        )
    diagnostic["cg_flow_parallel_speed"][-1] = _scalar("parallel_speed")
    diagnostic["cg_flow_lateral_speed"][-1] = _scalar("lateral_speed")
    diagnostic["cg_flow_release_reward"][-1] = _scalar("release_reward")
    diagnostic["cg_flow_progress"][-1] = _scalar("progress")
    diagnostic["cg_flow_progress_rate"][-1] = _scalar("progress_rate")
    diagnostic["cg_flow_lateral_offset"][-1] = _scalar("lateral_offset")
    diagnostic["cg_flow_progress_reward"][-1] = _scalar("progress_reward")


def _save_diagnostic(diagnostic: dict) -> None:
    """Persist the trace in a self-describing NumPy archive."""
    if not diagnostic["step"]:
        print("[WARN] Diagnostic requested but no samples were recorded.")
        return
    metadata_keys = {
        "path", "stride", "joint_ids", "joint_names", "action_ids", "reward_term_names", "task_state_names",
        "ball_spawn_source_names",
        "trunk_joint_ids", "trunk_joint_names", "trunk_action_ids", "trunk_full_action_ids", "trunk_body_ids",
        "trunk_reference_body_ids", "trunk_body_names",
        "constraint_group", "constraint_margin", "constraint_joint_names", "constraint_groups",
        "constraint_margins", "waist_roll_stiffness_scale", "waist_roll_damping_scale",
        "direct_upper_body_latent", "schema_version", "sample_timing", "action_value_semantics",
        "contact_settings", "contact_region_frame", "contact_body_names", "contact_surface",
        "contact_force_threshold", "contact_cg_gated", "contact_cg_surface_gated", "locomotion_command_frame",
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
    arrays["task_state_names"] = diagnostic["task_state_names"]
    arrays["ball_spawn_source_names"] = diagnostic["ball_spawn_source_names"]
    arrays["constraint_group"] = np.asarray(diagnostic["constraint_group"])
    arrays["constraint_margin"] = np.asarray(diagnostic["constraint_margin"])
    arrays["constraint_joint_names"] = diagnostic["constraint_joint_names"]
    arrays["constraint_groups"] = diagnostic["constraint_groups"]
    arrays["constraint_margins"] = diagnostic["constraint_margins"]
    arrays["waist_roll_stiffness_scale"] = np.asarray(diagnostic["waist_roll_stiffness_scale"])
    arrays["waist_roll_damping_scale"] = np.asarray(diagnostic["waist_roll_damping_scale"])
    arrays["direct_upper_body_latent"] = np.asarray(diagnostic["direct_upper_body_latent"])
    arrays["schema_version"] = np.asarray(diagnostic["schema_version"])
    arrays["sample_timing"] = np.asarray(diagnostic["sample_timing"])
    arrays["action_value_semantics"] = diagnostic["action_value_semantics"]
    arrays["contact_region_frame"] = np.asarray(diagnostic["contact_region_frame"])
    arrays["contact_body_names"] = diagnostic["contact_body_names"]
    arrays["contact_surface"] = np.asarray(diagnostic["contact_surface"])
    arrays["contact_force_threshold"] = np.asarray(diagnostic["contact_force_threshold"])
    arrays["contact_cg_gated"] = np.asarray(diagnostic["contact_cg_gated"])
    arrays["contact_cg_surface_gated"] = np.asarray(diagnostic["contact_cg_surface_gated"])
    arrays["locomotion_command_frame"] = np.asarray(diagnostic["locomotion_command_frame"])
    np.savez_compressed(diagnostic["path"], **arrays)

    def _finite_mean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else np.nan

    contact_rate = float(np.mean(arrays["ball_contact"]))
    ball_distance = float(np.mean(arrays["ball_pelvis_xy_distance"]))
    ball_speed = float(np.mean(arrays["ball_xy_speed"]))
    ball_forward_speed = _finite_mean(arrays["ball_command_forward_speed"])
    pelvis_speed = float(np.mean(arrays["pelvis_xy_speed"]))
    foot_error = float(np.mean(arrays["foot_reference_position_error"]))
    heading_error = _finite_mean(np.abs(arrays["heading_error"]))
    local_twist_abs_error = np.abs(arrays["twist_local_error"])
    local_vx_error = _finite_mean(local_twist_abs_error[:, 0])
    local_vy_error = _finite_mean(local_twist_abs_error[:, 1])
    local_wz_error = _finite_mean(local_twist_abs_error[:, 2])
    final_twist_blend_alpha = float(arrays["twist_blend_alpha"][-1])
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
    trunk_action_delta = np.diff(arrays["trunk_effective_action"], axis=0)
    finite_trunk_action_delta = np.linalg.norm(trunk_action_delta, axis=1)
    finite_trunk_action_delta = finite_trunk_action_delta[np.isfinite(finite_trunk_action_delta)]
    trunk_action_step = (
        float(np.mean(finite_trunk_action_delta)) if finite_trunk_action_delta.size else np.nan
    )
    action_snapshot_coverage = float(np.mean(arrays["action_snapshot_available"].astype(np.float32)))
    post_step_state_coverage = float(np.mean(arrays["post_step_state_valid"].astype(np.float32)))
    contact_mask = arrays["ball_contact"].astype(bool)
    legal_touch_rate = (
        float(np.mean(arrays["contact_legal_touch"][contact_mask])) if np.any(contact_mask) else np.nan
    )
    cg_eligible_touch_rate = (
        float(np.mean(arrays["contact_cg_eligible"][contact_mask])) if np.any(contact_mask) else np.nan
    )
    generic_instep_rate = (
        float(np.mean(arrays["contact_generic_instep"][contact_mask])) if np.any(contact_mask) else np.nan
    )
    cg_expected_touch_mask = arrays["cg_label_available"].astype(bool) & arrays["cg_reference_contact"].astype(bool)
    cg_foot_match_rate = (
        float(np.mean(arrays["cg_contact_foot_match"][cg_expected_touch_mask]))
        if np.any(cg_expected_touch_mask)
        else np.nan
    )
    s2_window_mask = arrays["s2_contact_window"].astype(bool)
    s2_region_distance = (
        _finite_mean(arrays["s2_target_region_distance"][s2_window_mask])
        if np.any(s2_window_mask)
        else np.nan
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
        f"contact_rate={contact_rate:.3f}  ball_pelvis_xy={ball_distance:.3f} m  "
        f"legal_touch={legal_touch_rate:.3f}  cg_eligible={cg_eligible_touch_rate:.3f}  "
        f"generic_instep={generic_instep_rate:.3f}  "
        f"cg_foot_match={cg_foot_match_rate:.3f}  "
        f"ball_xy_speed={ball_speed:.3f} m/s  ball_cmd_speed={ball_forward_speed:.3f} m/s  "
        f"pelvis_xy_speed={pelvis_speed:.3f} m/s  "
        f"local_twist_abs_err=(vx={local_vx_error:.3f}, vy={local_vy_error:.3f}, wz={local_wz_error:.3f})  "
        f"twist_blend_alpha={final_twist_blend_alpha:.3f}  "
        f"task_states=({task_state_summary})  stop_settled={stop_settle_rate:.3f}  "
        f"stop_successes={stop_successes}  "
        f"stop_ball_speed={stop_ball_speed:.3f} m/s  stop_pelvis_speed={stop_pelvis_speed:.3f} m/s  "
        f"stop_pelvis_w={stop_pelvis_angular_speed:.3f} rad/s  "
        f"foot_ref_err={foot_error:.3f} m  mean_abs_heading_err={heading_error:.3f} rad  "
        f"arm_joint_err={arm_error:.3f} rad  waist_joint_err={trunk_error:.3f} rad  "
        f"waist_effective_action_step={trunk_action_step:.3f}  action_snapshot={action_snapshot_coverage:.3f}  "
        f"post_state_valid={post_step_state_coverage:.3f}  "
        f"torso_rel_tilt={torso_rel_tilt:.3f} rad  "
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
    if np.any(s2_window_mask):
        print(
            "[INFO] S2 contact labels: "
            f"window_samples={int(np.count_nonzero(s2_window_mask))}  "
            f"mean_target_region_distance={s2_region_distance:.3f} m  "
            "side_ids=(left=0, right=1, dead/unknown=-1)"
        )


def _load_local_twist_command_plan(plan_path: str) -> tuple[list[tuple[float, float, float, float]], bool, bool]:
    """Read and validate a reproducible pelvis-local Cartesian command plan."""
    path = pathlib.Path(plan_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Locomotion command plan not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in locomotion command plan {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Locomotion command plan must be a JSON object.")
    frame = str(payload.get("frame", "pelvis_local")).strip().lower()
    if frame != "pelvis_local":
        raise ValueError(
            f"Locomotion command plan frame must be 'pelvis_local', got {payload.get('frame')!r}."
        )
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Locomotion command plan needs a non-empty 'segments' array.")
    segments: list[tuple[float, float, float, float]] = []
    required = ("vx", "vy", "wz", "duration_s")
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"Plan segment {index} must be an object.")
        missing = [field for field in required if field not in raw_segment]
        if missing:
            raise ValueError(f"Plan segment {index} is missing {missing}.")
        values = tuple(float(raw_segment[field]) for field in required)
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"Plan segment {index} contains a non-finite value.")
        if values[3] <= 0.0:
            raise ValueError(f"Plan segment {index} duration_s must be positive.")
        segments.append(values)
    return segments, bool(payload.get("loop", True)), bool(payload.get("reset_on_end", False))


def _apply_play_locomotion_command(env, args_cli) -> bool:
    """Apply a JSON local-twist plan, polar segments, or one direct twist."""
    plan_set = args_cli.locomotion_cmd_plan is not None
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
        args_cli.locomotion_cmd_wz,
    )
    if not plan_set and not polar_set and all(v is None for v in cartesian_fields):
        return False

    base_env = _resolve_base_env(env)
    cmd = base_env.command_manager.get_term("motion")
    if not hasattr(cmd, "set_locomotion_manual_command"):
        print("[WARN] Task motion command has no manual locomotion API; --locomotion_cmd_* ignored.")
        return False

    if plan_set:
        if polar_set or any(value is not None for value in cartesian_fields):
            raise ValueError(
                "--locomotion_cmd_plan cannot be combined with any other --locomotion_cmd_* value."
            )
        if str(getattr(cmd.cfg, "locomotion_command_frame", "world")) != "pelvis_local":
            raise ValueError("--locomotion_cmd_plan requires a pelvis-local locomotion task.")
        if not hasattr(cmd, "set_locomotion_local_twist_sequence"):
            raise RuntimeError("Task motion command does not support local-twist sequences.")
        segments, loop, reset_on_end = _load_local_twist_command_plan(args_cli.locomotion_cmd_plan)
        cmd.set_locomotion_local_twist_sequence(
            segments,
            hold_last=not loop,
            reset_on_end=reset_on_end,
        )
        print(f"[INFO] Local-twist command plan: {args_cli.locomotion_cmd_plan}")
        for index, (vx, vy, wz, duration_s) in enumerate(segments):
            print(
                f"  [{index + 1}] vx={vx:.3f} m/s  vy={vy:.3f} m/s  "
                f"wz={wz:.3f} rad/s  duration={duration_s:.2f} s"
            )
        print("  final segment -> environment reset -> segment 1" if reset_on_end else (
            "  looping: final segment -> first segment" if loop else "  holding: final segment remains active"
        ))
        return True

    if polar_set and hasattr(cmd, "set_locomotion_polar_command"):
        speeds = args_cli.locomotion_cmd_speed
        headings = args_cli.locomotion_cmd_heading
        durations = args_cli.locomotion_cmd_duration
        task_states = args_cli.locomotion_task_state

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
        is_local_twist = str(getattr(cmd.cfg, "locomotion_command_frame", "world")) == "pelvis_local"
        if is_local_twist:
            if task_states is not None:
                print("[WARN] local-twist tasks ignore --locomotion_task_state and keep DRIBBLE compatibility state.")
            task_states = ["dribble"] * n
        elif task_states is None:
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
    wz_values = args_cli.locomotion_cmd_wz
    if wz_values is None or len(wz_values) == 0:
        wz = cur_ang[2]
    elif len(wz_values) == 1:
        wz = float(wz_values[0])
    else:
        raise ValueError("Cartesian manual control accepts at most one --locomotion_cmd_wz value.")
    ang = [
        cur_ang[0] if args_cli.locomotion_cmd_wx is None else args_cli.locomotion_cmd_wx,
        cur_ang[1] if args_cli.locomotion_cmd_wy is None else args_cli.locomotion_cmd_wy,
        wz,
    ]
    cartesian_task_state = None
    if args_cli.locomotion_task_state is not None:
        if len(args_cli.locomotion_task_state) != 1:
            raise ValueError("Cartesian manual control accepts exactly one --locomotion_task_state.")
        cartesian_task_state = args_cli.locomotion_task_state[0]
    is_local_twist = str(getattr(cmd.cfg, "locomotion_command_frame", "world")) == "pelvis_local"
    if is_local_twist and hasattr(cmd, "set_locomotion_local_twist_command"):
        if cartesian_task_state is not None:
            print("[WARN] local-twist tasks keep the compatibility state at DRIBBLE; --locomotion_task_state is ignored.")
        cmd.set_locomotion_local_twist_command(vx=lin[0], vy=lin[1], wz=ang[2])
        frame = "current pelvis local frame"
    else:
        cmd.set_locomotion_manual_command(lin_vel=lin, ang_vel=ang, task_state=cartesian_task_state)
        frame = "task +X/+Y/+Z frame"
    print(
        f"[INFO] Locomotion manual cmd: lin_vel={lin} m/s  ang_vel={ang} rad/s  ({frame})"
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
        if hasattr(cmd, "style_phase_reference_info"):
            style_info = cmd.style_phase_reference_info()
            cycle_len = int(style_info["cycle_length"][i].item())
            source_first = int(style_info["source_first_frame"][i].item())
            source_second = int(style_info["source_second_frame"][i].item())
            seam_blend = float(style_info["seam_blend"][i].item())
            if bool(style_info["in_seam_bridge"][i].item()):
                lines.append(
                    f"Style seam: {source_first}->{source_second}  blend={seam_blend:.2f}"
                )
            else:
                lines.append(f"Style frame: {source_first}  phase {t}/{max(cycle_len - 1, 0)}")
        else:
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

        force_xy = float(
            soccer_ball_contact_force_magnitude(base_env, _BALL_SENSOR_NAME)[i].item()
        )
        sim_touch = force_xy > _CONTACT_FORCE_THRESHOLD

        ankle_parts = [f"contact={'YES' if sim_touch else 'NO'}"]
        if hasattr(cmd, "motion_has_dribble_cg_label") and bool(cmd.motion_has_dribble_cg_label[i].item()):
            ref_contact = bool(cmd.dribble_cg_contact_ref[i].item())
            match = "match" if ref_contact == sim_touch else "MISMATCH"
            ankle_parts.append(f"demo={'YES' if ref_contact else 'NO'}")
            ankle_parts.append(match)
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
    if args_cli.show_s2_contact_regions and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.debug_vis = True
        setattr(env_cfg.commands.motion, "dribble_cg_s2_debug_regions_only", True)

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
                _append_upper_body_manifold_diagnostic(diagnostic, env, dones)
                _complete_cg_flow_diagnostic(diagnostic, env)
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
