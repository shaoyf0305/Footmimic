"""Evaluate per-reference motion learning quality and optionally filter bad clips.

Runs one reference .npz at a time with a trained checkpoint, aggregates tracking /
termination metrics, writes a JSON + CSV report, and can symlink passing motions
into a filtered directory for re-training.

Example (dribbling stage-2, dribble-distance references):
  python scripts/rsl_rl/eval_references.py \\
    --task Tracking-CG-G1-Dribbling-RNN-unified-control \\
    --experiment_name g1_dribbling \\
    --motion_path motions/dribble-distance \\
    --load_run 2026-06-09_03-45-57_resumed_dribble \\
    --checkpoint model_84000.pt \\
    --num_rollouts 5 \\
    --headless

  # Default: same failure terminations as play_multi (ball_lost, ee_body_pos, …).
  # Tracking-only full-clip metrics: add --disable_training_terminations

Filter only (no Isaac Sim):
  python scripts/rsl_rl/filter_motions_from_eval.py \\
    --report logs/rsl_rl/g1_dribbling/<run>/eval_references/report.json \\
    --output_dir motions/dribble_filtered
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate per-reference motion learning quality.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Gym task id, e.g. Tracking-CG-G1-Dribbling-RNN-unified-control.",
)
parser.add_argument(
    "--motion_path",
    type=str,
    required=True,
    help="Directory (or single .npz) with reference motions to evaluate.",
)
parser.add_argument(
    "--motion_glob",
    type=str,
    default="*.npz",
    help="When motion_path is a directory, only evaluate files matching this glob (default: *.npz).",
)
parser.add_argument(
    "--num_rollouts",
    type=int,
    default=3,
    help="Independent rollouts per reference (sequential, num_envs=1).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for eval report (default: <checkpoint_dir>/eval_references).",
)
parser.add_argument(
    "--write_filtered_dir",
    type=str,
    default=None,
    help="If set, symlink passing motions into this directory.",
)
parser.add_argument(
    "--min_completion_rate",
    type=float,
    default=0.67,
    help="Pass if this fraction of rollouts finish the clip without failure terms.",
)
parser.add_argument(
    "--max_joint_pos_error",
    type=float,
    default=1.0,
    help="Pass if mean L2 joint-position tracking error stays below this value.",
)
parser.add_argument(
    "--max_anchor_pos_error",
    type=float,
    default=0.35,
    help="Pass if mean anchor position error stays below this value.",
)
parser.add_argument(
    "--min_ball_contact_ratio",
    type=float,
    default=0.0,
    help="For dribbling tasks: minimum ankle-ball contact ratio (0 disables).",
)
parser.add_argument(
    "--min_quality_score",
    type=float,
    default=0.35,
    help="Pass if composite quality score stays above this value.",
)
parser.add_argument(
    "--eval_grace_steps",
    type=int,
    default=0,
    help="Ignore failure terminations for the first N steps (0 = same as play_multi).",
)
parser.add_argument(
    "--disable_training_terminations",
    action="store_true",
    default=False,
    help="Turn off failure terminations; only measure full-clip tracking/contact (failure_rate is N/A).",
)
parser.add_argument(
    "--recreate_env_per_rollout",
    action="store_true",
    default=False,
    help="Recreate Isaac env every rollout (very slow; may look hung at 'Setting seed: N').",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for the eval environment.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import soccer.tasks  # noqa: F401
from soccer.tasks.tracking.mdp.rewards_dribbling import soccer_ball_contact_force_magnitude
from soccer.utils.checkpoint_loading import load_checkpoint_with_obs_expand

TRACKING_METRIC_KEYS = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_body_pos",
    "error_body_rot",
    "error_joint_pos",
    "error_joint_vel",
)

_BALL_SENSOR_NAME = "soccer_ball_contact"
_CONTACT_FORCE_THRESHOLD = 1.0


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def motion_frame_count(motion_file: str) -> int:
    data = np.load(motion_file)
    if "joint_pos" in data:
        return int(data["joint_pos"].shape[0])
    for key in data.files:
        arr = data[key]
        if hasattr(arr, "shape") and len(arr.shape) >= 1:
            return int(arr.shape[0])
    raise ValueError(f"Could not infer frame count from {motion_file}")


def get_motion_files(motion_path: str, motion_glob: str) -> list[str]:
    if os.path.isfile(motion_path):
        return [motion_path]
    if os.path.isdir(motion_path):
        pattern = os.path.join(motion_path, motion_glob)
        motion_files = sorted(glob.glob(pattern))
        if not motion_files:
            raise ValueError(f"No files matching {pattern}")
        return motion_files
    raise ValueError(f"Invalid motion_path: {motion_path}")


def _resolve_base_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    if hasattr(base, "unwrapped"):
        base = base.unwrapped
    return base


def _failure_terms_active(base_env, env_idx: int = 0) -> list[str]:
    tm = getattr(base_env, "termination_manager", None)
    if tm is None:
        return []
    active: list[str] = []
    for name in tm.active_terms:
        try:
            if tm.get_term_cfg(name).time_out:
                continue
            if bool(tm.get_term(name)[env_idx].item()):
                active.append(name)
        except Exception:
            continue
    return active


def _zero_range_dict() -> dict[str, tuple[float, float]]:
    return {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}


def configure_eval_env(env_cfg, motion_files: list[str]) -> int:
    """Disable training randomization for repeatable per-reference scoring."""
    zero = _zero_range_dict()
    env_cfg.scene.num_envs = 1
    step_s = float(env_cfg.decimation) * float(env_cfg.sim.dt)
    n_frames = max(motion_frame_count(f) for f in motion_files)
    env_cfg.episode_length_s = n_frames * step_s + 2.0

    motion_cmd = env_cfg.commands.motion
    motion_cmd.pose_range = zero
    motion_cmd.velocity_range = zero
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_strategy = "uniform"

    if hasattr(motion_cmd, "curve_offset_range") and isinstance(motion_cmd.curve_offset_range, dict):
        height = motion_cmd.curve_offset_range.get("height", 0.11)
        motion_cmd.curve_offset_range = {
            "radius": (0.0, 0.0),
            "lateral_spawn_jitter": 0.0,
            "height": height,
        }

    if hasattr(motion_cmd, "enable_soccer_ball_init_vel"):
        motion_cmd.enable_soccer_ball_init_vel = False

    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.observations, "critic"):
        env_cfg.observations.critic.enable_corruption = False

    # Match play_multi: only failure terms end the episode, not a short time_out.
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    return n_frames


def disable_failure_terminations(env_cfg) -> None:
    """Eval runs the full clip; training early-exit terms distort quality measurement."""
    if not hasattr(env_cfg, "terminations"):
        return
    for name in (
        "anchor_pos_z",
        "anchor_ori",
        "ee_body_pos",
        "ball_lost",
        "dribbling_no_contact",
    ):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)


def _reset_episode_state(base_env, env_ids: torch.Tensor) -> None:
    if hasattr(base_env, "episode_length_buf"):
        base_env.episode_length_buf[env_ids] = 0
    if hasattr(base_env, "reset_buf"):
        base_env.reset_buf[env_ids] = 0
    for mgr_name in ("termination_manager", "reward_manager", "observation_manager"):
        mgr = getattr(base_env, mgr_name, None)
        if mgr is None:
            continue
        try:
            mgr.reset(env_ids.cpu().tolist())
        except Exception:
            pass


def _detach_asset_tensors(asset) -> None:
    if asset is None:
        return
    data = getattr(asset, "data", None)
    if data is None:
        return
    for name in dir(data):
        if name.startswith("__"):
            continue
        try:
            val = getattr(data, name)
        except (AttributeError, RuntimeError):
            continue
        if isinstance(val, torch.Tensor):
            try:
                setattr(data, name, val.clone())
            except (AttributeError, RuntimeError):
                pass


def _detach_cmd_tensors(cmd) -> None:
    """Clone buffers so soft-reset can write after inference_mode rollouts."""
    for name in (
        "soccer_ball_pos",
        "target_point_pos",
        "target_destination_pos",
        "initial_target_point_pos",
        "motion_idx",
        "motion_length",
        "time_steps",
        "blind_distance_min",
        "blind_distance_max",
        "is_in_blind_zone",
        "last_visible_target_point_base",
        "curve_radius_offset",
    ):
        if hasattr(cmd, name):
            val = getattr(cmd, name)
            if isinstance(val, torch.Tensor):
                setattr(cmd, name, val.clone())


def _start_motion_rollout(base_env, motion_idx: int) -> None:
    """Soft-reset to the start of one reference clip (no gym.make / env.reset)."""
    cmd = base_env.command_manager.get_term("motion")
    device = cmd.device
    _detach_cmd_tensors(cmd)
    _detach_asset_tensors(getattr(cmd, "soccer_ball", None))
    _detach_asset_tensors(getattr(cmd, "robot", None))
    env_ids = cmd._to_env_id_tensor([0])

    cmd.motion_idx[env_ids] = motion_idx
    cmd.motion_length[env_ids] = cmd.motion.file_lengths[motion_idx]
    cmd.time_steps[env_ids] = 0

    cmd._sample_soccer_offset(env_ids)
    cmd._compute_soccer_ball_positions(env_ids)
    cmd._update_soccer_ball(env_ids)
    cmd._update_target_points(env_ids)
    cmd._update_destination_points(env_ids)

    blind_min_low, blind_min_high = cmd.cfg.blind_distance_min_range
    blind_max_low, blind_max_high = cmd.cfg.blind_distance_max_range
    cmd.blind_distance_min[env_ids] = 0.5 * (blind_min_low + blind_min_high)
    cmd.blind_distance_max[env_ids] = 0.5 * (blind_max_low + blind_max_high)
    cmd.is_in_blind_zone[env_ids] = False
    cmd.last_visible_target_point_base[env_ids] = 0.0

    # Eval config zeros pose/velocity/joint noise — write reference frame 0 directly.
    joint_pos = cmd.joint_pos.clone()
    joint_vel = cmd.joint_vel.clone()
    root_pos = cmd.body_pos_w[:, 0].clone()
    root_ori = cmd.body_quat_w[:, 0].clone()
    root_lin_vel = cmd.body_lin_vel_w[:, 0].clone()
    root_ang_vel = cmd.body_ang_vel_w[:, 0].clone()

    soft_joint_pos_limits = cmd.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
        joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    cmd.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
    cmd.robot.write_root_state_to_sim(
        torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
        env_ids=env_ids,
    )

    flag_name = f"{cmd._state_prefix}_motion_resampled"
    resample_flags = getattr(base_env, flag_name, None)
    if resample_flags is None or resample_flags.shape[0] != cmd.num_envs:
        resample_flags = torch.zeros(cmd.num_envs, dtype=torch.bool, device=device)
    else:
        resample_flags = resample_flags.to(device=device, dtype=torch.bool)
    resample_flags[env_ids] = True
    setattr(base_env, flag_name, resample_flags)

    _reset_episode_state(base_env, env_ids)

    if hasattr(cmd, "kick_contact_tracker"):
        try:
            cmd.kick_contact_tracker.reset(env_ids)
        except Exception:
            pass


def assign_motion_files(env_cfg, motion_files: list[str]) -> None:
    approach_files = [f for f in motion_files if f.endswith("_approach.npz")]
    strike_files = [f for f in motion_files if f.endswith("_strike.npz")]
    if approach_files and strike_files:
        env_cfg.commands.motion.motion_files = approach_files
        if hasattr(env_cfg.commands.motion, "strike_motion_files"):
            env_cfg.commands.motion.strike_motion_files = strike_files
    else:
        env_cfg.commands.motion.motion_files = motion_files
        if hasattr(env_cfg.commands.motion, "strike_motion_files"):
            env_cfg.commands.motion.strike_motion_files = motion_files


@dataclass
class RolloutResult:
    completed: bool
    failed: bool
    fail_reasons: list[str]
    steps: int
    mean_errors: dict[str, float]
    ball_contact_ratio: float
    episode_return: float


@dataclass
class MotionSummary:
    motion_file: str
    motion_name: str
    num_frames: int
    num_rollouts: int
    completion_rate: float
    failure_rate: float
    mean_error_joint_pos: float
    mean_error_anchor_pos: float
    mean_error_body_pos: float
    mean_ball_contact_ratio: float
    quality_score: float
    passed: bool
    fail_reason_counts: dict[str, int] = field(default_factory=dict)
    rollouts: list[dict[str, Any]] = field(default_factory=list)


def _rollout_once(
    env,
    policy,
    base_env,
    n_frames: int,
    *,
    grace_steps: int = 0,
    check_failures: bool = True,
    obs=None,
) -> RolloutResult:
    if obs is None:
        obs, _ = env.get_observations()
    error_sums = {k: 0.0 for k in TRACKING_METRIC_KEYS}
    contact_steps = 0
    steps = 0
    episode_return = 0.0
    failed = False
    fail_reasons: list[str] = []

    for _ in range(n_frames):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _extras = env.step(actions)

        steps += 1
        episode_return += float(rewards[0].item()) if rewards is not None else 0.0

        cmd = base_env.command_manager.get_term("motion")
        for key in TRACKING_METRIC_KEYS:
            if key in cmd.metrics:
                error_sums[key] += float(cmd.metrics[key][0].item())

        try:
            force_mag = soccer_ball_contact_force_magnitude(base_env, _BALL_SENSOR_NAME)
            if float(force_mag[0].item()) > _CONTACT_FORCE_THRESHOLD:
                contact_steps += 1
        except Exception:
            pass

        if check_failures and steps > grace_steps:
            active_terms = _failure_terms_active(base_env, 0)
            if active_terms:
                failed = True
                fail_reasons = active_terms
                break

            if dones is not None and bool(dones[0].item()):
                active_terms = _failure_terms_active(base_env, 0)
                if active_terms:
                    failed = True
                    fail_reasons = active_terms
                break

    mean_errors = {k: error_sums[k] / max(steps, 1) for k in TRACKING_METRIC_KEYS}
    completed = (not failed) and steps >= n_frames
    return RolloutResult(
        completed=completed,
        failed=failed,
        fail_reasons=fail_reasons,
        steps=steps,
        mean_errors=mean_errors,
        ball_contact_ratio=contact_steps / max(steps, 1),
        episode_return=episode_return,
    )


def compute_quality_score(summary: dict[str, float]) -> float:
    completion = summary["completion_rate"]
    tracking = math.exp(-summary["mean_error_joint_pos"] / 0.5)
    fail_penalty = 1.0 - summary["failure_rate"]
    contact = summary.get("mean_ball_contact_ratio", 0.0)
    contact_factor = 0.5 + 0.5 * min(1.0, contact) if contact > 0 else 1.0
    return completion * tracking * fail_penalty * contact_factor


def passes_thresholds(summary: MotionSummary, args: argparse.Namespace) -> bool:
    if summary.completion_rate < args.min_completion_rate:
        return False
    if summary.mean_error_joint_pos > args.max_joint_pos_error:
        return False
    if summary.mean_error_anchor_pos > args.max_anchor_pos_error:
        return False
    if args.min_ball_contact_ratio > 0.0 and summary.mean_ball_contact_ratio < args.min_ball_contact_ratio:
        return False
    if summary.quality_score < args.min_quality_score:
        return False
    return True


def summarize_motion(motion_file: str, rollouts: list[RolloutResult], args: argparse.Namespace) -> MotionSummary:
    n = len(rollouts)
    completion_rate = sum(1 for r in rollouts if r.completed) / n
    failure_rate = sum(1 for r in rollouts if r.failed) / n
    mean_error_joint_pos = sum(r.mean_errors.get("error_joint_pos", 0.0) for r in rollouts) / n
    mean_error_anchor_pos = sum(r.mean_errors.get("error_anchor_pos", 0.0) for r in rollouts) / n
    mean_error_body_pos = sum(r.mean_errors.get("error_body_pos", 0.0) for r in rollouts) / n
    mean_ball_contact_ratio = sum(r.ball_contact_ratio for r in rollouts) / n

    fail_reason_counts: dict[str, int] = {}
    for r in rollouts:
        for reason in r.fail_reasons:
            fail_reason_counts[reason] = fail_reason_counts.get(reason, 0) + 1

    score_inputs = {
        "completion_rate": completion_rate,
        "failure_rate": failure_rate,
        "mean_error_joint_pos": mean_error_joint_pos,
        "mean_ball_contact_ratio": mean_ball_contact_ratio,
    }
    quality_score = compute_quality_score(score_inputs)

    summary = MotionSummary(
        motion_file=os.path.abspath(motion_file),
        motion_name=os.path.splitext(os.path.basename(motion_file))[0],
        num_frames=motion_frame_count(motion_file),
        num_rollouts=n,
        completion_rate=completion_rate,
        failure_rate=failure_rate,
        mean_error_joint_pos=mean_error_joint_pos,
        mean_error_anchor_pos=mean_error_anchor_pos,
        mean_error_body_pos=mean_error_body_pos,
        mean_ball_contact_ratio=mean_ball_contact_ratio,
        quality_score=quality_score,
        passed=False,
        fail_reason_counts=fail_reason_counts,
        rollouts=[asdict(r) for r in rollouts],
    )
    summary.passed = passes_thresholds(summary, args)
    return summary


def write_report(report: dict[str, Any], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = os.path.join(output_dir, "summary.csv")
    fieldnames = [
        "motion_name",
        "passed",
        "quality_score",
        "completion_rate",
        "failure_rate",
        "mean_error_joint_pos",
        "mean_error_anchor_pos",
        "mean_error_body_pos",
        "mean_ball_contact_ratio",
        "num_frames",
        "fail_reason_counts",
        "motion_file",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in report["motions"]:
            row = {k: entry.get(k) for k in fieldnames}
            row["fail_reason_counts"] = json.dumps(entry.get("fail_reason_counts", {}), ensure_ascii=False)
            writer.writerow(row)

    manifest_path = os.path.join(output_dir, "passed_motions.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in report["motions"]:
            if entry["passed"]:
                f.write(entry["motion_file"] + "\n")

    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {manifest_path}")


def write_filtered_dir(report: dict[str, Any], filtered_dir: str) -> None:
    os.makedirs(filtered_dir, exist_ok=True)
    count = 0
    for entry in report["motions"]:
        if not entry["passed"]:
            continue
        src = entry["motion_file"]
        dst = os.path.join(filtered_dir, os.path.basename(src))
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        count += 1
    print(f"[INFO] Linked {count} passing motions into {filtered_dir}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.sim.device = args_cli.device if getattr(args_cli, "device", None) else env_cfg.sim.device

    motion_files = get_motion_files(args_cli.motion_path, args_cli.motion_glob)
    _log(f"Evaluating {len(motion_files)} reference(s) from {args_cli.motion_path}")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    _log(f"Checkpoint: {resume_path}")

    if args_cli.recreate_env_per_rollout:
        _log(
            "WARNING: --recreate_env_per_rollout is enabled; each rollout rebuilds Isaac "
            "(minutes of silence at 'Setting seed: N' is expected)."
        )

    configure_eval_env(env_cfg, motion_files)
    if not args_cli.disable_training_terminations:
        _log("Failure terminations ON (same family as play_multi: ball_lost, ee_body_pos, …).")
    else:
        _log(
            "WARNING: --disable_training_terminations — failure_rate / fail_reason_counts "
            "are not meaningful; only full-clip tracking metrics are measured."
        )
        disable_failure_terminations(env_cfg)
    assign_motion_files(env_cfg, motion_files)
    env_cfg.seed = args_cli.seed

    output_dir = args_cli.output_dir or os.path.join(os.path.dirname(resume_path), "eval_references")
    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "checkpoint": resume_path,
        "motion_path": os.path.abspath(args_cli.motion_path),
        "num_rollouts": args_cli.num_rollouts,
        "eval_grace_steps": args_cli.eval_grace_steps,
        "termination_checks_enabled": not args_cli.disable_training_terminations,
        "thresholds": {
            "min_completion_rate": args_cli.min_completion_rate,
            "max_joint_pos_error": args_cli.max_joint_pos_error,
            "max_anchor_pos_error": args_cli.max_anchor_pos_error,
            "min_ball_contact_ratio": args_cli.min_ball_contact_ratio,
            "min_quality_score": args_cli.min_quality_score,
        },
        "motions": [],
    }

    policy = None
    runner = None
    agent_dict = agent_cfg.to_dict()
    env = None
    base_env = None

    def _make_eval_env():
        _log("Creating Isaac env (one-time; first load may take 1–3 min)…")
        wrapped = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(wrapped.unwrapped, DirectMARLEnv):
            wrapped = multi_agent_to_single_agent(wrapped)
        return RslRlVecEnvWrapper(wrapped)

    if not args_cli.recreate_env_per_rollout:
        env = _make_eval_env()
        base_env = _resolve_base_env(env)
        runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
        load_checkpoint_with_obs_expand(runner, resume_path)
        policy = runner.get_inference_policy(device=base_env.device)
        _log("Policy loaded; starting per-reference rollouts.")

    check_failures = not args_cli.disable_training_terminations
    grace_steps = args_cli.eval_grace_steps if check_failures else 0

    for idx, motion_file in enumerate(motion_files):
        motion_name = os.path.basename(motion_file)
        n_frames = motion_frame_count(motion_file)
        _log(f"({idx + 1}/{len(motion_files)}) {motion_name} — {n_frames} frames")

        rollouts: list[RolloutResult] = []
        for rollout_idx in range(args_cli.num_rollouts):
            if args_cli.recreate_env_per_rollout:
                if env is not None:
                    env.close()
                env_cfg.seed = args_cli.seed + rollout_idx
                env = _make_eval_env()
                base_env = _resolve_base_env(env)
                runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
                load_checkpoint_with_obs_expand(runner, resume_path)
                policy = runner.get_inference_policy(device=base_env.device)

            # Fresh RNN hidden state per rollout.
            if runner is not None and hasattr(runner.alg, "policy"):
                pol = runner.alg.policy
                if hasattr(pol, "reset"):
                    pol.reset()
                elif hasattr(pol, "memory_a") and hasattr(pol.memory_a, "reset"):
                    pol.memory_a.reset(torch.ones(1, dtype=torch.bool, device=base_env.device))

            _log(f"  rollout {rollout_idx + 1}/{args_cli.num_rollouts}: preparing…")
            with torch.inference_mode():
                _start_motion_rollout(base_env, idx)
            obs, _ = env.get_observations()
            result = _rollout_once(
                env,
                policy,
                base_env,
                n_frames,
                grace_steps=grace_steps,
                check_failures=check_failures,
                obs=obs,
            )
            rollouts.append(result)
            status = "OK" if result.completed else f"FAIL({','.join(result.fail_reasons) or 'incomplete'})"
            _log(
                f"  rollout {rollout_idx + 1}/{args_cli.num_rollouts}: {status} "
                f"joint_err={result.mean_errors.get('error_joint_pos', 0.0):.3f} "
                f"steps={result.steps}/{n_frames}"
            )

        summary = summarize_motion(motion_file, rollouts, args_cli)
        report["motions"].append(asdict(summary))
        _log(
            f"  => passed={summary.passed} quality={summary.quality_score:.3f} "
            f"completion={summary.completion_rate:.2f} joint_err={summary.mean_error_joint_pos:.3f}"
        )

    if env is not None:
        env.close()

    passed = sum(1 for m in report["motions"] if m["passed"])
    report["summary"] = {
        "total": len(report["motions"]),
        "passed": passed,
        "failed": len(report["motions"]) - passed,
        "pass_rate": passed / max(len(report["motions"]), 1),
    }

    write_report(report, output_dir)
    if args_cli.write_filtered_dir:
        write_filtered_dir(report, args_cli.write_filtered_dir)

    print(
        f"\n[INFO] Done: {passed}/{len(report['motions'])} references passed. "
        f"Report: {output_dir}"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
