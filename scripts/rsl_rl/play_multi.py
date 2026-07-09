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
    "--locomotion_cmd_wz",
    type=float,
    nargs="*",
    default=None,
    help="Optional yaw rate (rad/s) per polar segment; one value broadcasts to all segments.",
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

from soccer.tasks.tracking.mdp.rewards_dribbling import soccer_ball_contact_force_magnitude

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


def _resolve_base_env(env):
    """Unwrap gym / RSL-RL wrappers to the underlying Isaac Lab env."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    if hasattr(base, "unwrapped"):
        base = base.unwrapped
    return base


def _apply_play_locomotion_command(env, args_cli) -> bool:
    """Apply CLI locomotion: multi-segment polar sequence or legacy vx/vy/wz."""
    polar_set = (
        args_cli.locomotion_cmd_speed is not None
        or args_cli.locomotion_cmd_heading is not None
        or args_cli.locomotion_cmd_duration is not None
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

        if speeds is None:
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
            segments = [(speeds[i], headings[i], durations[i], wz_list[i]) for i in range(n)]
            cmd.set_locomotion_polar_sequence(segments)
            print(f"[INFO] Locomotion sequence ({n} segments):")
            for i, (sp, hd, dur, wz) in enumerate(segments):
                print(f"  [{i + 1}] speed={sp:.3f} m/s  heading={hd:.3f} rad  duration={dur:.2f} s  wz={wz:.3f}")
            return True

        cmd.set_locomotion_command_mode("manual")
        cmd.set_locomotion_polar_command(
            speed=speeds[0],
            heading=headings[0],
            duration_s=durations[0],
            wz=wz_list[0],
        )
        print(
            f"[INFO] Locomotion polar cmd: speed={speeds[0]:.3f} m/s  heading={headings[0]:.3f} rad  "
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
    cmd.set_locomotion_manual_command(lin_vel=lin, ang_vel=ang)
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
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if video_recorder is not None:
            overlay = _get_play_overlay(env)
            video_recorder.capture(overlay_text=overlay)

        timestep += 1
        if play_steps is not None and timestep >= play_steps:
            break

    if video_recorder is not None:
        video_recorder.save()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
