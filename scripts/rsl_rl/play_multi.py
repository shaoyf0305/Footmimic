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
    soccer_ball_contact_force_magnitude,
    soccer_ball_contact_net_force_w,
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
    """Disable all MDP termination terms so playback runs full clips for video eval."""
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


def _setup_play_episode_limit(env_cfg, motion_files: list[str]) -> int:
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


_HUD_ANKLE_BODIES = ["right_ankle_roll_link", "left_ankle_roll_link"]
_BALL_SENSOR_NAME = "soccer_ball_contact"
_CONTACT_FORCE_THRESHOLD = 1.0


def _resolve_base_env(env):
    """Unwrap gym / RSL-RL wrappers to the underlying Isaac Lab env."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    if hasattr(base, "unwrapped"):
        base = base.unwrapped
    return base


def _get_play_overlay(env, timestep: int) -> str:
    """HUD for dual-view video: speeds, distances, contact, and CG labels."""
    lines: list[str] = [f"Step: {timestep}"]
    try:
        base_env = _resolve_base_env(env)
        cmd = base_env.command_manager.get_term("motion")
        i = 0

        if hasattr(cmd, "motion_idx") and hasattr(cmd, "motion") and hasattr(cmd.motion, "motion_name"):
            mi = int(cmd.motion_idx[i].item())
            names = cmd.motion.motion_name
            n = len(names)
            ref = names[mi] if 0 <= mi < n else f"motion_{mi}"
            lines.insert(0, f"Ref: {ref}  ({mi + 1}/{n})")

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
        lines.append(
            f"Pelvis-Ball: {pelvis_ball_xy:.2f} m (xy)  |  {pelvis_ball_3d:.2f} m (3D)"
        )

        robot = base_env.scene["robot"]
        ankle_dists: list[float] = []
        for fname in _HUD_ANKLE_BODIES:
            if fname not in robot.body_names:
                continue
            bidx = robot.body_names.index(fname)
            fpos = robot.data.body_pos_w[i, bidx].detach().cpu().numpy()
            ankle_dists.append(float(np.linalg.norm(fpos - ball_pos)))
        if ankle_dists:
            lines.append(f"Ankle-Ball (min): {min(ankle_dists):.2f} m (3D)")

        force_xy = float(
            soccer_ball_contact_force_magnitude(base_env, _BALL_SENSOR_NAME)[i].item()
        )
        force_vec = soccer_ball_contact_net_force_w(base_env, _BALL_SENSOR_NAME)[i].detach().cpu().numpy()
        force_z = float(abs(force_vec[2]))
        sim_touch = force_xy > _CONTACT_FORCE_THRESHOLD

        lines.append(
            f"Robot-Ball: {'YES' if sim_touch else 'NO'}  |  "
            f"F_xy={force_xy:.1f} N  |  F_z~{force_z:.1f} N (ground)"
        )

        if hasattr(cmd, "motion_has_dribble_cg_label") and bool(cmd.motion_has_dribble_cg_label[i].item()):
            ref_contact = bool(cmd.dribble_cg_contact_ref[i].item())
            ref_foot = int(cmd.dribble_cg_foot_ref[i].item())
            foot_lbl = {0: "L", 1: "R"}.get(ref_foot, "-")
            match = "ok" if ref_contact == sim_touch else "MISMATCH"
            lines.append(
                f"CG label: contact={int(ref_contact)} foot={foot_lbl}  ({match})"
            )

        if hasattr(cmd, "dribble_cg_foot_ball_dist_ref"):
            demo_dist = float(cmd.dribble_cg_foot_ball_dist_ref[i].item())
            if demo_dist >= 0.0:
                lines.append(f"Foot-Ball demo: {demo_dist:.2f} m")
            else:
                lines.append("Foot-Ball demo: -")
        elif hasattr(cmd, "kick_frame"):
            kf = int(cmd.kick_frame[i].item())
            margin = 5
            if kf < 0:
                lines.append("Kick CG: no annotation")
            elif t < kf - margin:
                lines.append(f"Kick CG: 0 (approach)  |  kick_frame={kf}")
            else:
                lines.append(f"Kick CG: 1 (kick)  |  kick_frame={kf}")

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

    total_play_steps = _setup_play_episode_limit(env_cfg, motion_files)
    if len(motion_files) > 1:
        print(
            f"[INFO] Sequential playback: {len(motion_files)} references, "
            f"{total_play_steps} steps total (episode cap {env_cfg.episode_length_s:.1f}s, terminations off)"
        )
    elif motion_files:
        print(
            f"[INFO] Playback episode cap {env_cfg.episode_length_s:.1f}s "
            f"({total_play_steps} frames, terminations off)"
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

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

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
        from dual_view_recorder import DualViewRecorder

        tag = "dual" if args_cli.dual_view else "play"
        video_dir = os.path.join(
            log_dir, "videos", f"{tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        video_recorder = DualViewRecorder(
            env=env.unwrapped if hasattr(env, "unwrapped") else env,
            output_dir=video_dir,
            resolution=(960, 540),
            front_offset=(4.0, 3.0, 2.5),
            back_offset=(-4.0, -3.0, 2.5),
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
        print("[INFO] HUD: Ref name, speeds, distances, contact, CG labels.")
    elif play_steps is not None:
        print(f"[INFO] Running {play_steps} steps (sequential references)")

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if video_recorder is not None:
            overlay = _get_play_overlay(env, timestep)
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
