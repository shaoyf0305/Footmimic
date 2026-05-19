"""Replay a motion ``.npz`` with the G1 robot and visualize the soccer ball trajectory.

Loads ``ball_pos_w`` from the motion file (e.g. after ``synthesize_dribble_ball_traj.py``)
and drives a kinematic soccer ball mesh each frame. Optionally draws a rolling trail of
marker spheres and prints CG contact / foot–ball distance in the terminal.

``--motion_path`` may be a single ``.npz`` or a directory (all ``*.npz`` inside, sorted).

.. code-block:: bash

    # Single clip
    python scripts/replay_npz_with_ball.py \\
        --motion_path motions/dribble-distance/FAST-seg1_unitree_g1.npz

    # Whole folder (plays each clip in order; loops folder by default)
    python scripts/replay_npz_with_ball.py \\
        --motion_path motions/dribble-distance --device cpu

    # No GPU machine — always use CPU to avoid cudaErrorNoDevice
    python scripts/replay_npz_with_ball.py \\
        --motion_path motions/dribble-distance --device cpu --real_time
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay motion .npz with ball trajectory visualization.")
parser.add_argument(
    "--motion_path",
    type=str,
    required=True,
    help="Path to a motion .npz or a directory of .npz files.",
)
parser.add_argument(
    "--trail_length",
    type=int,
    default=80,
    help="Number of past ball positions drawn as trail markers (0 to disable).",
)
parser.add_argument(
    "--trail_stride",
    type=int,
    default=2,
    help="Subsample trail markers every N frames (reduces clutter).",
)
parser.add_argument(
    "--real_time",
    action="store_true",
    help="Sleep so playback matches motion fps (default: run as fast as sim can render).",
)
parser.add_argument(
    "--loop",
    action="store_true",
    default=True,
    help="Loop the current clip when finished (default: True).",
)
parser.add_argument(
    "--no_loop",
    action="store_false",
    dest="loop",
    help="Stop after one play-through of the current clip.",
)
parser.add_argument(
    "--playlist_loop",
    action="store_true",
    default=True,
    help="When motion_path is a directory, loop the full playlist (default: True).",
)
parser.add_argument(
    "--no_playlist_loop",
    action="store_false",
    dest="playlist_loop",
    help="Stop after playing every file in the directory once.",
)
parser.add_argument(
    "--gap_seconds",
    type=float,
    default=0.8,
    help="Pause between clips when playing a directory (seconds).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _resolve_sim_device(requested: str | None) -> str:
    """Pick a valid Isaac Sim device; fall back to CPU if CUDA is unavailable."""
    if requested is None or str(requested).strip() == "":
        requested = "cuda:0"
    dev = str(requested).strip().lower()
    if dev == "cuda":
        dev = "cuda:0"
    wants_cuda = dev.startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        print(
            "[WARN] No CUDA-capable GPU detected (cudaErrorNoDevice). "
            "Using --device cpu. Set explicitly: --device cpu"
        )
        return "cpu"
    return requested if not wants_cuda or torch.cuda.is_available() else "cpu"


# Ensure AppLauncher / SimulationContext never default to missing CUDA.
args_cli.device = _resolve_sim_device(getattr(args_cli, "device", None))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from soccer.robots.g1 import G1_CYLINDER_CFG
from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import SOCCER_ASSET_PATH, SOCCER_BALL_RADIUS
from soccer.tasks.tracking.mdp.commands import MotionLoader


def collect_motion_files(motion_path: str) -> list[str]:
    """Return sorted list of .npz paths (single file or directory)."""
    if os.path.isfile(motion_path):
        return [motion_path]
    if os.path.isdir(motion_path):
        files = sorted(glob.glob(os.path.join(motion_path, "*.npz")))
        if not files:
            raise ValueError(f"No .npz files found in directory: {motion_path}")
        print(f"[playlist] {len(files)} motion file(s) in {motion_path}")
        for f in files:
            print(f"  - {os.path.basename(f)}")
        return files
    raise ValueError(f"Invalid motion_path: {motion_path} (must be a file or directory)")


class SoccerMotionReplayData:
    """Motion loader extended with optional ball / CG arrays from ``.npz``."""

    def __init__(self, motion_file: str, device: str):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.motion_file = motion_file
        self.motion_name = os.path.basename(motion_file)
        data = np.load(motion_file)

        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 50.0
        self.time_step_total = int(data["joint_pos"].shape[0])

        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

        self.has_ball = "ball_pos_w" in data.files
        if self.has_ball:
            ball = np.asarray(data["ball_pos_w"], dtype=np.float32).reshape(-1, 3)[: self.time_step_total]
            self.ball_pos_w = torch.tensor(ball, dtype=torch.float32, device=device)
            if self.time_step_total > 1:
                vel = np.zeros_like(ball)
                vel[1:] = (ball[1:] - ball[:-1]) * self.fps
                vel[0] = vel[1]
                self.ball_lin_vel_w = torch.tensor(vel, dtype=torch.float32, device=device)
            else:
                self.ball_lin_vel_w = torch.zeros(1, 3, device=device, dtype=torch.float32)
        else:
            self.ball_pos_w = None
            self.ball_lin_vel_w = None

        self.cg_contact = None
        if "dribble_cg_contact" in data.files:
            cc = np.asarray(data["dribble_cg_contact"], dtype=np.int8).reshape(-1)[: self.time_step_total]
            self.cg_contact = torch.tensor(cc, device=device, dtype=torch.bool)

        self.cg_foot_ball_dist = None
        if "dribble_cg_foot_ball_dist" in data.files:
            cd = np.asarray(data["dribble_cg_foot_ball_dist"], dtype=np.float32).reshape(-1)[: self.time_step_total]
            self.cg_foot_ball_dist = torch.tensor(cd, device=device, dtype=torch.float32)

        self._body_index = torch.tensor([0], dtype=torch.long, device=device)


@configclass
class ReplayWithBallSceneCfg(InteractiveSceneCfg):
    """Robot + soccer ball + optional debug markers."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCCER_ASSET_PATH,
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, SOCCER_BALL_RADIUS)),
    )


def _make_trail_markers() -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/BallTrail",
        markers={
            "trail": sim_utils.SphereCfg(
                radius=SOCCER_BALL_RADIUS * 0.55,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _make_contact_markers() -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/BallContact",
        markers={
            "contact": sim_utils.SphereCfg(
                radius=SOCCER_BALL_RADIUS * 0.75,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.95, 0.25)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _update_trail(
    trail: collections.deque[np.ndarray],
    trail_markers: VisualizationMarkers | None,
    ball_pos: np.ndarray,
    trail_length: int,
    trail_stride: int,
    device: str,
) -> None:
    if trail_markers is None or trail_length <= 0:
        return
    trail.append(ball_pos.copy())
    while len(trail) > trail_length:
        trail.popleft()
    positions = np.stack(list(trail)[:: max(1, trail_stride)], axis=0)
    if positions.shape[0] == 0:
        return
    orientations = np.zeros((positions.shape[0], 4), dtype=np.float32)
    orientations[:, 0] = 1.0
    trail_markers.visualize(
        translations=torch.tensor(positions, dtype=torch.float32, device=device),
        orientations=torch.tensor(orientations, dtype=torch.float32, device=device),
    )


def _load_motion(motion_path: str, device: str) -> tuple[SoccerMotionReplayData, MotionLoader]:
    motion = SoccerMotionReplayData(motion_path, device=device)
    motion_legacy = MotionLoader(motion_path, torch.tensor([0], device=device), device)
    if not motion.has_ball:
        print(
            f"[WARN] {motion.motion_name}: no ball_pos_w — robot only. "
            "Run: python scripts/dribble/synthesize_dribble_ball_traj.py --motion_path <dir>"
        )
    return motion, motion_legacy


def run_simulator(sim: SimulationContext, scene: InteractiveScene, motion_files: list[str]) -> None:
    robot: Articulation = scene["robot"]
    soccer_ball: RigidObject = scene["soccer_ball"]
    sim_dt = sim.get_physics_dt()
    multi_file = len(motion_files) > 1

    file_idx = 0
    motion, motion_legacy = _load_motion(motion_files[file_idx], sim.device)
    frame_dt = 1.0 / motion.fps if motion.fps > 0 else sim_dt

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    trail_markers = _make_trail_markers() if args_cli.trail_length > 0 else None
    contact_markers = _make_contact_markers()
    trail_deque: collections.deque[np.ndarray] = collections.deque()

    def _switch_clip(next_idx: int) -> None:
        nonlocal file_idx, motion, motion_legacy, frame_dt, time_steps, trail_deque
        file_idx = next_idx
        print(f"\n[playlist] ({file_idx + 1}/{len(motion_files)}) {motion_files[file_idx]}")
        motion, motion_legacy = _load_motion(motion_files[file_idx], sim.device)
        frame_dt = 1.0 / motion.fps if motion.fps > 0 else sim_dt
        time_steps[:] = 0
        trail_deque.clear()

    print(f"[sim] device={sim.device}")
    print(f"[playlist] clip 1/{len(motion_files)}: {motion_files[0]}")

    while simulation_app.is_running():
        frame_start = time.perf_counter()
        t = int(time_steps[0].item())

        if t >= motion.time_step_total:
            if not multi_file:
                if args_cli.loop:
                    time_steps[:] = 0
                    trail_deque.clear()
                else:
                    print("\n[done] finished clip.")
                    break
            else:
                next_idx = file_idx + 1
                if next_idx >= len(motion_files):
                    if args_cli.playlist_loop:
                        next_idx = 0
                    else:
                        print("\n[playlist] finished all clips.")
                        break
                if args_cli.gap_seconds > 0:
                    time.sleep(args_cli.gap_seconds)
                _switch_clip(next_idx)
            continue

        time_steps += 1
        ts = torch.tensor([t], dtype=torch.long, device=sim.device)

        root_states = robot.data.default_root_state.clone()
        origin = scene.env_origins
        root_states[:, :3] = motion_legacy.body_pos_w[ts] + origin
        root_states[:, 3:7] = motion_legacy.body_quat_w[ts]
        root_states[:, 7:10] = motion_legacy.body_lin_vel_w[ts]
        root_states[:, 10:] = motion_legacy.body_ang_vel_w[ts]
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion_legacy.joint_pos[ts], motion_legacy.joint_vel[ts])

        if motion.has_ball:
            ball_pos_w = motion.ball_pos_w[t] + origin[0]
            ball_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)
            ball_lin_vel = motion.ball_lin_vel_w[t].unsqueeze(0)
            ball_ang_vel = torch.zeros(1, 3, device=sim.device)
            ball_state = torch.cat([ball_pos_w.unsqueeze(0), ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
            soccer_ball.write_root_state_to_sim(ball_state)

            ball_np = ball_pos_w.detach().cpu().numpy()
            _update_trail(
                trail_deque,
                trail_markers,
                ball_np,
                args_cli.trail_length,
                args_cli.trail_stride,
                sim.device,
            )

            if motion.cg_contact is not None:
                if bool(motion.cg_contact[t].item()):
                    contact_markers.visualize(
                        translations=ball_pos_w.unsqueeze(0),
                        orientations=ball_quat,
                    )
                else:
                    contact_markers.visualize(
                        translations=torch.tensor([[0.0, 0.0, -10.0]], device=sim.device),
                        orientations=ball_quat,
                    )

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].detach().cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.2, 2.2, 0.55]), pos_lookat)

        hud = f"\r[{file_idx + 1}/{len(motion_files)}] {motion.motion_name}  "
        hud += f"Frame {t:4d}/{motion.time_step_total - 1}  t={t / motion.fps:6.2f}s"
        if motion.has_ball:
            bp = motion.ball_pos_w[t].detach().cpu().numpy()
            spd = float(torch.norm(motion.ball_lin_vel_w[t]).item())
            hud += f"  ball=({bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:.2f})  |v|={spd:.2f}"
        if motion.cg_contact is not None:
            hud += f"  CG={int(motion.cg_contact[t].item())}"
        if motion.cg_foot_ball_dist is not None:
            d = float(motion.cg_foot_ball_dist[t].item())
            if d >= 0.0:
                hud += f"  d_foot_ball={d:.3f}m"
        print(hud, end="", flush=True)

        if args_cli.real_time:
            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)


def main() -> None:
    motion_files = collect_motion_files(args_cli.motion_path)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayWithBallSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene, motion_files)


if __name__ == "__main__":
    main()
    simulation_app.close()
