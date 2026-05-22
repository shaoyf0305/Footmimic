from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.rewards_dribbling import (
    _dribbling_recent_contact_gate,
    soccer_ball_contact_force_magnitude,
)
from soccer.tasks.tracking.mdp.rewards_dribbling import gather_dribble_phase_bundle
from soccer.tasks.tracking.mdp.task_frame import task_forward_offset

from soccer.tasks.tracking.mdp.rewards import _get_body_indexes


def _dribble_phase_context(
    env: ManagerBasedRLEnv,
    command_name: str,
    ball_sensor_name: str,
    contact_force_threshold: float,
    recent_contact_window: int,
    chase_min_ahead: float,
    approach_max_dist: float,
    approach_ball_speed_max: float,
    chase_ball_speed_min: float = 0.25,
    chase_speed_margin: float = 0.08,
    chase_catchup_ratio: float = 0.90,
    chase_to_approach_dist: float = 0.60,
    chase_max_steps: int = 32,
    approach_enter_speed_margin: float = 0.06,
    foot_cfg: SceneEntityCfg | None = None,
    close_max_dist: float = 0.48,
    close_x_min: float = 0.18,
    close_x_max: float = 0.62,
    seek_touch_min_steps: int = 2,
    seek_touch_commit_dist: float = 0.24,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pelvis–ball geometry and phase masks shared by dribble terminations."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    ball_pos_w = soccer_ball.data.root_pos_w
    dist_xy = torch.norm(ball_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2], dim=-1)

    bundle = gather_dribble_phase_bundle(
        env,
        command,
        ball_sensor_name,
        contact_force_threshold,
        recent_contact_window,
        foot_cfg,
        chase_min_ahead=chase_min_ahead,
        chase_ball_speed_min=chase_ball_speed_min,
        chase_speed_margin=chase_speed_margin,
        chase_catchup_ratio=chase_catchup_ratio,
        chase_to_approach_dist=chase_to_approach_dist,
        chase_max_steps=chase_max_steps,
        approach_enter_speed_margin=approach_enter_speed_margin,
        approach_max_dist=approach_max_dist,
        approach_ball_speed_max=approach_ball_speed_max,
        close_max_dist=close_max_dist,
        close_x_min=close_x_min,
        close_x_max=close_x_max,
        seek_touch_min_steps=seek_touch_min_steps,
        seek_touch_commit_dist=seek_touch_commit_dist,
    )
    has_contact = soccer_ball_contact_force_magnitude(env, ball_sensor_name) > contact_force_threshold
    return (
        has_contact,
        bundle.touch,
        bundle.seek_touch,
        bundle.chase,
        bundle.approach,
        dist_xy,
    )


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def motion_finished(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    last_step = (command.motion_length - 1).clamp(min=0)
    return command.time_steps >= last_step


def ball_lost_dribbling(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    max_distance: float = 1.25,
    max_distance_chase: float = 1.85,
    max_vel_divergence: float = 3.5,
    grace_steps: int = 50,
    recent_contact_window: int = 8,
    chase_min_ahead: float = 0.35,
    chase_ball_speed_min: float = 0.25,
    chase_speed_margin: float = 0.08,
    chase_catchup_ratio: float = 0.90,
    chase_to_approach_dist: float = 0.60,
    chase_max_steps: int = 32,
    approach_enter_speed_margin: float = 0.06,
    approach_max_dist: float = 0.55,
    approach_ball_speed_max: float = 0.35,
) -> torch.Tensor:
    """Terminate the episode if the ball is lost during dribbling.

    During **chase** (ball ahead, no touch), only XY distance is checked and the
    limit is relaxed (``max_distance_chase``). Velocity mismatch is ignored while
    chasing so a kicked ball rolling ahead does not instantly fail the episode.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    _, _, _, chase_phase, _, dist_xy = _dribble_phase_context(
        env,
        command_name,
        ball_sensor_name,
        contact_force_threshold,
        recent_contact_window,
        chase_min_ahead,
        approach_max_dist,
        approach_ball_speed_max,
        chase_ball_speed_min=chase_ball_speed_min,
        chase_speed_margin=chase_speed_margin,
        chase_catchup_ratio=chase_catchup_ratio,
        chase_to_approach_dist=chase_to_approach_dist,
        chase_max_steps=chase_max_steps,
        approach_enter_speed_margin=approach_enter_speed_margin,
    )

    ball_vel_xy = soccer_ball.data.root_lin_vel_w[:, :2]
    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
    vel_diff = torch.norm(ball_vel_xy - pelvis_vel_xy, dim=-1)

    step_buf = getattr(env, "episode_length_buf", torch.zeros(env.num_envs, device=env.device))
    past_grace = step_buf > grace_steps

    dist_limit = torch.where(chase_phase, torch.full_like(dist_xy, max_distance_chase), torch.full_like(dist_xy, max_distance))
    lost_dist = dist_xy > dist_limit
    lost_vel = past_grace & (~chase_phase) & (vel_diff > max_vel_divergence)
    return past_grace & (lost_dist | lost_vel)


def dribbling_no_ball_contact_timeout(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    grace_steps: int = 50,
    max_steps_without_contact: int = 200,
    recent_contact_window: int = 8,
    chase_min_ahead: float = 0.35,
    chase_ball_speed_min: float = 0.25,
    chase_speed_margin: float = 0.08,
    chase_catchup_ratio: float = 0.90,
    chase_to_approach_dist: float = 0.60,
    chase_max_steps: int = 32,
    approach_enter_speed_margin: float = 0.06,
    approach_max_dist: float = 0.55,
    approach_ball_speed_max: float = 0.35,
) -> torch.Tensor:
    """End the episode if the ball sees no robot contact for too long after warm-up.

    The counter does **not** advance during chase phase (ball ahead, pursuing after
    a kick). It only accumulates in approach / idle phases to block "hover near
    the ball without ever touching" without cutting kick–chase–kick rollouts short.
    """
    has_contact, _, _, chase_phase, _, _ = _dribble_phase_context(
        env,
        command_name,
        ball_sensor_name,
        contact_force_threshold,
        recent_contact_window,
        chase_min_ahead,
        approach_max_dist,
        approach_ball_speed_max,
        chase_ball_speed_min=chase_ball_speed_min,
        chase_speed_margin=chase_speed_margin,
        chase_catchup_ratio=chase_catchup_ratio,
        chase_to_approach_dist=chase_to_approach_dist,
        chase_max_steps=chase_max_steps,
        approach_enter_speed_margin=approach_enter_speed_margin,
    )

    step_buf = getattr(env, "episode_length_buf", torch.zeros(env.num_envs, device=env.device))
    past_grace = step_buf > grace_steps
    reset_m = step_buf == 0

    buf_name = "_dribbling_no_contact_step_count"
    cnt = getattr(env, buf_name, None)
    if cnt is None or cnt.shape[0] != env.num_envs:
        cnt = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    # Do not count steps while chasing after a kick; only approach / idle without touch.
    should_count = past_grace & (~has_contact) & (~chase_phase)

    cnt = torch.where(
        reset_m,
        torch.zeros_like(cnt),
        torch.where(has_contact, torch.zeros_like(cnt), torch.where(should_count, cnt + 1, cnt)),
    )
    setattr(env, buf_name, cnt)

    return cnt >= max_steps_without_contact


def contact_phase_violation(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
) -> torch.Tensor:
    """Terminate if the robot touches the ball outside the kick window.

    Uses the Segmented Contact Graph:
      - Phase 1 (frame 0 → kick_start_frame): NO contact allowed → terminate
      - Phase 2 (kick_start_frame → kick_end_frame): contact allowed (no termination)
      - Phase 3 (kick_end_frame → end): NO contact allowed → terminate

    Motions WITHOUT contact graph annotations (kick_start_frame == -1)
    are NEVER terminated by this function, ensuring full backward
    compatibility with MoCap data.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    kick_start = command.kick_start_frame   # [num_envs], -1 = not annotated
    kick_end = command.kick_end_frame       # [num_envs], -1 = not annotated

    # Only enforce on envs that have BOTH annotations.
    has_graph = (kick_start >= 0) & (kick_end >= 0)
    if not torch.any(has_graph):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    t = command.time_steps  # [num_envs] current frame
    in_phase1 = t < kick_start
    in_phase3 = t > kick_end
    outside_window = (in_phase1 | in_phase3) & has_graph

    if not torch.any(outside_window):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Detect robot-ball contact using horizontal forces only.
    # Ground contact is vertical (Z-axis) and must be excluded.
    ball_sensor = env.scene.sensors[ball_sensor_name]
    forces = ball_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
    if forces is None or forces.numel() == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Horizontal (XY) force indicates robot-ball contact, not ground.
    force_horizontal = torch.norm(forces[..., :2], dim=-1)  # [num_envs, num_bodies]
    has_contact = torch.any(force_horizontal > 5.0, dim=-1)  # [num_envs]

    # Terminate: outside kick window AND robot-ball contact detected.
    return outside_window & has_contact


def interaction_termination(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_speed_threshold: float = 0.3,
    grace_frames: int = 10,
) -> torch.Tensor:
    """Terminate if the robot fails to kick the ball after the kick window.

    After kick_end_frame + grace_frames, if the ball's XY speed is still
    below `ball_speed_threshold`, the episode is terminated.  This prevents
    the robot from 'freeloading' by just tracking motion without kicking.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    kick_end = command.kick_end_frame  # (num_envs,), -1 = not annotated
    t = command.time_steps

    has_annotation = kick_end >= 0
    past_window = has_annotation & (t > (kick_end + grace_frames))

    if not torch.any(past_window):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    soccer_ball = env.scene["soccer_ball"]
    ball_speed_xy = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)

    # Check if kick tracker recorded a successful contact.
    from soccer.tasks.tracking.mdp.rewards import _get_kick_tracker
    tracker = _get_kick_tracker(command)
    contact_awarded = tracker.get_contact_awarded()

    # Terminate if past window AND (no contact recorded OR ball barely moved).
    failed = past_window & (~contact_awarded | (ball_speed_xy < ball_speed_threshold))
    return failed
