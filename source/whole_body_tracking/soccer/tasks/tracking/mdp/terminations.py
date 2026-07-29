from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand, locomotion_task_state_mask
from soccer.tasks.tracking.mdp.rewards_dribbling import (
    dribbling_stop_settle_state,
    dribbling_stable_coast_state,
    soccer_ball_contact_force_magnitude,
)

from soccer.tasks.tracking.mdp.rewards import _get_body_indexes


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
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
    grace_steps_after_resample: int = 0,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    failed = torch.any(error > threshold, dim=-1)
    if grace_steps_after_resample > 0:
        steps = getattr(command, "_steps_since_resample", None)
        if steps is not None:
            failed = failed & (steps > grace_steps_after_resample)
    return failed


def motion_finished(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    last_step = (command.motion_length - 1).clamp(min=0)
    return command.time_steps >= last_step


def ball_lost_dribbling(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    max_distance: float = 1.0,
    max_vel_divergence: float = 2.0,
    grace_steps: int = 50,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Terminate the episode if the ball is lost during dribbling.

    The ball is considered "lost" when EITHER:
    - The XY distance between ball and pelvis exceeds ``max_distance`` (m), OR
    - The XY velocity difference between ball and pelvis exceeds
      ``max_vel_divergence`` (m/s).

    A ``grace_steps`` warm-up period is provided at the start of each episode
    so the robot has time to approach the ball before termination kicks in.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    active_task_state = locomotion_task_state_mask(command, active_task_states)
    soccer_ball = env.scene["soccer_ball"]

    # XY distance between ball and pelvis
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]
    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    dist_xy = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)

    # XY velocity divergence
    ball_vel_xy = soccer_ball.data.root_lin_vel_w[:, :2]
    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
    vel_diff = torch.norm(ball_vel_xy - pelvis_vel_xy, dim=-1)

    # Grace period: don't terminate during the first N steps
    step_buf = getattr(env, "episode_length_buf", torch.zeros(env.num_envs, device=env.device))
    past_grace = step_buf > grace_steps

    lost = active_task_state & past_grace & ((dist_xy > max_distance) | (vel_diff > max_vel_divergence))
    setattr(env, "_dribbling_ball_lost_task_active", active_task_state)
    return lost


def dribbling_no_ball_contact_timeout(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    grace_steps: int = 50,
    max_steps_without_contact: int = 125,
    recovery_window_steps: int = 0,
    recovery_max_distance: float = 0.0,
    recovery_min_closing_speed: float = 0.0,
    recovery_counter_increment: float = 1.0,
    proximity_recovery_max_steps: int = 0,
    proximity_recovery_max_distance: float = 0.0,
    proximity_recovery_max_relative_speed: float = 0.0,
    proximity_recovery_counter_increment: float = 1.0,
    allow_stable_coast: bool = False,
    stable_coast_counter_decrement: float = 1.0,
    coast_min_command_speed: float = 0.35,
    coast_min_ball_speed_ratio: float = 0.70,
    coast_min_pelvis_speed_ratio: float = 0.70,
    coast_max_forward_speed_error: float = 0.25,
    coast_min_forward_offset: float = 0.22,
    coast_max_forward_offset: float = 0.75,
    coast_max_lateral_offset: float = 0.22,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """End the episode if the ball sees no robot contact for too long after warm-up.

    Counts simulation steps (post-``grace_steps``) where ball contact force stays
    at or below ``contact_force_threshold``. Resets the counter on contact or on
    episode start. Complements ``ball_lost_dribbling`` by discouraging "pose near
    the ball but never touch" strategies.

    A task-specific recovery window can slow (not clear) the counter after a
    locomotion command change.  A second, bounded recovery allowance applies
    while the ball is close and coasting at a recoverable relative speed.  Both
    paths have finite budgets, so neither permits unlimited no-contact survival.

    Control tasks may additionally classify a well-contained, matched-speed
    ball as stable coast.  That is not loss of control: its counter decays
    while the state remains safe and resumes immediately when speed or geometry
    leaves the configured coast envelope.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    active_task_state = locomotion_task_state_mask(command, active_task_states)
    force_mag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    has_contact = force_mag > contact_force_threshold

    step_buf = getattr(env, "episode_length_buf", torch.zeros(env.num_envs, device=env.device))
    past_grace = step_buf > grace_steps
    reset_m = step_buf == 0

    buf_name = "_dribbling_no_contact_step_count"
    cnt = getattr(env, buf_name, None)
    if cnt is None or cnt.shape[0] != env.num_envs:
        cnt = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    recovery_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    proximity_recovery_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    stable_coast_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    closing_speed = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    relative_speed = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    ball_distance = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if (
        (recovery_window_steps > 0 and recovery_max_distance > 0.0)
        or (proximity_recovery_max_steps > 0 and proximity_recovery_max_distance > 0.0)
    ):
        ball = env.scene["soccer_ball"]
        delta_xy = ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
        ball_distance = torch.norm(delta_xy, dim=-1)
        direction_xy = delta_xy / ball_distance.unsqueeze(-1).clamp(min=1e-6)
        relative_vel_xy = ball.data.root_lin_vel_w[:, :2] - command.robot_anchor_lin_vel_w[:, :2]
        relative_speed = torch.norm(relative_vel_xy, dim=-1)
        # Positive means the pelvis is reducing the ball distance.
        closing_speed = -torch.sum(relative_vel_xy * direction_xy, dim=-1)
        if recovery_window_steps > 0 and recovery_max_distance > 0.0:
            command_age = getattr(command, "_locomotion_cmd_steps_since_change", None)
            if command_age is not None:
                recovery_active = (
                    (command_age <= int(recovery_window_steps))
                    & (ball_distance <= float(recovery_max_distance))
                    & (closing_speed >= float(recovery_min_closing_speed))
                )
        if proximity_recovery_max_steps > 0 and proximity_recovery_max_distance > 0.0:
            proximity_buf_name = "_dribbling_no_contact_proximity_recovery_steps"
            proximity_steps = getattr(env, proximity_buf_name, None)
            if proximity_steps is None or proximity_steps.shape[0] != env.num_envs:
                proximity_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
            proximity_candidate = (
                (ball_distance <= float(proximity_recovery_max_distance))
                & (relative_speed <= float(proximity_recovery_max_relative_speed))
                & (proximity_steps < int(proximity_recovery_max_steps))
            )
            proximity_recovery_active = proximity_candidate & ~has_contact & past_grace & active_task_state
            proximity_steps = torch.where(
                reset_m | has_contact | ~active_task_state,
                torch.zeros_like(proximity_steps),
                proximity_steps + proximity_recovery_active.to(proximity_steps.dtype),
            )
            setattr(env, proximity_buf_name, proximity_steps)

    if allow_stable_coast:
        stable_coast_active = dribbling_stable_coast_state(
            env,
            command_name=command_name,
            min_command_speed=coast_min_command_speed,
            min_ball_speed_ratio=coast_min_ball_speed_ratio,
            min_pelvis_speed_ratio=coast_min_pelvis_speed_ratio,
            max_forward_speed_error=coast_max_forward_speed_error,
            min_forward_offset=coast_min_forward_offset,
            max_forward_offset=coast_max_forward_offset,
            max_lateral_offset=coast_max_lateral_offset,
        )

    command_increment = torch.where(
        recovery_active,
        torch.full_like(cnt, float(recovery_counter_increment)),
        torch.ones_like(cnt),
    )
    increment = torch.where(
        proximity_recovery_active,
        torch.full_like(cnt, float(proximity_recovery_counter_increment)),
        command_increment,
    )

    no_contact_cnt = torch.where(
        stable_coast_active,
        torch.clamp(cnt - float(stable_coast_counter_decrement), min=0.0),
        cnt + increment,
    )
    cnt = torch.where(
        reset_m | ~active_task_state,
        torch.zeros_like(cnt),
        torch.where(
            past_grace,
            torch.where(has_contact, torch.zeros_like(cnt), no_contact_cnt),
            torch.zeros_like(cnt),
        ),
    )
    setattr(env, buf_name, cnt)
    # Compact diagnostics for the play HUD and evaluation collectors.
    setattr(env, "_dribbling_no_contact_force", force_mag)
    setattr(env, "_dribbling_no_contact_task_active", active_task_state)
    setattr(env, "_dribbling_no_contact_count", cnt)
    setattr(env, "_dribbling_no_contact_recovery_active", recovery_active)
    setattr(env, "_dribbling_no_contact_proximity_recovery_active", proximity_recovery_active)
    setattr(env, "_dribbling_no_contact_stable_coast_active", stable_coast_active)
    setattr(env, "_dribbling_no_contact_ball_distance", ball_distance)
    setattr(env, "_dribbling_no_contact_closing_speed", closing_speed)
    setattr(env, "_dribbling_no_contact_relative_speed", relative_speed)

    return active_task_state & (cnt >= max_steps_without_contact)


def dribbling_stop_success(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_settle_duration_s: float = 0.5,
    settled_pelvis_speed: float = 0.05,
    settled_pelvis_angular_speed: float = 0.20,
) -> torch.Tensor:
    """End a successful episode after a sustained, stable robot STOP.

    The ball is deliberately outside this completion condition and may keep
    rolling after the final dribble. Count elapsed *control* time only while
    the robot remains stationary and stable, then request a normal environment
    reset. The command term's reset path restarts the task at its IDLE state.
    """
    (
        active,
        pelvis_speed,
        pelvis_angular_speed,
        ball_speed,
        forward_offset,
        lateral_offset,
        settled,
    ) = dribbling_stop_settle_state(
        env,
        command_name=command_name,
        settled_pelvis_speed=settled_pelvis_speed,
        settled_pelvis_angular_speed=settled_pelvis_angular_speed,
    )
    elapsed_name = "_dribbling_stop_settle_elapsed_s"
    elapsed = getattr(env, elapsed_name, None)
    if elapsed is None or elapsed.shape[0] != env.num_envs:
        elapsed = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    step_buf = getattr(env, "episode_length_buf", torch.zeros(env.num_envs, device=env.device))
    reset_m = step_buf == 0
    step_dt = float(getattr(env, "step_dt", 0.02))
    elapsed = torch.where(
        reset_m | ~settled,
        torch.zeros_like(elapsed),
        elapsed + step_dt,
    )
    success = active & (elapsed >= float(min_settle_duration_s))
    setattr(env, elapsed_name, elapsed)
    setattr(env, "_dribbling_stop_success", success)
    # The reward normally publishes these too, but the termination manager may
    # execute first, so diagnostics must be complete regardless of manager order.
    setattr(env, "_dribbling_stop_active", active)
    setattr(env, "_dribbling_stop_pelvis_speed", pelvis_speed)
    setattr(env, "_dribbling_stop_pelvis_angular_speed", pelvis_angular_speed)
    setattr(env, "_dribbling_stop_ball_speed", ball_speed)
    setattr(env, "_dribbling_stop_forward_offset", forward_offset)
    setattr(env, "_dribbling_stop_lateral_offset", lateral_offset)
    setattr(env, "_dribbling_stop_settled", settled)
    return success


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
