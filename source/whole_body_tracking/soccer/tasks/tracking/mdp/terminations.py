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
    dribbling_s2_contact_event_state,
    dribbling_stop_settle_state,
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


def motion_clip_finished(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Return the command-owned clip-end flag for strict-reference episodes."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    finished = getattr(command, "_motion_clip_finished", None)
    if not isinstance(finished, torch.Tensor) or finished.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return finished


def locomotion_manual_sequence_finished(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """End playback when a manual locomotion sequence reaches its final segment.

    This is inactive for normal resampled training commands.  A manual play
    command opts in through ``reset_on_end`` and the subsequent environment
    reset restores both the robot/ball scene and segment zero of its plan.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    finished = getattr(command, "_locomotion_sequence_finished", None)
    if not isinstance(finished, torch.Tensor) or finished.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return finished


def ball_lost_dribbling(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    max_distance: float = 1.0,
    max_vel_divergence: float = 2.0,
    grace_steps: int = 50,
    max_consecutive_steps: int = 1,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Terminate the episode if the ball is lost during dribbling.

    The ball is considered "lost" when EITHER:
    - The XY distance between ball and pelvis exceeds ``max_distance`` (m), OR
    - The XY velocity difference exceeds a positive ``max_vel_divergence``.

    The condition must persist for ``max_consecutive_steps``.  Setting
    ``max_vel_divergence <= 0`` disables the velocity branch, which is the S2
    behavior; legacy callers retain their old defaults.
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

    velocity_lost = (
        vel_diff > max_vel_divergence
        if max_vel_divergence > 0.0
        else torch.zeros_like(dist_xy, dtype=torch.bool)
    )
    lost_now = active_task_state & past_grace & ((dist_xy > max_distance) | velocity_lost)
    count = getattr(env, "_dribbling_ball_lost_consecutive_steps", None)
    if not isinstance(count, torch.Tensor) or count.shape[0] != env.num_envs:
        count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    reset = step_buf == 0
    count = torch.where(reset | ~lost_now, torch.zeros_like(count), count + 1)
    setattr(env, "_dribbling_ball_lost_consecutive_steps", count)
    lost = lost_now & (count >= max(1, int(max_consecutive_steps)))
    setattr(env, "_dribbling_ball_lost_task_active", active_task_state)
    setattr(env, "_dribbling_ball_lost_distance", dist_xy)
    setattr(env, "_dribbling_ball_lost_velocity_difference", vel_diff)
    return lost


def dribbling_missed_contact(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    max_touch_force: float = 150.0,
    soft_touch_force_start: float = 80.0,
    side_deadzone: float = 0.04,
    proximity_contact_distance_max: float = 0.25,
    proximity_front_gate_min: float = -0.05,
    proximity_front_gate_full: float = 0.05,
    target_region_std: float = 0.12,
    proximity_approach_seconds: float = 0.30,
    proximity_approach_min_weight: float = 0.20,
    missed_contact_grace_steps: int = 3,
    max_curriculum_level: int | None = None,
) -> torch.Tensor:
    """Terminate an early-level S2 episode after a missed selected event."""
    state = dribbling_s2_contact_event_state(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        all_body_cfg=all_body_cfg,
        num_ankle_links=num_ankle_links,
        require_expected_foot=require_expected_foot,
        target_side_enabled=target_side_enabled,
        max_touch_force=max_touch_force,
        soft_touch_force_start=soft_touch_force_start,
        side_deadzone=side_deadzone,
        proximity_contact_distance_max=proximity_contact_distance_max,
        proximity_front_gate_min=proximity_front_gate_min,
        proximity_front_gate_full=proximity_front_gate_full,
        target_region_std=target_region_std,
        proximity_approach_seconds=proximity_approach_seconds,
        proximity_approach_min_weight=proximity_approach_min_weight,
        missed_contact_grace_steps=missed_contact_grace_steps,
    )
    missed = state["missed_contact"]
    if max_curriculum_level is None:
        return missed
    command = env.command_manager.get_term(command_name)
    episode_level = getattr(command, "s2_episode_curriculum_level", None)
    if not isinstance(episode_level, torch.Tensor) or episode_level.shape[0] != env.num_envs:
        return missed
    return missed & (episode_level <= int(max_curriculum_level))


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

    cnt = torch.where(
        reset_m | ~active_task_state,
        torch.zeros_like(cnt),
        torch.where(
            past_grace,
            torch.where(has_contact, torch.zeros_like(cnt), cnt + increment),
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
