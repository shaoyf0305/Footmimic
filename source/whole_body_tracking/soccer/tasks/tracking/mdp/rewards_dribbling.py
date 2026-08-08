"""Dribbling-specific reward functions.

Encourages close ball control without strike-the-ball objectives. Contact
legality is **side-based**: the first ``num_ankle_links`` entries in
``all_body_cfg.body_names`` are ankle links (typically both feet), and an
instep is classified only as the inside or outside side in the contacted
foot's yaw frame. Knees/wrists listed after incur
``dribbling_undesired_contact_penalty`` when closest to the ball. No
``kick_leg`` labels are required.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude, yaw_quat

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import (
    TASK_STATE_DRIBBLE,
    TASK_STATE_IDLE,
    TASK_STATE_STOP,
    MotionCommand,
    locomotion_task_state_mask,
)
from soccer.tasks.tracking.mdp.rewards import motion_relative_foot_position_error_exp
from soccer.tasks.tracking.mdp.task_frame import (
    forward_dominance_gate,
    task_forward_offset,
    task_forward_speed,
    task_lateral_offset,
    task_pelvis_heading_cos_world_x,
    task_velocity_forward_dominance,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Shared: net contact force on the ball (world frame)
# ---------------------------------------------------------------------------


def soccer_ball_contact_net_force_w(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
) -> torch.Tensor:
    """Net contact force on the soccer ball body, shape ``(num_envs, 3)``.

    Zeros are returned when the sensor has no usable data (same convention as
    ``_identify_contact_body``).
    """
    device = env.device
    num_envs = env.num_envs
    zero = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)

    contact_sensor: ContactSensor = env.scene.sensors[ball_sensor_name]
    forces_data = contact_sensor.data

    forces = None
    if hasattr(forces_data, "net_forces_w_history"):
        fh = forces_data.net_forces_w_history
        if fh is not None and fh.numel() > 0:
            forces = fh.to(device)
            if forces.ndim >= 4:
                forces = forces.max(dim=1).values
    if forces is None:
        if hasattr(forces_data, "net_forces_w"):
            f = forces_data.net_forces_w
            if f is not None and f.numel() > 0:
                forces = f.to(device)

    if forces is None or forces.ndim < 2:
        return zero

    if forces.ndim == 3:
        return forces[:, 0, :]
    if forces.shape[-1] >= 3:
        return forces[:, :3]
    return zero


def soccer_ball_contact_force_magnitude(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    *,
    horizontal_only: bool = True,
) -> torch.Tensor:
    """Scalar contact-force magnitude on the ball, shape ``(num_envs,)``.

    Defaults to horizontal (XY) magnitude so ground support (mostly Z) is not
    counted as robot-ball contact — same convention as
    ``terminations.contact_phase_violation``.
    """
    f = soccer_ball_contact_net_force_w(env, ball_sensor_name)
    if horizontal_only:
        return torch.norm(f[:, :2], dim=-1)
    return torch.norm(f, dim=-1)


def _dribbling_sim_contact(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str,
    contact_force_threshold: float,
) -> torch.Tensor:
    """Bool per env: horizontal ball contact force above threshold."""
    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    return fmag > contact_force_threshold


def _dribbling_effective_cg_label_mask(command: MotionCommand, labeled: torch.Tensor) -> torch.Tensor:
    """Disable CG-only objectives in an optional synthesized style seam."""
    seam = getattr(command, "style_seam_bridge_mask", None)
    if isinstance(seam, torch.Tensor) and seam.shape == labeled.shape:
        return labeled & ~seam
    return labeled


def _dribbling_cg_gated_sim_contact(
    command: MotionCommand,
    sim_contact: torch.Tensor,
) -> torch.Tensor:
    """On CG-labeled clips, only count touches on annotated contact frames."""
    if not hasattr(command, "motion_has_dribble_cg_label"):
        return sim_contact
    labeled = _dribbling_effective_cg_label_mask(command, command.motion_has_dribble_cg_label)
    if not torch.any(labeled):
        return sim_contact
    ref = command.dribble_cg_contact_ref
    return torch.where(labeled, sim_contact & ref, sim_contact)


def _dribbling_recent_contact_gate(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str,
    contact_force_threshold: float,
    recent_contact_window: int,
    buf_name: str = "_dribbling_steps_since_contact",
    command: MotionCommand | None = None,
    cg_gated: bool = False,
) -> torch.Tensor:
    """Per-env gate in ``[0, 1]``: 1 iff robot-ball contact within the last N steps."""
    sim_contact = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    if cg_gated and command is not None:
        has_contact = _dribbling_cg_gated_sim_contact(command, sim_contact)
    else:
        has_contact = sim_contact

    if recent_contact_window <= 0:
        return has_contact.to(torch.float32)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        step_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    reset_mask = step_buf == 0

    cnt = getattr(env, buf_name, None)
    if cnt is None or cnt.shape[0] != env.num_envs:
        cnt = torch.full(
            (env.num_envs,),
            fill_value=recent_contact_window + 1,
            device=env.device,
            dtype=torch.int32,
        )

    cnt = torch.where(
        reset_mask,
        torch.full_like(cnt, recent_contact_window + 1),
        torch.where(has_contact, torch.zeros_like(cnt), cnt + 1),
    )
    setattr(env, buf_name, cnt)
    return (cnt <= int(recent_contact_window)).to(torch.float32)


def _command_direction_xy(command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the active locomotion direction and speed in world XY.

    The task frame is fixed to world ``+X`` for the legacy forward task.  In
    control, however, the direction must come from the active external command
    so that reward geometry rotates together with a requested heading.
    """
    if hasattr(command, "locomotion_lin_vel_command_w"):
        command_vel_xy = command.locomotion_lin_vel_command_w()[:, :2]
    else:
        command_vel_xy = command.anchor_lin_vel_w[:, :2]
    speed = torch.norm(command_vel_xy, dim=-1)
    # At zero speed (IDLE/STOP), keep the last commanded heading as the ball
    # corridor.  Falling back unconditionally to +X would make a stopped
    # turning run suddenly evaluate its ball geometry in the wrong direction.
    heading = getattr(command, "locomotion_cmd_heading", None)
    if isinstance(heading, torch.Tensor):
        fallback = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    else:
        fallback = torch.zeros_like(command_vel_xy)
        fallback[:, 0] = 1.0
    direction = torch.where(
        (speed > 1.0e-4).unsqueeze(-1),
        command_vel_xy / speed.unsqueeze(-1).clamp(min=1.0e-4),
        fallback,
    )
    return direction, speed


def _command_frame_components(vector_xy: torch.Tensor, direction_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project XY vectors onto the commanded forward and lateral axes."""
    forward = torch.sum(vector_xy * direction_xy, dim=-1)
    lateral_axis = torch.stack((-direction_xy[:, 1], direction_xy[:, 0]), dim=-1)
    lateral = torch.sum(vector_xy * lateral_axis, dim=-1)
    return forward, lateral


def _pelvis_yaw_local_vector(command: MotionCommand, vector_w: torch.Tensor) -> torch.Tensor:
    """Express a world vector in the current pelvis yaw frame."""
    return quat_apply_inverse(yaw_quat(command.robot_pelvis_quat_w), vector_w)


def dribbling_idle_stand_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    linear_speed_std: float = 0.10,
    angular_speed_std: float = 0.35,
) -> torch.Tensor:
    """Reward a quiet robot only while the task explicitly waits in IDLE."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    pelvis_id = robot.body_names.index("pelvis")
    pelvis_linear_speed = torch.norm(robot.data.body_lin_vel_w[:, pelvis_id, :2], dim=-1)
    pelvis_angular_speed = torch.norm(robot.data.body_ang_vel_w[:, pelvis_id], dim=-1)
    reward = torch.exp(
        -torch.square(pelvis_linear_speed) / max(float(linear_speed_std), 1.0e-6) ** 2
        -torch.square(pelvis_angular_speed) / max(float(angular_speed_std), 1.0e-6) ** 2
    )
    active = locomotion_task_state_mask(command, (TASK_STATE_IDLE,))
    setattr(env, "_dribbling_idle_active", active)
    setattr(env, "_dribbling_idle_pelvis_speed", pelvis_linear_speed)
    setattr(env, "_dribbling_idle_pelvis_angular_speed", pelvis_angular_speed)
    return reward * active.to(reward.dtype)


def dribbling_stop_settle_state(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    settled_pelvis_speed: float = 0.05,
    settled_pelvis_angular_speed: float = 0.20,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return the STOP success predicate and its underlying physical measurements.

    The reward and successful-STOP termination call this same helper so an
    episode cannot reset under criteria different from those it was trained to
    optimize.  STOP is intentionally robot-only: the ball may continue rolling
    or leave the control corridor after the final dribble touch.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    pelvis_id = robot.body_names.index("pelvis")
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    forward_offset, lateral_offset = _command_frame_components(offset_xy, direction_xy)
    ball_speed = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    pelvis_speed = torch.norm(robot.data.body_lin_vel_w[:, pelvis_id, :2], dim=-1)
    pelvis_angular_speed = torch.norm(robot.data.body_ang_vel_w[:, pelvis_id], dim=-1)
    active = locomotion_task_state_mask(command, (TASK_STATE_STOP,))
    settled = (
        active
        & (pelvis_speed <= float(settled_pelvis_speed))
        & (pelvis_angular_speed <= float(settled_pelvis_angular_speed))
    )
    return active, pelvis_speed, pelvis_angular_speed, ball_speed, forward_offset, lateral_offset, settled


def dribbling_stop_settle_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    pelvis_speed_std: float = 0.10,
    pelvis_angular_speed_std: float = 0.35,
    settled_pelvis_speed: float = 0.05,
    settled_pelvis_angular_speed: float = 0.20,
) -> torch.Tensor:
    """Reward a stable robot STOP without constraining the ball's outcome."""
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
    speed_score = torch.exp(
        -torch.square(pelvis_speed) / max(float(pelvis_speed_std), 1.0e-6) ** 2
        -torch.square(pelvis_angular_speed) / max(float(pelvis_angular_speed_std), 1.0e-6) ** 2
    )
    setattr(env, "_dribbling_stop_active", active)
    setattr(env, "_dribbling_stop_pelvis_speed", pelvis_speed)
    setattr(env, "_dribbling_stop_pelvis_angular_speed", pelvis_angular_speed)
    setattr(env, "_dribbling_stop_ball_speed", ball_speed)
    setattr(env, "_dribbling_stop_forward_offset", forward_offset)
    setattr(env, "_dribbling_stop_lateral_offset", lateral_offset)
    setattr(env, "_dribbling_stop_position_score", torch.full_like(pelvis_speed, float("nan")))
    setattr(env, "_dribbling_stop_speed_score", speed_score)
    setattr(env, "_dribbling_stop_settled", settled)
    return speed_score * active.to(speed_score.dtype)


# ---------------------------------------------------------------------------
# 1) Velocity Tracking  — ball vel aligned with pelvis vel
# ---------------------------------------------------------------------------


def dribbling_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 1.0,
    pelvis_speed_min: float = 0.0,
    ball_speed_min: float = 0.0,
    require_contact: bool = False,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    recent_contact_window: int = 0,
    cg_gated_contact: bool = False,
    min_forward_dominance: float = 0.55,
) -> torch.Tensor:
    """Reward matching task +X (forward) speed between ball and pelvis after a touch.

    Only the forward (+X) velocity components are compared so lateral / crab-walking
    sync (matching ``v_y`` while ``v_x`` is small) does not score highly.

    Optional **anti-cheese gates**:

    - ``pelvis_speed_min`` / ``ball_speed_min``: multiply the reward by
      ``clamp(|v_xy| / min, max=1)`` so near-zero speeds do not yield a full
      score from ``exp(0)==1``.
    - ``require_contact``: multiply by a contact gate. If
      ``recent_contact_window > 0``, contact must have occurred within the last
      N steps (shared counter with forward-progress reward); otherwise the
      current step must show contact.
    - ``min_forward_dominance``: both ball and pelvis XY velocities must be
      sufficiently forward-dominant (task +X share of speed).

    Returns a value in ``[0, 1]`` per environment.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
    pelvis_vel_w = command.robot_anchor_lin_vel_w[:, :3]
    ball_vel_xy = ball_vel_w[:, :2]
    pelvis_vel_xy = pelvis_vel_w[:, :2]

    # Task-frame forward speed only — lateral crab sync must not match.
    ball_fwd = task_forward_speed(ball_vel_w, clamp_forward=False)
    pelvis_fwd = task_forward_speed(pelvis_vel_w, clamp_forward=False)
    error = (ball_fwd - pelvis_fwd) ** 2
    base = torch.exp(-error / (std ** 2))

    pelvis_sp = torch.norm(pelvis_vel_xy, dim=-1)
    ball_sp = torch.norm(ball_vel_xy, dim=-1)

    gate = torch.ones_like(base)
    if pelvis_speed_min > 0.0:
        gate = gate * torch.clamp(pelvis_sp / pelvis_speed_min, max=1.0)
    if ball_speed_min > 0.0:
        gate = gate * torch.clamp(ball_sp / ball_speed_min, max=1.0)

    if require_contact:
        gate = gate * _dribbling_recent_contact_gate(
            env,
            ball_sensor_name,
            contact_force_threshold,
            recent_contact_window,
            command=command,
            cg_gated=cg_gated_contact,
        )

    if min_forward_dominance > 0.0:
        pelvis_dom = task_velocity_forward_dominance(pelvis_vel_xy)
        ball_dom = task_velocity_forward_dominance(ball_vel_xy)
        gate = gate * forward_dominance_gate(pelvis_dom, min_forward_dominance)
        gate = gate * forward_dominance_gate(ball_dom, min_forward_dominance)

    return base * gate


# ---------------------------------------------------------------------------
# 2) Dynamic Proximity  — ball inside the "safe zone" in front of the robot
# ---------------------------------------------------------------------------

def dribbling_dynamic_proximity(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    near_dist: float = 0.2,
    far_dist: float = 0.5,
    penalty_std: float = 0.15,
    pelvis_speed_min: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    no_contact_zone_damping: float = 1.0,
    zone_lateral_abs_max: float = 0.18,
) -> torch.Tensor:
    """Reward keeping the ball inside a longitudinal safe-zone ahead on task +X.

    Ball offset uses env-local / world-parallel axes (pelvis → ball):
    - x_task in [near_dist, far_dist] → reward = 1.0
    - Outside that range → exponential decay with ``penalty_std``

    Lateral deviation (|y_task|) is also penalised with the same decay.

    If ``pelvis_speed_min > 0``, the reward is multiplied by
    ``clamp(|v_pelvis_xy| / pelvis_speed_min, max=1)`` so standing still in the
    safe zone is not a local optimum.

    If ``no_contact_zone_damping < 1``, when the ball sits in the forward corridor
    (longitudinal band + ``|y_task| <= zone_lateral_abs_max``) but the ball
    sensor reports no contact, the proximity reward is scaled by that factor.
    This reduces the \"park in front of the ball and wiggle\" optimum.

    Returns a value in [0, 1] per environment.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w  # (N, 3)
    pelvis_pos_w = command.robot_pelvis_pos_w  # (N, 3)

    x_task = task_forward_offset(ball_pos_w, pelvis_pos_w)
    y_task = task_lateral_offset(ball_pos_w, pelvis_pos_w)

    x_error = torch.where(
        x_task < near_dist,
        near_dist - x_task,
        torch.where(
            x_task > far_dist,
            x_task - far_dist,
            torch.zeros_like(x_task),
        ),
    )

    y_error = torch.abs(y_task)

    total_error = x_error ** 2 + y_error ** 2
    proximity_reward = torch.exp(-total_error / (penalty_std ** 2))

    if pelvis_speed_min > 0.0:
        pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
        pelvis_sp = torch.norm(pelvis_vel_xy, dim=-1)
        proximity_reward = proximity_reward * torch.clamp(pelvis_sp / pelvis_speed_min, max=1.0)

    if no_contact_zone_damping < 1.0 - 1e-6:
        in_corridor = (
            (x_task >= near_dist)
            & (x_task <= far_dist)
            & (torch.abs(y_task) <= zone_lateral_abs_max)
        )
        fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
        no_touch = fmag <= contact_force_threshold
        damp = torch.where(
            in_corridor & no_touch,
            torch.full_like(proximity_reward, no_contact_zone_damping),
            torch.ones_like(proximity_reward),
        )
        proximity_reward = proximity_reward * damp

    return proximity_reward


def dribbling_command_dynamic_proximity(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    near_dist: float = 0.2,
    far_dist: float = 0.5,
    penalty_std: float = 0.15,
    pelvis_speed_min: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    no_contact_zone_damping: float = 1.0,
    zone_lateral_abs_max: float = 0.18,
) -> torch.Tensor:
    """Keep the ball in a safe corridor ahead along the requested heading."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    forward_offset, lateral_offset = _command_frame_components(offset_xy, direction_xy)

    forward_error = torch.where(
        forward_offset < near_dist,
        near_dist - forward_offset,
        torch.where(forward_offset > far_dist, forward_offset - far_dist, torch.zeros_like(forward_offset)),
    )
    total_error = forward_error.square() + lateral_offset.square()
    reward = torch.exp(-total_error / max(penalty_std, 1.0e-6) ** 2)

    if pelvis_speed_min > 0.0:
        pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
        reward = reward * torch.clamp(pelvis_speed / pelvis_speed_min, max=1.0)

    if no_contact_zone_damping < 1.0 - 1.0e-6:
        in_corridor = (
            (forward_offset >= near_dist)
            & (forward_offset <= far_dist)
            & (torch.abs(lateral_offset) <= zone_lateral_abs_max)
        )
        no_touch = soccer_ball_contact_force_magnitude(env, ball_sensor_name) <= contact_force_threshold
        reward = torch.where(in_corridor & no_touch, reward * no_contact_zone_damping, reward)
    return reward


def dribbling_pelvis_local_dynamic_proximity(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    near_dist: float = 0.2,
    far_dist: float = 0.5,
    penalty_std: float = 0.15,
    pelvis_speed_min: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    no_contact_zone_damping: float = 1.0,
    zone_lateral_abs_max: float = 0.18,
) -> torch.Tensor:
    """Keep the ball in front of the *current* pelvis, with no world axis.

    Unlike the task/world-frame corridor, this stays meaningful after any
    accumulated yaw drift and is therefore suitable for a local-twist policy.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    offset_b = _pelvis_yaw_local_vector(
        command, soccer_ball.data.root_pos_w[:, :3] - command.robot_pelvis_pos_w
    )
    forward_offset, lateral_offset = offset_b[:, 0], offset_b[:, 1]
    forward_error = torch.where(
        forward_offset < near_dist,
        near_dist - forward_offset,
        torch.where(forward_offset > far_dist, forward_offset - far_dist, torch.zeros_like(forward_offset)),
    )
    reward = torch.exp(-(forward_error.square() + lateral_offset.square()) / max(penalty_std, 1.0e-6) ** 2)
    if pelvis_speed_min > 0.0:
        pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
        reward = reward * torch.clamp(pelvis_speed / pelvis_speed_min, max=1.0)
    if no_contact_zone_damping < 1.0 - 1.0e-6:
        in_corridor = (
            (forward_offset >= near_dist)
            & (forward_offset <= far_dist)
            & (torch.abs(lateral_offset) <= zone_lateral_abs_max)
        )
        no_touch = soccer_ball_contact_force_magnitude(env, ball_sensor_name) <= contact_force_threshold
        reward = torch.where(in_corridor & no_touch, reward * no_contact_zone_damping, reward)
    return reward


# ---------------------------------------------------------------------------
# 2a) Stall in front of ball without touching — kills "back up, then freeze"
# ---------------------------------------------------------------------------


def dribbling_stall_no_touch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    max_xy_dist: float = 0.52,
    pelvis_speed_max: float = 0.16,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Penalty in ``[0, 1]`` when the ball is close in XY but pelvis is nearly static and there is no contact.

    Targets the local optimum: robot brings the ball into a comfortable pose in
    front of the body, then stops or only sways without registering foot-ball
    contact on the ball sensor.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]
    dist_xy = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)

    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
    pelvis_sp = torch.norm(pelvis_vel_xy, dim=-1)

    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    no_touch = fmag <= contact_force_threshold

    near = dist_xy <= max_xy_dist
    slow = pelvis_sp <= pelvis_speed_max
    active = locomotion_task_state_mask(command, active_task_states)
    return (near & slow & no_touch & active).to(torch.float32)


# ---------------------------------------------------------------------------
# Helper: identify which robot body caused the ball contact
# ---------------------------------------------------------------------------

def _identify_contact_body(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    ball_sensor_name: str,
    all_body_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify which robot body is closest to the ball when contact occurs.

    Returns:
        has_contact: (N,) bool — whether ball has non-zero contact force
        contact_force_mag: (N,) float — force magnitude on ball
        closest_body_idx: (N,) long — index into all_body_cfg.body_names
                                       of the body closest to ball
    """
    device = env.device
    num_envs = env.num_envs

    # Default outputs
    has_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    closest_body_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

    force_vec = soccer_ball_contact_net_force_w(env, ball_sensor_name)
    # Robot-foot contact appears in XY; ball-on-ground support is mostly Z.
    force_mag = torch.norm(force_vec[:, :2], dim=-1)
    has_contact = force_mag > 1.0  # minimal threshold to filter sensor noise
    contact_force_mag = force_mag

    if not torch.any(has_contact):
        return has_contact, contact_force_mag, closest_body_idx

    # For envs with contact, find which robot body is closest to ball
    robot = env.scene[all_body_cfg.name]
    soccer_ball = env.scene["soccer_ball"]

    # Resolve body indices (cached; invalidate if body list changes)
    cache_name = "_dribbling_body_indices_cache"
    cached = getattr(env, cache_name, None)
    names_t = tuple(all_body_cfg.body_names)
    if cached is None or cached.get("names") != names_t:
        body_indices = torch.as_tensor(
            robot.find_bodies(all_body_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long, device=device,
        )
        setattr(env, cache_name, {"names": names_t, "idx": body_indices})
    body_indices = getattr(env, cache_name)["idx"]

    # Contact envs only
    contact_env_ids = torch.nonzero(has_contact, as_tuple=False).squeeze(-1)
    body_pos = robot.data.body_pos_w[contact_env_ids][:, body_indices]  # (M, B, 3)
    ball_pos = soccer_ball.data.root_pos_w[contact_env_ids]  # (M, 3)

    dist = torch.norm(body_pos - ball_pos.unsqueeze(1), dim=-1)  # (M, B)
    closest = torch.argmin(dist, dim=-1)  # (M,) — index into body_names
    closest_body_idx[contact_env_ids] = closest

    return has_contact, contact_force_mag, closest_body_idx


def _is_dribble_legal_ankle_contact(closest_body_idx: torch.Tensor, num_ankle_links: int) -> torch.Tensor:
    """True when the closest link index is one of the leading ankle entries."""
    if num_ankle_links <= 0:
        return torch.zeros_like(closest_body_idx, dtype=torch.bool)
    return closest_body_idx < num_ankle_links


_DRIBBLE_CONTACT_SURFACES = ("any", "instep", "inside_instep", "outside_instep")


def _dribbling_contact_surface_geometry(
    env: ManagerBasedRLEnv,
    all_body_cfg: SceneEntityCfg | None,
    closest_body_idx: torch.Tensor,
    *,
    num_ankle_links: int,
    medial_y_min: float,
) -> dict[str, torch.Tensor]:
    """Compute the shared foot-yaw inside/outside contact geometry.

    The reward and playback diagnostic both consume this helper. Keeping the
    coordinates and thresholds in one place makes a recorded ``instep`` match
    exactly the match used by the learning objective.
    """
    device = env.device
    num_envs = env.num_envs
    dtype = torch.float32
    empty = {
        "ball_offset_foot_yaw": torch.full((num_envs, 3), torch.nan, device=device, dtype=dtype),
        "medial_offset": torch.full((num_envs,), torch.nan, device=device, dtype=dtype),
        "medial_sign": torch.zeros(num_envs, device=device, dtype=dtype),
        "legal_ankle": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "known_foot": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "instep": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "inside_instep": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "outside_instep": torch.zeros(num_envs, dtype=torch.bool, device=device),
    }
    if all_body_cfg is None or num_ankle_links <= 0:
        return empty
    if medial_y_min < 0.0:
        raise ValueError("medial_y_min must be non-negative.")

    names = tuple(all_body_cfg.body_names)
    if not names:
        return empty

    robot = env.scene[all_body_cfg.name]
    cache_name = "_dribbling_body_indices_cache"
    cached = getattr(env, cache_name, None)
    if cached is None or cached.get("names") != names:
        body_indices = torch.as_tensor(
            robot.find_bodies(all_body_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=device,
        )
        setattr(env, cache_name, {"names": names, "idx": body_indices})
    body_indices = getattr(env, cache_name)["idx"]

    selected_cfg_idx = closest_body_idx.clamp(min=0, max=len(names) - 1)
    selected_body_idx = body_indices[selected_cfg_idx]
    batch_ids = torch.arange(num_envs, device=device)
    foot_pos_w = robot.data.body_pos_w[batch_ids, selected_body_idx]
    foot_yaw_w = yaw_quat(robot.data.body_quat_w[batch_ids, selected_body_idx])
    ball_pos_w = env.scene["soccer_ball"].data.root_pos_w[:, :3]
    ball_local = quat_apply_inverse(foot_yaw_w, ball_pos_w - foot_pos_w)

    # In each foot yaw frame +Y is its local left side. Thus the right foot's
    # medial direction is +Y while the left foot's medial direction is -Y.
    medial_sign = torch.zeros(num_envs, dtype=ball_local.dtype, device=device)
    for cfg_idx, body_name in enumerate(names):
        if "right_ankle" in body_name:
            medial_sign[selected_cfg_idx == cfg_idx] = 1.0
        elif "left_ankle" in body_name:
            medial_sign[selected_cfg_idx == cfg_idx] = -1.0

    medial_offset = ball_local[:, 1] * medial_sign
    known_foot = medial_sign != 0.0
    legal_ankle = _is_dribble_legal_ankle_contact(closest_body_idx, num_ankle_links)
    inside_instep = medial_offset >= medial_y_min
    outside_instep = medial_offset <= -medial_y_min
    return {
        "ball_offset_foot_yaw": ball_local,
        "medial_offset": medial_offset,
        "medial_sign": medial_sign,
        "legal_ankle": legal_ankle,
        "known_foot": known_foot,
        "instep": inside_instep | outside_instep,
        "inside_instep": inside_instep,
        "outside_instep": outside_instep,
    }


def _dribbling_cg_reference_surface_match(
    command: MotionCommand,
    geometry: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Match the current inside/outside region to the per-frame CG label.

    ``dribble_cg_surface`` stores ``0=inside_instep`` and
    ``1=outside_instep``. A missing label never silently becomes a legal
    reference-surface touch: S2 must either supply the label or use a task
    that does not request reference-surface gating.
    """
    reference_surface = getattr(command, "dribble_cg_surface_ref", None)
    if not isinstance(reference_surface, torch.Tensor):
        return torch.zeros_like(geometry["legal_ankle"])
    return (
        ((reference_surface == 0) & geometry["inside_instep"])
        | ((reference_surface == 1) & geometry["outside_instep"])
    )


def _dribbling_contact_surface_match(
    env: ManagerBasedRLEnv,
    all_body_cfg: SceneEntityCfg | None,
    closest_body_idx: torch.Tensor,
    *,
    command_name: str = "motion",
    contact_surface: str,
    num_ankle_links: int,
    medial_y_min: float = 0.018,
    cg_surface_gated: bool = False,
) -> torch.Tensor:
    """Check a requested instep side in the contacted foot's yaw frame.

    Isaac's ball contact sensor reports the net force on the ball but not the
    individual collision capsule that produced it.  We therefore identify the
    closest contacted ankle link and express the ankle-to-ball offset in that
    foot's yaw frame. Only its signed lateral coordinate is used: right-foot
    medial is ``+Y`` and left-foot medial is ``-Y``. There is deliberately no
    fore/aft or height gate; the task distinguishes only inside from outside.
    """
    surface = str(contact_surface).lower().strip()
    if surface not in _DRIBBLE_CONTACT_SURFACES:
        raise ValueError(
            f"Unsupported contact_surface={contact_surface!r}; expected one of {_DRIBBLE_CONTACT_SURFACES}."
        )
    if surface == "any" and not cg_surface_gated:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    geometry = _dribbling_contact_surface_geometry(
        env,
        all_body_cfg,
        closest_body_idx,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
    )
    if surface == "any":
        surface_geometry = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    elif surface == "instep":
        surface_geometry = geometry["instep"]
    elif surface == "inside_instep":
        surface_geometry = geometry["inside_instep"]
    else:
        surface_geometry = geometry["outside_instep"]
    match = geometry["legal_ankle"] & geometry["known_foot"] & surface_geometry
    if cg_surface_gated:
        command: MotionCommand = env.command_manager.get_term(command_name)
        match = match & _dribbling_cg_reference_surface_match(command, geometry)
    return match


def _dribbling_band_error(
    value: torch.Tensor, lower: float, upper: float
) -> torch.Tensor:
    """Distance to a closed scalar interval, or zero while inside it."""
    if lower > upper:
        raise ValueError(f"Invalid S2 target interval [{lower}, {upper}]")
    return torch.relu(float(lower) - value) + torch.relu(value - float(upper))


def dribbling_s2_contact_event_state(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    max_touch_force: float = 100.0,
    soft_touch_force_start: float = 60.0,
    side_deadzone: float = 0.04,
    target_forward_min: float = -0.06,
    target_forward_max: float = 0.14,
    target_side_max: float = 0.16,
    target_region_std: float = 0.12,
    missed_contact_grace_steps: int = 3,
) -> dict[str, torch.Tensor]:
    """Evaluate one S2 contact event and update its shared state machine.

    The result is cached for the current environment step.  Contact
    proximity, new-touch, side bonus, diagnostics, and missed-contact
    termination therefore consume exactly the same physical event.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    required = (
        "s2_contact_event_id_ref",
        "s2_contact_event_frame_ref",
        "s2_contact_event_foot_ref",
        "s2_contact_event_side_ref",
        "s2_contact_reference_foot_pose_w",
    )
    if all_body_cfg is None or any(not hasattr(command, name) for name in required):
        zero = torch.zeros(env.num_envs, device=env.device)
        false = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return {
            "contact_window": false,
            "contact_proximity": zero,
            "target_region_distance": zero,
            "new_touch": false,
            "correct_side_touch": false,
            "touch_force_soft_penalty": zero,
            "over_force": false,
            "dead_zone": false,
            "wrong_foot": false,
            "wrong_side": false,
            "invalid_body_contact": false,
            "undesired_contact_penalty": zero,
            "missed_contact": false,
            "timing_error_seconds": zero,
        }

    step = getattr(
        env, "episode_length_buf", torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    ).to(torch.long)
    frame = command.time_steps.to(torch.long)
    motion_idx = command.motion_idx.to(torch.long)
    generation = getattr(command, "s2_episode_generation", None)
    if not isinstance(generation, torch.Tensor) or generation.shape[0] != env.num_envs:
        generation = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    cached = getattr(env, "_dribbling_s2_contact_event_cache", None)
    if (
        isinstance(cached, dict)
        and isinstance(cached.get("step"), torch.Tensor)
        and torch.equal(cached["step"], step)
        and isinstance(cached.get("frame"), torch.Tensor)
        and torch.equal(cached["frame"], frame)
        and isinstance(cached.get("motion_idx"), torch.Tensor)
        and torch.equal(cached["motion_idx"], motion_idx)
        and isinstance(cached.get("generation"), torch.Tensor)
        and torch.equal(cached["generation"], generation)
    ):
        return cached["state"]

    previous_generation = getattr(env, "_dribbling_s2_episode_generation", None)
    if not isinstance(previous_generation, torch.Tensor) or previous_generation.shape[0] != env.num_envs:
        reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        reset = generation != previous_generation
    setattr(env, "_dribbling_s2_episode_generation", generation.detach().clone())
    event_id = command.s2_contact_event_id_ref
    event_frame = command.s2_contact_event_frame_ref
    expected_foot = command.s2_contact_event_foot_ref
    expected_side = command.s2_contact_event_side_ref
    active = (event_id >= 0) & ((expected_foot == 0) | (expected_foot == 1))

    reference_foot_pos_w, reference_foot_yaw_w = command.s2_contact_reference_foot_pose_w()
    ball_pos_w = env.scene["soccer_ball"].data.root_pos_w[:, :3]
    ball_from_reference_foot = quat_apply_inverse(
        reference_foot_yaw_w, ball_pos_w - reference_foot_pos_w
    )

    def _target_region_distance(ball_from_foot: torch.Tensor) -> torch.Tensor:
        forward_error = _dribbling_band_error(
            ball_from_foot[:, 0], target_forward_min, target_forward_max
        )
        left_error = _dribbling_band_error(
            ball_from_foot[:, 1], side_deadzone, target_side_max
        )
        right_error = _dribbling_band_error(
            ball_from_foot[:, 1], -target_side_max, -side_deadzone
        )
        labeled_side_error = torch.where(expected_side == 0, left_error, right_error)
        side_error = (
            labeled_side_error if target_side_enabled else torch.minimum(left_error, right_error)
        )
        return torch.sqrt(forward_error.square() + side_error.square())

    # Retain the frozen-reference distance for visualization and offline
    # auditing, but do not use it as the trainable proximity signal.  The
    # reset ball is already near that reference region, so rewarding it every
    # window frame produced a large action-independent return.
    reference_target_distance = _target_region_distance(ball_from_reference_foot)

    has_contact, force_magnitude, closest_body = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg
    )
    names = tuple(all_body_cfg.body_names)
    try:
        left_cfg_index = names.index("left_ankle_roll_link")
        right_cfg_index = names.index("right_ankle_roll_link")
    except ValueError as exc:
        raise ValueError("S2 contact body config must contain both ankle roll links") from exc
    expected_body = torch.where(
        expected_foot == 1,
        torch.full_like(expected_foot, right_cfg_index),
        torch.full_like(expected_foot, left_cfg_index),
    )
    legal_ankle = _is_dribble_legal_ankle_contact(closest_body, num_ankle_links)
    correct_foot = legal_ankle & (closest_body == expected_body)

    robot = env.scene[all_body_cfg.name]
    pose_cache_name = "_dribbling_s2_foot_pose_indices"
    pose_indices = getattr(env, pose_cache_name, None)
    if not isinstance(pose_indices, torch.Tensor):
        resolved = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )[0]
        pose_indices = torch.as_tensor(resolved, dtype=torch.long, device=env.device)
        setattr(env, pose_cache_name, pose_indices)
    selected_pose_index = torch.where(
        expected_foot == 1,
        torch.full_like(expected_foot, int(pose_indices[1].item())),
        torch.full_like(expected_foot, int(pose_indices[0].item())),
    )
    batch = torch.arange(env.num_envs, device=env.device)
    actual_foot_pos = robot.data.body_pos_w[batch, selected_pose_index]
    actual_foot_yaw = yaw_quat(robot.data.body_quat_w[batch, selected_pose_index])
    ball_from_actual_foot = quat_apply_inverse(
        actual_foot_yaw, ball_pos_w - actual_foot_pos
    )
    current_lateral = ball_from_actual_foot[:, 1]
    target_distance = _target_region_distance(ball_from_actual_foot)
    proximity = torch.exp(
        -target_distance.square() / max(float(target_region_std), 1.0e-6) ** 2
    ) * active.to(torch.float32)

    previous_contact = getattr(env, "_dribbling_s2_previous_contact", None)
    previous_lateral = getattr(env, "_dribbling_s2_previous_lateral", None)
    previous_event_id = getattr(env, "_dribbling_s2_previous_event_id", None)
    if not isinstance(previous_contact, torch.Tensor) or previous_contact.shape[0] != env.num_envs:
        previous_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not isinstance(previous_lateral, torch.Tensor) or previous_lateral.shape[0] != env.num_envs:
        previous_lateral = current_lateral
    if not isinstance(previous_event_id, torch.Tensor) or previous_event_id.shape[0] != env.num_envs:
        previous_event_id = torch.full_like(event_id, -1)
    previous_contact = previous_contact & ~reset
    lateral_for_touch = torch.where(
        (previous_event_id == event_id) & ~reset, previous_lateral, current_lateral
    )
    new_physical_contact = has_contact & ~previous_contact & ~reset
    setattr(env, "_dribbling_s2_previous_contact", has_contact.detach().clone())
    setattr(env, "_dribbling_s2_previous_lateral", current_lateral.detach().clone())
    setattr(env, "_dribbling_s2_previous_event_id", event_id.detach().clone())

    side_known = (expected_side == 0) | (expected_side == 1)
    side_correct = (
        ((expected_side == 0) & (lateral_for_touch >= float(side_deadzone)))
        | ((expected_side == 1) & (lateral_for_touch <= -float(side_deadzone)))
    )
    side_wrong = (
        ((expected_side == 0) & (lateral_for_touch <= -float(side_deadzone)))
        | ((expected_side == 1) & (lateral_for_touch >= float(side_deadzone)))
    )
    force_valid = force_magnitude <= float(max_touch_force)
    selected_foot_valid = correct_foot if require_expected_foot else legal_ankle
    over_force = (
        active & new_physical_contact & selected_foot_valid & ~force_valid
    )
    valid_touch = active & new_physical_contact & selected_foot_valid & force_valid

    tracked_event = getattr(env, "_dribbling_s2_tracked_event", None)
    event_succeeded = getattr(env, "_dribbling_s2_event_succeeded", None)
    grace_count = getattr(env, "_dribbling_s2_event_grace_count", None)
    if not isinstance(tracked_event, torch.Tensor) or tracked_event.shape[0] != env.num_envs:
        tracked_event = torch.full_like(event_id, -1)
    if not isinstance(event_succeeded, torch.Tensor) or event_succeeded.shape[0] != env.num_envs:
        event_succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not isinstance(grace_count, torch.Tensor) or grace_count.shape[0] != env.num_envs:
        grace_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    tracked_event = torch.where(reset, torch.full_like(tracked_event, -1), tracked_event)
    event_succeeded = torch.where(reset, torch.zeros_like(event_succeeded), event_succeeded)
    grace_count = torch.where(reset, torch.zeros_like(grace_count), grace_count)

    transitioned = active & (tracked_event >= 0) & (event_id != tracked_event)
    deadline = (
        ~active
        & (tracked_event >= 0)
        & (grace_count >= max(0, int(missed_contact_grace_steps)))
    )
    completed = transitioned | deadline
    completed_success = completed & event_succeeded
    missed_contact = completed & ~event_succeeded

    tracked_event = torch.where(completed, torch.full_like(tracked_event, -1), tracked_event)
    event_succeeded = torch.where(completed, torch.zeros_like(event_succeeded), event_succeeded)
    grace_count = torch.where(completed, torch.zeros_like(grace_count), grace_count)
    new_event = active & (tracked_event != event_id)
    tracked_event = torch.where(new_event, event_id, tracked_event)
    event_succeeded = torch.where(new_event, torch.zeros_like(event_succeeded), event_succeeded)
    grace_count = torch.where(new_event | active, torch.zeros_like(grace_count), grace_count)

    new_touch = valid_touch & ~event_succeeded
    correct_side_touch = new_touch & side_known & side_correct
    wrong_side = new_touch & side_known & side_wrong
    dead_zone = new_touch & side_known & ~side_correct & ~side_wrong
    if not 0.0 <= float(soft_touch_force_start) < float(max_touch_force):
        raise ValueError(
            "soft_touch_force_start must satisfy 0 <= start < max_touch_force; "
            f"got {soft_touch_force_start} and {max_touch_force}"
        )
    force_scale = max(float(max_touch_force) - float(soft_touch_force_start), 1.0e-6)
    force_excess = torch.relu(force_magnitude - float(soft_touch_force_start)) / force_scale
    # The hard 100 N validity cap remains binary.  This separate continuous
    # signal starts at 60 N and grows through (and moderately beyond) the cap,
    # allowing PPO to learn gentler impacts before a touch becomes invalid.
    touch_force_soft_penalty = torch.clamp(force_excess, max=2.0) * (
        active & new_physical_contact & selected_foot_valid
    ).to(force_excess.dtype)
    event_succeeded = event_succeeded | new_touch

    # A full-clip curriculum cannot provide grace frames beyond the source
    # boundary.  Settle its final active event on that boundary so the last
    # touch is not omitted from success/miss metrics.  Short curricula exclude
    # such terminal events and therefore retain their configured grace period.
    source_last_frame = command.motion.file_lengths[command.motion_idx] - 1
    terminal_completed = active & (tracked_event == event_id) & (frame >= source_last_frame)
    completed = completed | terminal_completed
    completed_success = completed_success | (terminal_completed & event_succeeded)
    missed_contact = missed_contact | (terminal_completed & ~event_succeeded)
    tracked_event = torch.where(
        terminal_completed, torch.full_like(tracked_event, -1), tracked_event
    )
    event_succeeded = torch.where(
        terminal_completed, torch.zeros_like(event_succeeded), event_succeeded
    )
    grace_count = torch.where(
        terminal_completed, torch.zeros_like(grace_count), grace_count
    )
    grace_count = torch.where(
        ~active & (tracked_event >= 0), grace_count + 1, grace_count
    )
    setattr(env, "_dribbling_s2_tracked_event", tracked_event)
    setattr(env, "_dribbling_s2_event_succeeded", event_succeeded)
    setattr(env, "_dribbling_s2_event_grace_count", grace_count)

    wrong_foot = active & new_physical_contact & legal_ankle & ~correct_foot & require_expected_foot
    invalid_body_contact = new_physical_contact & ~legal_ankle
    undesired_penalty = invalid_body_contact.to(torch.float32) + 0.5 * wrong_foot.to(torch.float32)
    fps = float(command.motion.fps.reshape(-1)[0])
    timing_error = (frame - event_frame).to(torch.float32) / max(fps, 1.0e-6)
    timing_error = torch.where(new_touch, timing_error, torch.zeros_like(timing_error))

    def _episode_counter(name: str, increment: torch.Tensor, dtype: torch.dtype = torch.float32):
        value = getattr(env, name, None)
        if not isinstance(value, torch.Tensor) or value.shape[0] != env.num_envs:
            value = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
        value = torch.where(reset, torch.zeros_like(value), value)
        value = value + increment.to(value.dtype)
        setattr(env, name, value)
        return value

    completed_count = _episode_counter("_dribbling_s2_completed_count", completed)
    success_count = _episode_counter("_dribbling_s2_success_count", completed_success)
    missed_count = _episode_counter("_dribbling_s2_missed_count", missed_contact)
    touch_attempt_count = _episode_counter(
        "_dribbling_s2_touch_attempt_count", active & new_physical_contact
    )
    correct_foot_count = _episode_counter(
        "_dribbling_s2_correct_foot_count", active & new_physical_contact & correct_foot & force_valid
    )
    side_attempt_count = _episode_counter("_dribbling_s2_side_attempt_count", new_touch & side_known)
    correct_side_count = _episode_counter("_dribbling_s2_correct_side_count", correct_side_touch)
    over_force_count = _episode_counter("_dribbling_s2_over_force_count", over_force)
    dead_zone_count = _episode_counter("_dribbling_s2_dead_zone_count", dead_zone)
    wrong_foot_count = _episode_counter("_dribbling_s2_wrong_foot_count", wrong_foot)
    wrong_side_count = _episode_counter("_dribbling_s2_wrong_side_count", wrong_side)
    invalid_body_count = _episode_counter("_dribbling_s2_invalid_body_count", invalid_body_contact)
    target_distance_sum = _episode_counter(
        "_dribbling_s2_target_distance_sum", target_distance * active.to(target_distance.dtype)
    )
    target_distance_count = _episode_counter("_dribbling_s2_target_distance_count", active)
    timing_signed_sum = _episode_counter("_dribbling_s2_timing_signed_sum", timing_error)
    timing_abs_sum = _episode_counter("_dribbling_s2_timing_abs_sum", torch.abs(timing_error))
    timing_count = _episode_counter("_dribbling_s2_timing_count", new_touch)

    streak = getattr(env, "_dribbling_s2_success_streak", None)
    max_streak = getattr(env, "_dribbling_s2_max_success_streak", None)
    if not isinstance(streak, torch.Tensor) or streak.shape[0] != env.num_envs:
        streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not isinstance(max_streak, torch.Tensor) or max_streak.shape[0] != env.num_envs:
        max_streak = torch.zeros_like(streak)
    streak = torch.where(reset, torch.zeros_like(streak), streak)
    max_streak = torch.where(reset, torch.zeros_like(max_streak), max_streak)
    streak = torch.where(completed, torch.where(completed_success, streak + 1, torch.zeros_like(streak)), streak)
    max_streak = torch.maximum(max_streak, streak)
    setattr(env, "_dribbling_s2_success_streak", streak)
    setattr(env, "_dribbling_s2_max_success_streak", max_streak)

    def _rate(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.where(denominator > 0, numerator / denominator.clamp(min=1.0), torch.zeros_like(numerator))

    command.metrics["s2_contact_success_rate"] = _rate(success_count, completed_count)
    command.metrics["s2_missed_contact_rate"] = _rate(missed_count, completed_count)
    command.metrics["s2_correct_foot_rate"] = _rate(correct_foot_count, touch_attempt_count)
    command.metrics["s2_correct_side_rate"] = _rate(correct_side_count, side_attempt_count)
    command.metrics["s2_contact_timing_error"] = _rate(timing_signed_sum, timing_count)
    command.metrics["s2_contact_timing_abs_error"] = _rate(timing_abs_sum, timing_count)
    command.metrics["s2_wrong_foot_count"] = wrong_foot_count
    command.metrics["s2_wrong_side_count"] = wrong_side_count
    command.metrics["s2_invalid_body_contact_count"] = invalid_body_count
    command.metrics["s2_target_region_distance"] = _rate(
        target_distance_sum, target_distance_count
    )
    command.metrics["s2_over_force_count"] = over_force_count
    command.metrics["s2_dead_zone_count"] = dead_zone_count
    command.metrics["s2_complete_2"] = (max_streak >= 2).to(torch.float32)
    command.metrics["s2_complete_4"] = (max_streak >= 4).to(torch.float32)
    command.metrics["s2_complete_8"] = (max_streak >= 8).to(torch.float32)

    state = {
        "contact_window": active,
        "contact_proximity": proximity,
        "target_region_distance": target_distance,
        "new_touch": new_touch,
        "correct_side_touch": correct_side_touch,
        "touch_force_soft_penalty": touch_force_soft_penalty,
        "over_force": over_force,
        "dead_zone": dead_zone,
        "wrong_foot": wrong_foot,
        "wrong_side": wrong_side,
        "invalid_body_contact": invalid_body_contact,
        "undesired_contact_penalty": undesired_penalty,
        "missed_contact": missed_contact,
        "timing_error_seconds": timing_error,
        "event_id": event_id,
        "event_frame": event_frame,
        "expected_foot": expected_foot,
        "expected_side": expected_side,
        "ball_from_reference_foot": ball_from_reference_foot,
        "reference_target_region_distance": reference_target_distance,
        "ball_from_actual_foot": ball_from_actual_foot,
    }
    setattr(env, "_dribbling_s2_contact_event_cache", {
        "step": step.detach().clone(),
        "frame": frame.detach().clone(),
        "motion_idx": motion_idx.detach().clone(),
        "generation": generation.detach().clone(),
        "state": state,
    })
    return state


def dribbling_s2_contact_proximity(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    max_touch_force: float = 100.0,
    soft_touch_force_start: float = 60.0,
    side_deadzone: float = 0.04,
    target_forward_min: float = -0.06,
    target_forward_max: float = 0.14,
    target_side_max: float = 0.16,
    target_region_std: float = 0.12,
    missed_contact_grace_steps: int = 3,
) -> torch.Tensor:
    """Continuous target-region signal, active only in a contact window."""
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        max_touch_force, soft_touch_force_start, side_deadzone, target_forward_min, target_forward_max,
        target_side_max, target_region_std, missed_contact_grace_steps,
    )["contact_proximity"]


def dribbling_s2_new_touch_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    max_touch_force: float = 100.0,
    soft_touch_force_start: float = 60.0,
    side_deadzone: float = 0.04,
    target_forward_min: float = -0.06,
    target_forward_max: float = 0.14,
    target_side_max: float = 0.16,
    target_region_std: float = 0.12,
    missed_contact_grace_steps: int = 3,
) -> torch.Tensor:
    """One reward pulse for the first valid specified-foot touch per event."""
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        max_touch_force, soft_touch_force_start, side_deadzone, target_forward_min, target_forward_max,
        target_side_max, target_region_std, missed_contact_grace_steps,
    )["new_touch"].to(torch.float32)


def dribbling_s2_correct_side_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    max_touch_force: float = 100.0,
    soft_touch_force_start: float = 60.0,
    side_deadzone: float = 0.04,
    target_forward_min: float = -0.06,
    target_forward_max: float = 0.14,
    target_side_max: float = 0.16,
    target_region_std: float = 0.12,
    missed_contact_grace_steps: int = 3,
) -> torch.Tensor:
    """Extra pulse when the new touch approaches from the labelled local side."""
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        max_touch_force, soft_touch_force_start, side_deadzone, target_forward_min, target_forward_max,
        target_side_max, target_region_std, missed_contact_grace_steps,
    )["correct_side_touch"].to(torch.float32)


def dribbling_s2_touch_force_soft_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    max_touch_force: float = 100.0,
    soft_touch_force_start: float = 60.0,
    side_deadzone: float = 0.04,
    target_forward_min: float = -0.06,
    target_forward_max: float = 0.14,
    target_side_max: float = 0.16,
    target_region_std: float = 0.12,
    missed_contact_grace_steps: int = 3,
) -> torch.Tensor:
    """Continuous correct-foot impact penalty above the soft force threshold."""
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        max_touch_force, soft_touch_force_start, side_deadzone, target_forward_min, target_forward_max,
        target_side_max, target_region_std, missed_contact_grace_steps,
    )["touch_force_soft_penalty"]


def dribbling_s2_undesired_ball_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    max_touch_force: float = 100.0,
    soft_touch_force_start: float = 60.0,
    side_deadzone: float = 0.04,
    target_forward_min: float = -0.06,
    target_forward_max: float = 0.14,
    target_side_max: float = 0.16,
    target_region_std: float = 0.12,
    missed_contact_grace_steps: int = 3,
) -> torch.Tensor:
    """Penalize new wrong-foot and non-ankle contacts without terminating."""
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        max_touch_force, soft_touch_force_start, side_deadzone, target_forward_min, target_forward_max,
        target_side_max, target_region_std, missed_contact_grace_steps,
    )["undesired_contact_penalty"]


def dribbling_s2_windowed_foot_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.3,
    foot_body_names: list[str] | None = None,
    contact_window_scale: float = 0.3,
) -> torch.Tensor:
    """Use one foot imitation term and soften it inside S2 contact windows."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    base = motion_relative_foot_position_error_exp(
        env, command_name, std, foot_body_names=foot_body_names
    )
    window = getattr(command, "s2_contact_window_ref", None)
    if not isinstance(window, torch.Tensor):
        return base
    scale = torch.where(
        window,
        torch.full_like(base, float(contact_window_scale)),
        torch.ones_like(base),
    )
    return base * scale


def dribbling_contact_telemetry(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    contact_surface: str = "any",
    medial_y_min: float = 0.018,
    cg_surface_gated: bool = False,
    contact_force_threshold: float = 14.0,
) -> dict[str, torch.Tensor]:
    """Return contact-region telemetry using the exact reward geometry.

    Values describe only a current sensor contact. ``ball_offset_foot_yaw``
    is therefore NaN and ``contact_body_idx`` is -1 when the ball has no robot
    contact. Foot ids use the CG convention: left=0, right=1, unknown=-1.
    """
    device = env.device
    num_envs = env.num_envs
    if all_body_cfg is None:
        has_contact = soccer_ball_contact_force_magnitude(env, ball_sensor_name) > 1.0
        force_mag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
        false = torch.zeros(num_envs, dtype=torch.bool, device=device)
        return {
            "has_contact": has_contact,
            "force_magnitude": force_mag,
            "contact_body_idx": torch.full((num_envs,), -1, dtype=torch.long, device=device),
            "contact_foot": torch.full((num_envs,), -1, dtype=torch.long, device=device),
            "legal_ankle": false,
            "generic_instep": false,
            "inside_instep": false,
            "outside_instep": false,
            "requested_surface_match": false,
            "gentle": force_mag <= contact_force_threshold,
            "legal_touch": false,
            "ball_offset_foot_yaw": torch.full((num_envs, 3), torch.nan, device=device),
            "medial_offset": torch.full((num_envs,), torch.nan, device=device),
        }

    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_body_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg
    )
    geometry = _dribbling_contact_surface_geometry(
        env,
        all_body_cfg,
        closest_body_idx,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
    )
    requested_surface_match = _dribbling_contact_surface_match(
        env,
        all_body_cfg,
        closest_body_idx,
        command_name=command_name,
        contact_surface=contact_surface,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
        cg_surface_gated=cg_surface_gated,
    )
    contact_foot = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    contact_foot[geometry["medial_sign"] < 0.0] = 0
    contact_foot[geometry["medial_sign"] > 0.0] = 1
    contact_body_idx = torch.where(
        has_contact,
        closest_body_idx,
        torch.full_like(closest_body_idx, -1),
    )
    offset = torch.where(
        has_contact.unsqueeze(-1),
        geometry["ball_offset_foot_yaw"],
        torch.full_like(geometry["ball_offset_foot_yaw"], torch.nan),
    )
    medial_offset = torch.where(
        has_contact,
        geometry["medial_offset"],
        torch.full_like(geometry["medial_offset"], torch.nan),
    )
    gentle = force_mag <= contact_force_threshold
    return {
        "has_contact": has_contact,
        "force_magnitude": force_mag,
        "contact_body_idx": contact_body_idx,
        "contact_foot": torch.where(has_contact, contact_foot, torch.full_like(contact_foot, -1)),
        "legal_ankle": has_contact & geometry["legal_ankle"],
        "generic_instep": has_contact & geometry["legal_ankle"] & geometry["known_foot"] & geometry["instep"],
        "inside_instep": has_contact & geometry["legal_ankle"] & geometry["known_foot"] & geometry["inside_instep"],
        "outside_instep": has_contact & geometry["legal_ankle"] & geometry["known_foot"] & geometry["outside_instep"],
        "requested_surface_match": has_contact & requested_surface_match,
        "gentle": has_contact & gentle,
        "legal_touch": has_contact & geometry["legal_ankle"] & requested_surface_match & gentle,
        "ball_offset_foot_yaw": offset,
        "medial_offset": medial_offset,
    }


# ---------------------------------------------------------------------------
# 2b) Dense approach — ankles near ball while force sensor shows no contact
# ---------------------------------------------------------------------------


def dribbling_approach_foot_ball_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    std: float = 0.22,
    pelvis_speed_min: float = 0.08,
) -> torch.Tensor:
    """``[0,1]`` shaping when the ball reports no contact: minimise ankle–ball gap."""
    if foot_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    no_contact = fmag <= contact_force_threshold

    robot = env.scene[foot_cfg.name]
    soccer_ball = env.scene["soccer_ball"]

    cache = getattr(env, "_dribbling_foot_ball_idx_cache", None)
    if cache is None or cache.get("names") != tuple(foot_cfg.body_names):
        idx = torch.as_tensor(
            robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=env.device,
        )
        cache = {"names": tuple(foot_cfg.body_names), "idx": idx}
        setattr(env, "_dribbling_foot_ball_idx_cache", cache)
    body_idx = cache["idx"]

    feet_pos = robot.data.body_pos_w[:, body_idx, :]
    ball_pos = soccer_ball.data.root_pos_w.unsqueeze(1)
    dist = torch.norm(feet_pos - ball_pos, dim=-1)
    min_dist = dist.min(dim=-1).values

    shaping = torch.exp(-(min_dist ** 2) / (std ** 2))
    out = torch.where(no_contact, shaping, torch.zeros_like(shaping))

    if pelvis_speed_min > 0.0:
        command: MotionCommand = env.command_manager.get_term(command_name)
        pelvis_sp = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
        out = out * torch.clamp(pelvis_sp / pelvis_speed_min, max=1.0)

    return out


# ---------------------------------------------------------------------------
# 2c) Pelvis orientation vs motion reference (reduces lean-back / arched cheat)
# ---------------------------------------------------------------------------


def dribbling_pelvis_quat_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.45,
) -> torch.Tensor:
    """Reward matching motion pelvis orientation (same frame as body tracking)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    names = command.cfg.body_names
    if "pelvis" not in names:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    pi = names.index("pelvis")
    ref = command.body_quat_relative_w[:, pi]
    rob = command.robot_body_quat_w[:, pi]
    err = quat_error_magnitude(ref, rob)
    return torch.exp(-(err ** 2) / (std ** 2))


# ---------------------------------------------------------------------------
# 2d) Penalise excessive horizontal ball speed (dribble vs kick)
# ---------------------------------------------------------------------------


def dribbling_ball_xy_speed_excess_penalty(
    env: ManagerBasedRLEnv,
    speed_cap: float = 3.5,
    linear_scale: float = 1.5,
) -> torch.Tensor:
    """Penalty in ``[0, 1]`` for ``|v_ball,xy|`` above ``speed_cap``."""
    soccer_ball = env.scene["soccer_ball"]
    sp = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    excess = torch.relu(sp - speed_cap)
    return torch.clamp(excess / linear_scale, max=1.0)


def dribbling_ball_coast_without_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    speed_threshold: float = 0.35,
    speed_scale: float = 0.45,
    max_close_xy_dist: float = 0.55,
) -> torch.Tensor:
    """Penalty when the ball slides fast **near** the robot with no contact.

    Only applies when the ball is within ``max_close_xy_dist`` of the pelvis so a
    kick-and-chase rollout (ball rolling ahead out of reach) is not penalised.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    ball_sp = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    no_touch = fmag <= contact_force_threshold

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]
    dist_xy = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)
    close = dist_xy <= max_close_xy_dist

    excess = torch.relu(ball_sp - speed_threshold)
    penalty = torch.clamp(excess / max(speed_scale, 1e-6), max=1.0)
    return penalty * no_touch.to(torch.float32) * close.to(torch.float32)


# ---------------------------------------------------------------------------
# 2e) Ball forward progress reward — encourage actual dribble advancement
# ---------------------------------------------------------------------------


def dribbling_ball_forward_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_speed: float = 0.2,
    speed_scale: float = 0.25,
    pelvis_speed_min: float = 0.06,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 0.5,
    require_recent_contact: bool = True,
    recent_contact_window: int = 10,
    cg_gated_contact: bool = False,
) -> torch.Tensor:
    """Reward forward ball velocity along task +X (world-parallel).

    This term rewards positive +X ball speed, and can optionally be gated by
    recent contact to avoid standing
    exploits where the policy does not engage the ball.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
    forward_speed = task_forward_speed(ball_vel_w)
    ball_dominance = task_velocity_forward_dominance(ball_vel_w)

    base = torch.clamp((forward_speed - min_forward_speed) / max(speed_scale, 1e-6), min=0.0, max=1.0)
    base = base * forward_dominance_gate(ball_dominance, 0.4)

    pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
    gate = torch.clamp(pelvis_speed / max(pelvis_speed_min, 1e-6), max=1.0)

    if require_recent_contact:
        gate = gate * _dribbling_recent_contact_gate(
            env,
            ball_sensor_name,
            contact_force_threshold,
            recent_contact_window,
            command=command,
            cg_gated=cg_gated_contact,
        )

    return base * gate


def dribbling_command_ball_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_speed: float = 0.2,
    command_speed_ratio: float = 0.50,
    speed_scale: float = 0.25,
    lateral_ratio_max: float = 0.70,
    pelvis_speed_min: float = 0.06,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 0.5,
    require_recent_contact: bool = True,
    recent_contact_window: int = 10,
    cg_gated_contact: bool = False,
) -> torch.Tensor:
    """Reward recent-contact ball progress along the active command direction.

    ``command_speed_ratio`` makes high-speed commands demand proportionally
    faster ball progress without asking the ball to exactly match pelvis speed.
    A lateral gate prevents a sideways kick from scoring as forward dribbling.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, command_speed = _command_direction_xy(command)
    forward_speed, lateral_speed = _command_frame_components(soccer_ball.data.root_lin_vel_w[:, :2], direction_xy)

    required_speed = torch.maximum(
        torch.full_like(command_speed, float(min_forward_speed)),
        command_speed * float(command_speed_ratio),
    )
    progress = torch.clamp(
        (forward_speed - required_speed) / max(speed_scale, 1.0e-6), min=0.0, max=1.0
    )
    lateral_ratio = torch.abs(lateral_speed) / forward_speed.clamp(min=1.0e-4)
    direction_gate = torch.clamp(
        1.0 - lateral_ratio / max(lateral_ratio_max, 1.0e-6), min=0.0, max=1.0
    )

    pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
    gate = torch.clamp(pelvis_speed / max(pelvis_speed_min, 1.0e-6), max=1.0)
    if require_recent_contact:
        gate = gate * _dribbling_recent_contact_gate(
            env,
            ball_sensor_name,
            contact_force_threshold,
            recent_contact_window,
            command=command,
            cg_gated=cg_gated_contact,
        )
    return progress * direction_gate * gate


# ---------------------------------------------------------------------------
# 2f) Contact-graph style phase alignment (approach -> control/push)
# ---------------------------------------------------------------------------


def dribbling_phase_graph_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 0.5,
    approach_xy_dist: float = 0.55,
    approach_dist_std: float = 0.20,
    push_speed_threshold: float = 0.22,
) -> torch.Tensor:
    """Phase-style shaping without explicit per-frame labels.

    - ``approach`` phase (ball far): reward getting closer while avoiding contact.
    - ``control/push`` phase (ball near): reward contact, and reward stronger when
      ball moves forward along task +X (push over static trapping).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]
    dist_xy = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)

    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    has_contact = fmag > contact_force_threshold

    ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
    forward_speed = task_forward_speed(ball_vel_w)

    phase_approach = dist_xy > approach_xy_dist
    phase_interact = ~phase_approach

    # Approach: closer is better, but do not touch too early.
    approach_core = torch.exp(-((dist_xy - approach_xy_dist).clamp(min=0.0) ** 2) / (approach_dist_std ** 2))
    approach_reward = approach_core * (~has_contact).to(torch.float32)

    # Interact: contact is required; encourage push speed on top of stable contact.
    push_gain = torch.clamp(forward_speed / max(push_speed_threshold, 1e-6), min=0.0, max=1.0)
    interact_reward = has_contact.to(torch.float32) * (0.55 + 0.45 * push_gain)

    return torch.where(phase_approach, approach_reward, interact_reward)


# ---------------------------------------------------------------------------
# 2g) Anti-orbit penalty — discourage circling around the ball without touch
# ---------------------------------------------------------------------------


def dribbling_orbiting_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    orbit_radius_max: float = 0.9,
    tangential_deadzone: float = 0.08,
    tangential_scale: float = 0.35,
) -> torch.Tensor:
    """Penalty in ``[0,1]`` for tangential pelvis motion around the ball.

    The penalty is active only when the pelvis is near the ball (within
    ``orbit_radius_max`` in XY) and ball contact force is weak
    (``<= contact_force_threshold``). This directly suppresses the common
    local optimum of \"one foot hovering, circling around the ball\".
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]

    r = pelvis_pos_xy - ball_pos_xy
    r_norm = torch.norm(r, dim=-1)
    r_hat = r / (r_norm.unsqueeze(-1) + 1e-6)

    # Tangent unit vector around the ball (CCW): [-y, x]
    t_hat = torch.stack((-r_hat[:, 1], r_hat[:, 0]), dim=-1)
    v_tan = torch.abs(torch.sum(pelvis_vel_xy * t_hat, dim=-1))

    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    weak_contact = fmag <= contact_force_threshold
    near_ball = r_norm <= orbit_radius_max

    core = torch.clamp((v_tan - tangential_deadzone) / tangential_scale, min=0.0, max=1.0)
    return core * (weak_contact & near_ball).to(torch.float32)


# ---------------------------------------------------------------------------
# 3a) Legal Foot Gentle Touch — small positive reward
# ---------------------------------------------------------------------------

def dribbling_legal_foot_touch(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    contact_surface: str = "any",
    medial_y_min: float = 0.018,
    cg_gated: bool = False,
    cg_surface_gated: bool = False,
    min_pelvis_heading: float = 0.0,
) -> torch.Tensor:
    """Reward a new gentle touch in the selected foot/instep side.

    ``contact_surface`` is ``any`` (legacy), ``instep`` (either side),
    ``inside_instep``, or ``outside_instep``. The instep variants use the
    contacted foot's yaw frame, so selecting one is stronger than merely
    selecting the left or right foot.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    surface_match = _dribbling_contact_surface_match(
        env,
        all_body_cfg,
        closest_idx,
        command_name=command_name,
        contact_surface=contact_surface,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
        cg_surface_gated=cg_surface_gated,
    )
    gentle = force_mag <= force_threshold
    touch = has_contact & is_ankle & surface_match & gentle
    if cg_gated:
        touch = _dribbling_cg_gated_sim_contact(command, touch)

    prev_name = "_dribbling_prev_legal_foot_touch"
    prev = getattr(env, prev_name, None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    new_touch = touch & ~prev
    setattr(env, prev_name, touch.detach().clone())

    reward = new_touch.to(torch.float32)
    if min_pelvis_heading > 0.0:
        heading = task_pelvis_heading_cos_world_x(command.robot_pelvis_quat_w).clamp(min=0.0, max=1.0)
        reward = reward * forward_dominance_gate(heading, min_pelvis_heading)
    return reward


# ---------------------------------------------------------------------------
# 3b) Micro-Contact Filter — moderate EMA penalty for legal foot hard kicks
# ---------------------------------------------------------------------------

def dribbling_micro_contact_filter(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    max_penalty: float = 2.0,
    ema_alpha: float = 0.4,
    all_body_cfg: SceneEntityCfg | None = None,
    foot_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
) -> torch.Tensor:
    """EMA-smoothed penalty when an **ankle** hits the ball too hard."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)

    # Only consider ankle hard contacts for this penalty
    legal_hard_force = torch.where(
        has_contact & is_ankle,
        force_mag,
        torch.zeros_like(force_mag),
    )

    # ── 5-frame EMA low-pass filter ──────────────────────────────────
    buf_name = "_dribbling_contact_ema"
    ema = getattr(env, buf_name, None)
    if ema is None or ema.shape[0] != env.num_envs:
        ema = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ema = ema_alpha * legal_hard_force + (1.0 - ema_alpha) * ema

    # Reset EMA for environments that just reset
    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is not None:
        reset_mask = step_buf == 0
        if torch.any(reset_mask):
            ema[reset_mask] = 0.0

    setattr(env, buf_name, ema)

    # ── Clipped penalty ──────────────────────────────────────────────
    excess = torch.clamp(ema - force_threshold, min=0.0)
    penalty = (excess / force_threshold).clamp(max=max_penalty)

    return penalty


# ---------------------------------------------------------------------------
# 3c) Undesired Contact Penalty — wrong body or instep side
# ---------------------------------------------------------------------------

def dribbling_undesired_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    contact_surface: str = "any",
    medial_y_min: float = 0.018,
    cg_surface_gated: bool = False,
    wrong_surface_penalty: float = 0.25,
) -> torch.Tensor:
    """Penalize non-ankle contact fully and wrong instep side more gently.

    The term is multiplied by its configured negative reward weight.  A
    non-ankle contact returns ``1.0``; an ankle contact on the wrong
    inside/outside instep returns ``wrong_surface_penalty``.  The latter is
    intentionally smaller: timing is already supervised independently by
    ``dribbling_cg_premature_contact_penalty``.
    """
    if not 0.0 <= wrong_surface_penalty <= 1.0:
        raise ValueError("wrong_surface_penalty must be in [0, 1].")
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    surface_match = _dribbling_contact_surface_match(
        env,
        all_body_cfg,
        closest_idx,
        command_name=command_name,
        contact_surface=contact_surface,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
        cg_surface_gated=cg_surface_gated,
    )

    non_ankle = has_contact & ~is_ankle
    wrong_surface = has_contact & is_ankle & ~surface_match
    return non_ankle.to(torch.float32) + wrong_surface.to(torch.float32) * wrong_surface_penalty


# ---------------------------------------------------------------------------
# 4) Annotated contact-graph (dribbling) — demo ball + label consistency
# ---------------------------------------------------------------------------


def dribbling_support_ankle_roll_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.24,
    deadzone: float = 0.10,
    error_cap: float = 0.35,
    left_ankle_roll_joint: str = "left_ankle_roll_joint",
    right_ankle_roll_joint: str = "right_ankle_roll_joint",
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Lightly keep the non-touching ankle roll near its style reference.

    On a labeled right-foot touch, only the left (support) ankle is scored;
    left-foot touches mirror the selection.  The touching ankle remains fully
    free.  A deadzone permits normal balance corrections, while the capped
    error keeps this cosmetic support-foot term from competing with ball
    control when a recovery needs a larger deviation.
    """
    if std <= 0.0 or deadzone < 0.0 or error_cap <= 0.0:
        raise ValueError("std and error_cap must be positive; deadzone must be non-negative.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = getattr(command, "motion_has_dribble_cg_label", None)
    if not isinstance(labeled, torch.Tensor) or not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref_contact = command.dribble_cg_contact_ref
    ref_touch_foot = command.dribble_cg_foot_ref
    active = labeled & ref_contact & (ref_touch_foot >= 0) & locomotion_task_state_mask(
        command, active_task_states
    )
    if not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    robot = env.scene[command.cfg.asset_name]
    joint_cache_name = "_dribbling_support_ankle_roll_joint_ids"
    joint_ids = getattr(command, joint_cache_name, None)
    if joint_ids is None:
        ids, found_names = robot.find_joints(
            [left_ankle_roll_joint, right_ankle_roll_joint], preserve_order=True
        )
        if len(ids) != 2:
            raise ValueError(
                "Could not resolve support ankle roll joints: "
                f"expected {[left_ankle_roll_joint, right_ankle_roll_joint]}, found {found_names}."
            )
        joint_ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
        setattr(command, joint_cache_name, joint_ids)

    # Ref foot id is 0=left / 1=right, so support is the opposite joint.
    support_joint_ids = torch.where(ref_touch_foot == 1, joint_ids[0], joint_ids[1])
    batch_ids = torch.arange(env.num_envs, device=env.device)
    actual = robot.data.joint_pos[batch_ids, support_joint_ids]
    reference = command.joint_pos[batch_ids, support_joint_ids]
    excess = torch.clamp(torch.abs(actual - reference) - deadzone, min=0.0)
    bounded_excess = torch.clamp(excess, max=error_cap)
    reward = torch.exp(-torch.square(bounded_excess) / (std**2))
    return reward * active.to(reward.dtype)


def dribbling_cg_contact_consistency(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """1.0 when sim contact presence matches the annotated CG contact bit."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = _dribbling_effective_cg_label_mask(command, command.motion_has_dribble_cg_label)
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref = command.dribble_cg_contact_ref
    sim_c = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    agree = (ref == sim_c).to(torch.float32)
    active = labeled & locomotion_task_state_mask(command, active_task_states)
    return agree * active.to(torch.float32)


def dribbling_cg_premature_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize sensor contact on CG non-contact (approach) frames."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = _dribbling_effective_cg_label_mask(command, command.motion_has_dribble_cg_label)
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref = command.dribble_cg_contact_ref
    sim_c = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    return (labeled & sim_c & ~ref).to(torch.float32)


def dribbling_ball_trapped_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_x: float = 0.18,
    max_ball_height: float = 0.20,
) -> torch.Tensor:
    """Penalize ball under the body, behind the pelvis, or popped up (夹球 / 蹦)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    ball_pos_w = soccer_ball.data.root_pos_w
    pelvis_pos_w = command.robot_pelvis_pos_w
    x_task = task_forward_offset(ball_pos_w, pelvis_pos_w)
    too_close = x_task < min_forward_x
    behind = x_task < 0.0
    popped = ball_pos_w[:, 2] > max_ball_height
    return (too_close | behind | popped).to(torch.float32)


def dribbling_command_ball_trapped_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_x: float = 0.18,
    max_ball_height: float = 0.20,
) -> torch.Tensor:
    """Penalize a trapped ball using the active command axis, not world ``+X``."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    forward_offset, _ = _command_frame_components(offset_xy, direction_xy)
    too_close = forward_offset < min_forward_x
    behind = forward_offset < 0.0
    popped = soccer_ball.data.root_pos_w[:, 2] > max_ball_height
    return (too_close | behind | popped).to(torch.float32)


def dribbling_pelvis_local_ball_trapped_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_x: float = 0.18,
    max_ball_height: float = 0.20,
) -> torch.Tensor:
    """Penalize a ball under/behind the current pelvis in pelvis-local axes."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    offset_b = _pelvis_yaw_local_vector(
        command, soccer_ball.data.root_pos_w[:, :3] - command.robot_pelvis_pos_w
    )
    too_close = offset_b[:, 0] < min_forward_x
    behind = offset_b[:, 0] < 0.0
    popped = soccer_ball.data.root_pos_w[:, 2] > max_ball_height
    return (too_close | behind | popped).to(torch.float32)


def dribbling_sustained_contact_penalty(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    max_contact_steps: int = 5,
) -> torch.Tensor:
    """Penalize keeping the ball squeezed between feet for too many consecutive steps."""
    in_contact = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    buf_name = "_dribbling_consecutive_contact_steps"
    cnt = getattr(env, buf_name, None)
    if cnt is None or cnt.shape[0] != env.num_envs:
        cnt = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    step_buf = getattr(env, "episode_length_buf", None)
    reset_mask = step_buf == 0 if step_buf is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    cnt = torch.where(reset_mask, torch.zeros_like(cnt), torch.where(in_contact, cnt + 1, torch.zeros_like(cnt)))
    setattr(env, buf_name, cnt)
    return (cnt > int(max_contact_steps)).to(torch.float32)


def dribbling_ball_bounce_penalty(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    vz_threshold: float = 0.32,
) -> torch.Tensor:
    """Penalize large vertical ball speed while in contact (pinch-bounce)."""
    soccer_ball = env.scene["soccer_ball"]
    in_contact = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    vz = torch.abs(soccer_ball.data.root_lin_vel_w[:, 2])
    return (in_contact & (vz > vz_threshold)).to(torch.float32)


def dribbling_cg_foot_consistency(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    left_ankle_body_name: str = "left_ankle_roll_link",
    right_ankle_body_name: str = "right_ankle_roll_link",
    contact_surface: str = "any",
    medial_y_min: float = 0.018,
    cg_surface_gated: bool = False,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Reward a labeled-foot contact, optionally matching its CG instep side."""
    if all_body_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = _dribbling_effective_cg_label_mask(command, command.motion_has_dribble_cg_label)
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref_c = command.dribble_cg_contact_ref
    ref_f = command.dribble_cg_foot_ref
    active = labeled & ref_c & (ref_f >= 0) & locomotion_task_state_mask(command, active_task_states)

    has_contact, _fm, closest = _identify_contact_body(env, command, ball_sensor_name, all_body_cfg)
    names = list(all_body_cfg.body_names)
    try:
        li = names.index(left_ankle_body_name)
        ri = names.index(right_ankle_body_name)
    except ValueError:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    expected = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    expected[ref_f == 0] = li
    expected[ref_f == 1] = ri

    surface_match = _dribbling_contact_surface_match(
        env,
        all_body_cfg,
        closest,
        command_name=command_name,
        contact_surface=contact_surface,
        num_ankle_links=2,
        medial_y_min=medial_y_min,
        cg_surface_gated=cg_surface_gated,
    )
    match = (closest == expected) & surface_match & has_contact & active
    return match.to(torch.float32)


def _dribbling_cg_flow_metrics(
    env: ManagerBasedRLEnv,
    *,
    command_name: str,
    ball_sensor_name: str,
    all_body_cfg: SceneEntityCfg | None,
    num_ankle_links: int,
    contact_surface: str,
    medial_y_min: float,
    contact_force_threshold: float,
    max_touch_force: float,
    release_window_steps: int,
    speed_lower_ratio: float,
    speed_upper_ratio: float,
    lateral_speed_std: float,
    overspeed_std: float,
    lateral_corridor_std: float,
    max_progress_rate: float,
    active_task_states: tuple[int, ...] | None,
) -> dict[str, torch.Tensor]:
    """Update and return the shared causal contact-flow reward state once per step."""
    if release_window_steps <= 0:
        raise ValueError("release_window_steps must be positive.")
    if not 0.0 < speed_lower_ratio <= speed_upper_ratio:
        raise ValueError("Expected 0 < speed_lower_ratio <= speed_upper_ratio.")

    step_counter = int(getattr(env, "common_step_counter", -1))
    episode_step = getattr(env, "episode_length_buf", None)
    if episode_step is None:
        episode_step = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    signature = (
        command_name,
        ball_sensor_name,
        tuple(all_body_cfg.body_names) if all_body_cfg is not None else (),
        num_ankle_links,
        contact_surface,
        medial_y_min,
        contact_force_threshold,
        max_touch_force,
        release_window_steps,
        speed_lower_ratio,
        speed_upper_ratio,
        lateral_speed_std,
        overspeed_std,
        lateral_corridor_std,
        max_progress_rate,
        active_task_states,
    )
    cache = getattr(env, "_dribbling_cg_flow_cache", None)
    if isinstance(cache, dict) and cache.get("signature") == signature:
        same_step = step_counter >= 0 and cache.get("step_counter") == step_counter
        if step_counter < 0:
            cached_episode_step = cache.get("episode_step")
            same_step = isinstance(cached_episode_step, torch.Tensor) and torch.equal(
                cached_episode_step, episode_step
            )
        if same_step:
            return cache["metrics"]

    device = env.device
    zeros = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
    false = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
    minus_one = torch.full((env.num_envs,), -1, device=device, dtype=torch.long)
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    has_flow = getattr(command, "motion_has_dribble_cg_flow_label", false)
    if not isinstance(has_flow, torch.Tensor):
        has_flow = false
    label_valid = getattr(command, "dribble_cg_flow_valid_ref", false)
    if not isinstance(label_valid, torch.Tensor):
        label_valid = false
    task_active = locomotion_task_state_mask(command, active_task_states)
    flow_state_enabled = _dribbling_effective_cg_label_mask(command, has_flow) & task_active
    label_valid = _dribbling_effective_cg_label_mask(command, has_flow & label_valid) & task_active
    flow_dir_local = getattr(command, "dribble_cg_flow_dir_local_ref", None)
    flow_distance = getattr(command, "dribble_cg_flow_distance_ref", None)
    flow_duration = getattr(command, "dribble_cg_flow_duration_ref", None)
    flow_anchor = getattr(command, "dribble_cg_flow_anchor_frame_ref", None)
    if not isinstance(flow_dir_local, torch.Tensor):
        flow_dir_local = torch.zeros(env.num_envs, 2, device=device)
    if not isinstance(flow_distance, torch.Tensor):
        flow_distance = torch.full_like(zeros, -1.0)
    if not isinstance(flow_duration, torch.Tensor):
        flow_duration = torch.full_like(zeros, -1.0)
    if not isinstance(flow_anchor, torch.Tensor):
        flow_anchor = minus_one

    state = getattr(env, "_dribbling_cg_flow_state", None)
    if not isinstance(state, dict) or state.get("num_envs") != env.num_envs:
        state = {
            "num_envs": env.num_envs,
            "active": false.clone(),
            "anchor_frame": minus_one.clone(),
            "target_direction_world": torch.zeros(env.num_envs, 2, device=device),
            "touch_ball_position": torch.zeros(env.num_envs, 2, device=device),
            "target_distance": zeros.clone(),
            "target_duration": zeros.clone(),
            "age_steps": torch.zeros(env.num_envs, device=device, dtype=torch.long),
            "best_progress": zeros.clone(),
            "previous_contact": false.clone(),
        }

    reset = episode_step == 0
    state["active"] = state["active"] & ~reset
    state["anchor_frame"] = torch.where(reset, minus_one, state["anchor_frame"])
    state["age_steps"] = torch.where(reset, torch.zeros_like(state["age_steps"]), state["age_steps"])
    state["best_progress"] = torch.where(reset, zeros, state["best_progress"])
    state["previous_contact"] = state["previous_contact"] & ~reset

    if all_body_cfg is None:
        has_contact = false
        force_mag = zeros
        closest = torch.zeros(env.num_envs, device=device, dtype=torch.long)
        surface_match = false
        actual_foot = minus_one
    else:
        has_contact, force_mag, closest = _identify_contact_body(
            env, command, ball_sensor_name, all_body_cfg
        )
        has_contact = has_contact & (force_mag >= contact_force_threshold)
        surface_match = _dribbling_contact_surface_match(
            env,
            all_body_cfg,
            closest,
            command_name=command_name,
            contact_surface=contact_surface,
            num_ankle_links=num_ankle_links,
            medial_y_min=medial_y_min,
            cg_surface_gated=True,
        )
        actual_foot = minus_one.clone()
        for body_idx, body_name in enumerate(all_body_cfg.body_names):
            if body_idx >= num_ankle_links:
                break
            if "left_ankle" in body_name:
                actual_foot[closest == body_idx] = 0
            elif "right_ankle" in body_name:
                actual_foot[closest == body_idx] = 1

    new_contact = has_contact & ~state["previous_contact"]
    state["previous_contact"] = has_contact.detach().clone()
    ref_contact = getattr(command, "dribble_cg_contact_ref", false)
    ref_foot = getattr(command, "dribble_cg_foot_ref", minus_one)
    direction_norm = torch.norm(flow_dir_local, dim=-1)
    label_ready = (
        label_valid
        & ref_contact
        & (flow_anchor >= 0)
        & (flow_distance > 1.0e-4)
        & (flow_duration > 1.0e-4)
        & (direction_norm > 1.0e-4)
    )
    correct_arrival_touch = (
        new_contact
        & (actual_foot == ref_foot)
        & surface_match
        & (force_mag <= max_touch_force)
        & ref_contact
    )
    correct_new_touch = (
        correct_arrival_touch
        & label_ready
        & (flow_anchor != state["anchor_frame"])
    )

    direction_local_unit = flow_dir_local / direction_norm.unsqueeze(-1).clamp(min=1.0e-6)
    direction_local_3d = torch.cat(
        (direction_local_unit, torch.zeros(env.num_envs, 1, device=device)), dim=-1
    )
    direction_world = quat_apply(yaw_quat(command.robot_pelvis_quat_w), direction_local_3d)[:, :2]
    direction_world = direction_world / torch.norm(direction_world, dim=-1, keepdim=True).clamp(min=1.0e-6)
    ball_position = soccer_ball.data.root_pos_w[:, :2]

    latch_2d = correct_new_touch.unsqueeze(-1)
    # A correct touch is the arrival condition for the previous segment. A
    # non-final anchor immediately latches the next outgoing segment; the final
    # anchor only clears the previous one because it has no flow label.
    state["active"] = torch.where(correct_arrival_touch, torch.zeros_like(state["active"]), state["active"])
    state["target_direction_world"] = torch.where(
        latch_2d, direction_world, state["target_direction_world"]
    )
    state["touch_ball_position"] = torch.where(
        latch_2d, ball_position, state["touch_ball_position"]
    )
    state["target_distance"] = torch.where(correct_new_touch, flow_distance, state["target_distance"])
    state["target_duration"] = torch.where(correct_new_touch, flow_duration, state["target_duration"])
    state["anchor_frame"] = torch.where(correct_new_touch, flow_anchor, state["anchor_frame"])
    state["age_steps"] = torch.where(correct_new_touch, torch.zeros_like(state["age_steps"]), state["age_steps"])
    state["best_progress"] = torch.where(correct_new_touch, zeros, state["best_progress"])
    state["active"] = torch.where(correct_new_touch, torch.ones_like(state["active"]), state["active"])

    segment_active = state["active"] & flow_state_enabled
    state["active"] = segment_active
    target_direction = state["target_direction_world"]
    lateral_axis = torch.stack((-target_direction[:, 1], target_direction[:, 0]), dim=-1)
    ball_velocity = soccer_ball.data.root_lin_vel_w[:, :2]
    parallel_speed = torch.sum(ball_velocity * target_direction, dim=-1)
    lateral_speed = torch.abs(torch.sum(ball_velocity * lateral_axis, dim=-1))
    nominal_speed = state["target_distance"] / state["target_duration"].clamp(min=1.0e-4)

    lower_speed = speed_lower_ratio * nominal_speed
    upper_speed = speed_upper_ratio * nominal_speed
    underspeed_score = torch.clamp(parallel_speed / lower_speed.clamp(min=1.0e-4), min=0.0, max=1.0)
    overspeed_score = torch.where(
        parallel_speed <= upper_speed,
        torch.ones_like(parallel_speed),
        torch.exp(-torch.square((parallel_speed - upper_speed) / max(overspeed_std, 1.0e-6))),
    )
    direction_score = torch.exp(-torch.square(lateral_speed / max(lateral_speed_std, 1.0e-6)))
    release_active = segment_active & (state["age_steps"] < int(release_window_steps))
    release_reward = underspeed_score * overspeed_score * direction_score * release_active.to(torch.float32)

    displacement = ball_position - state["touch_ball_position"]
    longitudinal = torch.sum(displacement * target_direction, dim=-1)
    lateral_offset = torch.abs(torch.sum(displacement * lateral_axis, dim=-1))
    raw_progress = torch.clamp(
        longitudinal / state["target_distance"].clamp(min=1.0e-4), min=0.0, max=1.0
    )
    next_best_progress = torch.where(
        segment_active, torch.maximum(state["best_progress"], raw_progress), state["best_progress"]
    )
    progress_delta = torch.clamp(next_best_progress - state["best_progress"], min=0.0)
    state["best_progress"] = next_best_progress
    step_dt = max(float(getattr(env, "step_dt", 0.02)), 1.0e-6)
    progress_rate = torch.clamp(progress_delta / step_dt, max=max_progress_rate)
    corridor_score = torch.exp(-torch.square(lateral_offset / max(lateral_corridor_std, 1.0e-6)))
    progress_reward = progress_rate * corridor_score * segment_active.to(torch.float32)
    state["age_steps"] = torch.where(
        segment_active, state["age_steps"] + 1, state["age_steps"]
    )
    setattr(env, "_dribbling_cg_flow_state", state)

    metrics = {
        "label_available": has_flow.to(torch.bool),
        "label_valid": label_valid,
        "label_anchor_frame": flow_anchor,
        "label_direction_local": flow_dir_local,
        "label_distance": flow_distance,
        "label_duration": flow_duration,
        "latched": state["active"],
        "release_active": release_active,
        "target_direction_world": target_direction,
        "nominal_speed": nominal_speed,
        "parallel_speed": parallel_speed,
        "lateral_speed": lateral_speed,
        "release_reward": release_reward,
        "progress": state["best_progress"],
        "progress_rate": progress_rate,
        "lateral_offset": lateral_offset,
        "progress_reward": progress_reward,
    }
    setattr(env, "_dribbling_cg_flow_telemetry", metrics)
    setattr(
        env,
        "_dribbling_cg_flow_cache",
        {
            "signature": signature,
            "step_counter": step_counter,
            "episode_step": episode_step.detach().clone(),
            "metrics": metrics,
        },
    )
    return metrics


def dribbling_cg_flow_release_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    contact_surface: str = "instep",
    medial_y_min: float = 0.018,
    contact_force_threshold: float = 1.0,
    max_touch_force: float = 20.0,
    release_window_steps: int = 8,
    speed_lower_ratio: float = 0.7,
    speed_upper_ratio: float = 1.6,
    lateral_speed_std: float = 0.35,
    overspeed_std: float = 0.5,
    lateral_corridor_std: float = 0.18,
    max_progress_rate: float = 6.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Reward outgoing ball direction and a broad speed band after a correct touch."""
    return _dribbling_cg_flow_metrics(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        all_body_cfg=all_body_cfg,
        num_ankle_links=num_ankle_links,
        contact_surface=contact_surface,
        medial_y_min=medial_y_min,
        contact_force_threshold=contact_force_threshold,
        max_touch_force=max_touch_force,
        release_window_steps=release_window_steps,
        speed_lower_ratio=speed_lower_ratio,
        speed_upper_ratio=speed_upper_ratio,
        lateral_speed_std=lateral_speed_std,
        overspeed_std=overspeed_std,
        lateral_corridor_std=lateral_corridor_std,
        max_progress_rate=max_progress_rate,
        active_task_states=active_task_states,
    )["release_reward"]


def dribbling_cg_flow_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    contact_surface: str = "instep",
    medial_y_min: float = 0.018,
    contact_force_threshold: float = 1.0,
    max_touch_force: float = 20.0,
    release_window_steps: int = 8,
    speed_lower_ratio: float = 0.7,
    speed_upper_ratio: float = 1.6,
    lateral_speed_std: float = 0.35,
    overspeed_std: float = 0.5,
    lateral_corridor_std: float = 0.18,
    max_progress_rate: float = 6.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Reward one-way progress toward the next contact without tracking a full path."""
    return _dribbling_cg_flow_metrics(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        all_body_cfg=all_body_cfg,
        num_ankle_links=num_ankle_links,
        contact_surface=contact_surface,
        medial_y_min=medial_y_min,
        contact_force_threshold=contact_force_threshold,
        max_touch_force=max_touch_force,
        release_window_steps=release_window_steps,
        speed_lower_ratio=speed_lower_ratio,
        speed_upper_ratio=speed_upper_ratio,
        lateral_speed_std=lateral_speed_std,
        overspeed_std=overspeed_std,
        lateral_corridor_std=lateral_corridor_std,
        max_progress_rate=max_progress_rate,
        active_task_states=active_task_states,
    )["progress_reward"]


def dribbling_cg_foot_ball_distance_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.12,
    use_xy_only: bool = False,
    left_ankle_body_name: str = "left_ankle_roll_link",
    right_ankle_body_name: str = "right_ankle_roll_link",
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Match sim foot–ball distance to demo distance from synthesized CG trajectory.

    Requires ``dribble_cg_foot_ball_dist`` in motion ``.npz`` (see
    ``scripts/dribble/synthesize_dribble_ball_traj.py``). Active on every frame
    with a stitched demo ball trajectory (contact **and** non-contact approach
    gaps), not only during annotated contact segments.

    - ``ref_dist`` = demo XY distance from the reference foot to synthesized ball.
    - Reference foot comes from ``dribble_cg_dist_foot`` when present, else
      falls back to ``dribble_cg_foot`` (legacy contact-only labels).
    - ``sim_dist`` = distance from that foot to the **sim** ball.
    - reward = ``exp(-(sim_dist - ref_dist)^2 / std^2)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = _dribbling_effective_cg_label_mask(command, command.motion_has_dribble_cg_foot_ball_dist_label)
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref_dist = command.dribble_cg_foot_ball_dist_ref
    ref_foot_dist = command.dribble_cg_dist_foot_ref
    ref_foot = torch.where(
        ref_foot_dist >= 0,
        ref_foot_dist,
        command.dribble_cg_foot_ref,
    )
    active = labeled & (ref_dist >= 0.0) & (ref_foot >= 0) & locomotion_task_state_mask(
        command, active_task_states
    )

    robot = env.scene[command.cfg.asset_name]
    soccer_ball = env.scene["soccer_ball"]
    ball_pos = soccer_ball.data.root_pos_w[:, :3]

    li = robot.body_names.index(left_ankle_body_name)
    ri = robot.body_names.index(right_ankle_body_name)
    left_pos = robot.data.body_pos_w[:, li]
    right_pos = robot.data.body_pos_w[:, ri]

    foot_pos = torch.where(
        (ref_foot == 0).unsqueeze(-1),
        left_pos,
        torch.where((ref_foot == 1).unsqueeze(-1), right_pos, left_pos),
    )

    delta = foot_pos - ball_pos
    if use_xy_only:
        sim_dist = torch.norm(delta[:, :2], dim=-1)
    else:
        sim_dist = torch.norm(delta, dim=-1)

    err = (sim_dist - ref_dist) ** 2
    rew = torch.exp(-err / max(std, 1e-6) ** 2)
    return rew * active.to(torch.float32)


def dribbling_gait_foot_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.3,
    foot_body_names: list[str] | None = None,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Foot imitation reward active only while the ball is **not** in contact.

    Between touches the policy should follow the reference gait (alternating
  steps) instead of freezing in a kick-ready stance with the dribble foot forward.
    """
    base = motion_relative_foot_position_error_exp(
        env, command_name, std, foot_body_names=foot_body_names
    )
    no_ball = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    command: MotionCommand = env.command_manager.get_term(command_name)
    active = no_ball & locomotion_task_state_mask(command, active_task_states)
    return base * active.to(torch.float32)


def dribbling_chase_ball_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    min_ball_ahead: float = 0.25,
    max_chase_xy_dist: float = 1.1,
    pelvis_forward_speed_min: float = 0.22,
    forward_speed_scale: float = 0.45,
) -> torch.Tensor:
    """Reward running forward to catch a ball that rolled ahead after a touch.

    Active when there is no ball contact, the ball lies on task +X ahead of the
    pelvis by at least ``min_ball_ahead``, and the robot is within chase range.
    This shapes kick → run → kick rather than shuffling beside the ball.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    no_ball = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    pelvis_pos_w = command.robot_pelvis_pos_w
    ball_pos_w = soccer_ball.data.root_pos_w

    x_ahead = task_forward_offset(ball_pos_w, pelvis_pos_w)
    dist_xy = torch.norm(ball_pos_w[:, :2] - pelvis_pos_w[:, :2], dim=-1)

    ball_ahead = x_ahead >= min_ball_ahead
    in_range = dist_xy <= max_chase_xy_dist

    pelvis_vel_w = command.robot_anchor_lin_vel_w[:, :3]
    forward = task_forward_speed(pelvis_vel_w)
    dominance = task_velocity_forward_dominance(pelvis_vel_w)
    speed_rew = torch.clamp(
        (forward - pelvis_forward_speed_min) / max(forward_speed_scale, 1e-6),
        min=0.0,
        max=1.0,
    )
    speed_rew = speed_rew * forward_dominance_gate(dominance, 0.45)

    active = no_ball & ball_ahead & in_range
    return speed_rew * active.to(torch.float32)


def dribbling_command_chase_ball_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    min_ball_ahead: float = 0.25,
    max_chase_xy_dist: float = 1.1,
    pelvis_forward_speed_min: float = 0.22,
    forward_speed_scale: float = 0.45,
) -> torch.Tensor:
    """Reward closing on a nearby ball ahead along the active command direction."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    ball_ahead, _ = _command_frame_components(offset_xy, direction_xy)
    dist_xy = torch.norm(offset_xy, dim=-1)
    pelvis_forward, _ = _command_frame_components(command.robot_anchor_lin_vel_w[:, :2], direction_xy)

    speed_reward = torch.clamp(
        (pelvis_forward - pelvis_forward_speed_min) / max(forward_speed_scale, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    no_ball = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    active = no_ball & (ball_ahead >= min_ball_ahead) & (dist_xy <= max_chase_xy_dist)
    return speed_reward * active.to(torch.float32)


def dribbling_rapid_retouch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    min_steps_between_touches: int = 22,
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    cg_gated: bool = False,
) -> torch.Tensor:
    """Penalty for a **new** legal gentle touch sooner than ``min_steps_between_touches``.

    Encourages kick → chase → kick cadence instead of tapping the ball every step.
    """
    if min_steps_between_touches <= 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )
    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    gentle = force_mag <= force_threshold
    touch = has_contact & is_ankle & gentle
    if cg_gated:
        touch = _dribbling_cg_gated_sim_contact(command, touch)

    prev_touch_name = "_dribbling_prev_retouch_touch"
    steps_name = "_dribbling_steps_since_legal_touch"
    prev_touch = getattr(env, prev_touch_name, None)
    if prev_touch is None or prev_touch.shape[0] != env.num_envs:
        prev_touch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    steps_since = getattr(env, steps_name, None)
    if steps_since is None or steps_since.shape[0] != env.num_envs:
        steps_since = torch.full(
            (env.num_envs,),
            fill_value=min_steps_between_touches + 1,
            device=env.device,
            dtype=torch.int32,
        )

    step_buf = getattr(env, "episode_length_buf", None)
    reset_mask = step_buf == 0 if step_buf is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    steps_since = torch.where(reset_mask, torch.full_like(steps_since, min_steps_between_touches + 1), steps_since + 1)

    new_touch = touch & ~prev_touch
    too_soon = new_touch & (steps_since < int(min_steps_between_touches))
    steps_since = torch.where(new_touch, torch.zeros_like(steps_since), steps_since)

    setattr(env, prev_touch_name, touch.detach().clone())
    setattr(env, steps_name, steps_since)
    return too_soon.to(torch.float32)


def dribbling_face_ball(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_distance: float = 0.05,
) -> torch.Tensor:
    """Reward ball ahead on task +X **and** pelvis facing task +X.

    Returns the product of (a) cos(angle from task +X to pelvis→ball) and
    (b) cos(pelvis yaw vs task +X), each clamped to [0, 1]. Sideways crab
    dribbling (body or ball mostly on ±Y) scores near zero.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    pelvis_pos_w = command.robot_pelvis_pos_w
    pelvis_quat_w = command.robot_pelvis_quat_w

    dx = task_forward_offset(ball_pos_w, pelvis_pos_w)
    dy = task_lateral_offset(ball_pos_w, pelvis_pos_w)
    dist = torch.norm(torch.stack([dx, dy], dim=-1), dim=-1)
    safe = dist > float(min_distance)
    ball_ahead = torch.where(
        safe,
        (dx / dist.clamp(min=1e-4)).clamp(min=0.0, max=1.0),
        torch.ones_like(dist),
    )
    pelvis_forward = task_pelvis_heading_cos_world_x(pelvis_quat_w).clamp(min=0.0, max=1.0)
    return ball_ahead * pelvis_forward


def dribbling_command_face_ball(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_distance: float = 0.05,
) -> torch.Tensor:
    """Reward ball placement and pelvis yaw aligned with the requested heading."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    ball_forward, _ = _command_frame_components(offset_xy, direction_xy)
    distance = torch.norm(offset_xy, dim=-1)
    ball_ahead = torch.where(
        distance > float(min_distance),
        (ball_forward / distance.clamp(min=1.0e-4)).clamp(min=0.0, max=1.0),
        torch.ones_like(distance),
    )

    local_forward = torch.zeros(env.num_envs, 3, device=env.device, dtype=offset_xy.dtype)
    local_forward[:, 0] = 1.0
    pelvis_forward_xy = quat_apply(command.robot_pelvis_quat_w, local_forward)[:, :2]
    pelvis_forward_xy = pelvis_forward_xy / torch.norm(pelvis_forward_xy, dim=-1, keepdim=True).clamp(min=1.0e-4)
    heading_alignment = torch.sum(pelvis_forward_xy * direction_xy, dim=-1).clamp(min=0.0, max=1.0)
    return ball_ahead * heading_alignment


def dribbling_pelvis_local_face_ball(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_distance: float = 0.05,
) -> torch.Tensor:
    """Reward a ball ahead of the current pelvis, independent of world yaw."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    offset_b = _pelvis_yaw_local_vector(
        command, soccer_ball.data.root_pos_w[:, :3] - command.robot_pelvis_pos_w
    )
    distance = torch.norm(offset_b[:, :2], dim=-1)
    return torch.where(
        distance > float(min_distance),
        (offset_b[:, 0] / distance.clamp(min=1.0e-4)).clamp(min=0.0, max=1.0),
        torch.ones_like(distance),
    )
