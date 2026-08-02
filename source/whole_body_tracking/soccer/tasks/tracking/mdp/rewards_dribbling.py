"""Dribbling-specific reward functions.

Encourages close ball control without strike-the-ball objectives. Contact
legality is **geometry-based**: the first ``num_ankle_links`` entries in
``all_body_cfg.body_names`` are ankle links (typically both feet); knees/wrists
listed after incur ``dribbling_undesired_contact_penalty`` when closest to the
ball. Trapping between feet is handled by ``dribbling_ball_trapped_penalty`` /
``dribbling_sustained_contact_penalty``. No ``kick_leg`` labels required.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude

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


def _dribbling_cg_gated_sim_contact(
    command: MotionCommand,
    sim_contact: torch.Tensor,
) -> torch.Tensor:
    """On CG-labeled clips, only count touches on annotated contact frames."""
    if not hasattr(command, "motion_has_dribble_cg_label"):
        return sim_contact
    labeled = command.motion_has_dribble_cg_label
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


_DRIBBLE_CONTACT_SURFACES = ("any", "inside_instep", "outside_instep")


def _dribbling_contact_surface_match(
    env: ManagerBasedRLEnv,
    all_body_cfg: SceneEntityCfg,
    closest_body_idx: torch.Tensor,
    *,
    contact_surface: str,
    num_ankle_links: int,
    medial_y_min: float = 0.018,
    instep_z_min: float = 0.010,
    instep_x_min: float = -0.035,
    instep_x_max: float = 0.140,
) -> torch.Tensor:
    """Check whether the ball centre lies in a requested foot-local instep zone.

    Isaac's ball contact sensor reports the net force on the ball but not the
    individual collision capsule that produced it.  We therefore identify the
    closest contacted ankle link and classify the ball centre in that link's
    local frame.  ``+Y`` points inward for the right foot and ``-Y`` inward
    for the left foot; ``+Z`` is the dorsal (instep) direction.  This is a
    stable semantic constraint despite ankle yaw/roll during a touch.
    """
    surface = str(contact_surface).lower().strip()
    if surface not in _DRIBBLE_CONTACT_SURFACES:
        raise ValueError(
            f"Unsupported contact_surface={contact_surface!r}; expected one of {_DRIBBLE_CONTACT_SURFACES}."
        )
    if surface == "any":
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if all_body_cfg is None or num_ankle_links <= 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if medial_y_min < 0.0 or instep_x_max <= instep_x_min:
        raise ValueError("Invalid foot-contact surface bounds.")

    names = tuple(all_body_cfg.body_names)
    if not names:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot = env.scene[all_body_cfg.name]
    cache_name = "_dribbling_body_indices_cache"
    cached = getattr(env, cache_name, None)
    if cached is None or cached.get("names") != names:
        body_indices = torch.as_tensor(
            robot.find_bodies(all_body_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=env.device,
        )
        setattr(env, cache_name, {"names": names, "idx": body_indices})
    body_indices = getattr(env, cache_name)["idx"]

    selected_cfg_idx = closest_body_idx.clamp(min=0, max=len(names) - 1)
    selected_body_idx = body_indices[selected_cfg_idx]
    batch_ids = torch.arange(env.num_envs, device=env.device)
    foot_pos_w = robot.data.body_pos_w[batch_ids, selected_body_idx]
    foot_quat_w = robot.data.body_quat_w[batch_ids, selected_body_idx]
    ball_pos_w = env.scene["soccer_ball"].data.root_pos_w[:, :3]
    ball_local = quat_apply_inverse(foot_quat_w, ball_pos_w - foot_pos_w)

    # In the robot's local coordinate system +Y is left.  Thus the right
    # foot's medial side is +Y, while the left foot's medial side is -Y.
    medial_sign = torch.zeros(env.num_envs, dtype=ball_local.dtype, device=env.device)
    for cfg_idx, body_name in enumerate(names):
        if "right_ankle" in body_name:
            medial_sign[selected_cfg_idx == cfg_idx] = 1.0
        elif "left_ankle" in body_name:
            medial_sign[selected_cfg_idx == cfg_idx] = -1.0

    instep = (
        (ball_local[:, 0] >= instep_x_min)
        & (ball_local[:, 0] <= instep_x_max)
        & (ball_local[:, 2] >= instep_z_min)
    )
    medial_offset = ball_local[:, 1] * medial_sign
    if surface == "inside_instep":
        side_match = medial_offset >= medial_y_min
    else:
        side_match = medial_offset <= -medial_y_min

    known_foot = medial_sign != 0.0
    legal_ankle = _is_dribble_legal_ankle_contact(closest_body_idx, num_ankle_links)
    return legal_ankle & known_foot & instep & side_match


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
    instep_z_min: float = 0.010,
    instep_x_min: float = -0.035,
    instep_x_max: float = 0.140,
    cg_gated: bool = False,
    min_pelvis_heading: float = 0.0,
) -> torch.Tensor:
    """Reward a new gentle touch in the selected foot/instep region.

    ``contact_surface`` is ``any`` (legacy), ``inside_instep``, or
    ``outside_instep``.  The latter two use the contacted foot's local frame,
    so selecting one is stronger than merely selecting the left or right foot.
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
        contact_surface=contact_surface,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
        instep_z_min=instep_z_min,
        instep_x_min=instep_x_min,
        instep_x_max=instep_x_max,
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
# 3c) Undesired Contact Penalty — severe instant penalty for wrong body
# ---------------------------------------------------------------------------

def dribbling_undesired_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    contact_surface: str = "any",
    medial_y_min: float = 0.018,
    instep_z_min: float = 0.010,
    instep_x_min: float = -0.035,
    instep_x_max: float = 0.140,
) -> torch.Tensor:
    """1.0 for a non-ankle contact or a touch outside the requested instep zone."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    surface_match = _dribbling_contact_surface_match(
        env,
        all_body_cfg,
        closest_idx,
        contact_surface=contact_surface,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
        instep_z_min=instep_z_min,
        instep_x_min=instep_x_min,
        instep_x_max=instep_x_max,
    )

    penalty = (has_contact & (~is_ankle | ~surface_match)).to(torch.float32)

    return penalty


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


def dribbling_cg_demo_ball_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.32,
) -> torch.Tensor:
    """Shaped tracking of the simulated ball toward the stitched demo trajectory."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    goal_w, mask = command.get_dribble_demo_ball_goal_world()
    if goal_w is None or mask is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    ball = env.scene["soccer_ball"].data.root_pos_w[:, :3]
    err = torch.norm(ball - goal_w, dim=-1)
    rew = torch.exp(-err / max(std, 1e-6))
    return rew * mask.to(torch.float32)


def dribbling_cg_contact_consistency(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """1.0 when sim contact presence matches the annotated CG contact bit."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = command.motion_has_dribble_cg_label
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
    labeled = command.motion_has_dribble_cg_label
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
    instep_z_min: float = 0.010,
    instep_x_min: float = -0.035,
    instep_x_max: float = 0.140,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Reward a labeled-foot contact, optionally limited to one instep surface."""
    if all_body_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = command.motion_has_dribble_cg_label
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
        contact_surface=contact_surface,
        num_ankle_links=2,
        medial_y_min=medial_y_min,
        instep_z_min=instep_z_min,
        instep_x_min=instep_x_min,
        instep_x_max=instep_x_max,
    )
    match = (closest == expected) & surface_match & has_contact & active
    return match.to(torch.float32)


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
    labeled = command.motion_has_dribble_cg_foot_ball_dist_label
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
