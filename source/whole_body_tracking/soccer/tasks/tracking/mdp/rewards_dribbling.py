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
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.rewards import motion_relative_foot_position_error_exp
from soccer.tasks.tracking.mdp.task_frame import (
    DribblePhaseBundle,
    compute_dribble_phase_bundle,
    forward_dominance_gate,
    update_monotonic_dribble_phase_level,
    update_seek_touch_zone_steps,
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


def _min_ankle_ball_distance(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg,
    ball_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Minimum 3D distance from any listed foot body to the ball, shape ``(num_envs,)``."""
    robot = env.scene[foot_cfg.name]
    cache_name = "_dribbling_phase_foot_idx"
    cached = getattr(env, cache_name, None)
    names_t = tuple(foot_cfg.body_names)
    if cached is None or cached.get("names") != names_t:
        foot_indices = torch.as_tensor(
            robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=env.device,
        )
        setattr(env, cache_name, {"names": names_t, "idx": foot_indices})
    foot_indices = getattr(env, cache_name)["idx"]
    feet_pos = robot.data.body_pos_w[:, foot_indices]
    return torch.norm(feet_pos - ball_pos_w.unsqueeze(1), dim=-1).min(dim=-1).values


def gather_dribble_phase_bundle(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    recent_contact_window: int = 8,
    foot_cfg: SceneEntityCfg | None = None,
    *,
    approach_run_min_ahead: float = 0.35,
    approach_run_max_dist: float = 1.85,
    approach_max_dist: float = 0.55,
    approach_ball_speed_max: float = 0.35,
    approach_ball_stopped_max_speed: float = 0.08,
    approach_min_x_ahead: float = 0.10,
    approach_max_x_ahead: float = 0.85,
    close_max_dist: float = 0.52,
    close_x_min: float = 0.15,
    close_x_max: float = 0.65,
    seek_touch_min_steps: int = 2,
    seek_touch_commit_dist: float = 0.24,
) -> DribblePhaseBundle:
    """Shared per-step phase state (contact, geometry, transition steps, ankle distance)."""
    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    has_contact = fmag > contact_force_threshold
    recent_contact = _dribbling_recent_contact_gate(
        env,
        ball_sensor_name,
        contact_force_threshold,
        recent_contact_window,
        command=command,
    )

    soccer_ball = env.scene["soccer_ball"]
    ball_pos_w = soccer_ball.data.root_pos_w
    pelvis_pos_w = command.robot_pelvis_pos_w
    x_ahead = task_forward_offset(ball_pos_w, pelvis_pos_w)
    dist_xy = torch.norm(ball_pos_w[:, :2] - pelvis_pos_w[:, :2], dim=-1)
    ball_vel_w = soccer_ball.data.root_lin_vel_w
    ball_speed_xy = torch.norm(ball_vel_w[:, :2], dim=-1)
    pelvis_index = command.robot.body_names.index("pelvis")
    pelvis_forward = task_forward_speed(command.robot.data.body_lin_vel_w[:, pelvis_index])
    ball_forward = task_forward_speed(ball_vel_w)

    min_ankle_dist = torch.full_like(dist_xy, 999.0)
    if foot_cfg is not None:
        min_ankle_dist = _min_ankle_ball_distance(env, foot_cfg, ball_pos_w)

    zero_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    phase_kwargs = dict(
        min_ankle_ball_dist=min_ankle_dist,
        approach_run_min_ahead=approach_run_min_ahead,
        approach_run_max_dist=approach_run_max_dist,
        approach_max_dist=approach_max_dist,
        approach_ball_speed_max=approach_ball_speed_max,
        approach_ball_stopped_max_speed=approach_ball_stopped_max_speed,
        approach_min_x_ahead=approach_min_x_ahead,
        approach_max_x_ahead=approach_max_x_ahead,
        close_max_dist=close_max_dist,
        close_x_min=close_x_min,
        close_x_max=close_x_max,
        seek_touch_min_steps=seek_touch_min_steps,
        seek_touch_commit_dist=seek_touch_commit_dist,
    )
    zone_pre = compute_dribble_phase_bundle(
        has_contact,
        recent_contact,
        x_ahead,
        dist_xy,
        ball_speed_xy,
        steps_in_seek_touch_zone=zero_steps,
        **phase_kwargs,
    )
    zone_steps = update_seek_touch_zone_steps(env, zone_pre.seek_touch_zone, has_contact)
    geo = compute_dribble_phase_bundle(
        has_contact,
        recent_contact,
        x_ahead,
        dist_xy,
        ball_speed_xy,
        steps_in_seek_touch_zone=zone_steps,
        **phase_kwargs,
    )
    level = update_monotonic_dribble_phase_level(
        env, has_contact, recent_contact, geo.seek_touch,
    )
    return compute_dribble_phase_bundle(
        has_contact,
        recent_contact,
        x_ahead,
        dist_xy,
        ball_speed_xy,
        steps_in_seek_touch_zone=zone_steps,
        phase_level=level,
        **phase_kwargs,
    )


# ---------------------------------------------------------------------------
# 1b) Phased forward velocity — approach (run) / seek_touch / touch
# ---------------------------------------------------------------------------


def dribbling_phased_forward_velocity(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.35,
    velocity_frame: str = "world",
    min_forward_dominance: float = 0.55,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    recent_contact_window: int = 8,
    v_touch: float = 0.20,
    approach_overshoot: float = 0.14,
    v_approach_floor: float = 0.38,
    v_approach_min: float = 0.28,
    v_approach_stopped_ball: float = 0.22,
    backward_penalty_scale: float = 0.35,
    approach_run_min_ahead: float = 0.35,
    approach_run_max_dist: float = 1.85,
    approach_max_dist: float = 0.55,
    approach_ball_speed_max: float = 0.35,
    approach_ball_stopped_max_speed: float = 0.08,
    approach_min_x_ahead: float = 0.10,
    approach_max_x_ahead: float = 0.85,
    close_max_dist: float = 0.52,
    close_x_min: float = 0.15,
    close_x_max: float = 0.65,
    seek_touch_min_steps: int = 2,
    seek_touch_commit_dist: float = 0.24,
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward pelvis speed vs phase target: approach (run, faster than ball), seek_touch / touch slow."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    pelvis_index = robot.body_names.index("pelvis")
    pelvis_lin_vel_w = robot.data.body_lin_vel_w[:, pelvis_index]

    if velocity_frame == "world":
        forward_speed = task_forward_speed(pelvis_lin_vel_w)
        dominance = task_velocity_forward_dominance(pelvis_lin_vel_w)
    elif velocity_frame == "pelvis":
        pelvis_quat_w = robot.data.body_quat_w[:, pelvis_index]
        pelvis_lin_vel_local = quat_apply(quat_inv(pelvis_quat_w), pelvis_lin_vel_w)
        forward_speed = pelvis_lin_vel_local[:, 0]
        dominance = task_velocity_forward_dominance(pelvis_lin_vel_local)
    else:
        raise ValueError(f"Unsupported velocity_frame={velocity_frame!r}; use 'world' or 'pelvis'.")

    bundle = gather_dribble_phase_bundle(
        env,
        command,
        ball_sensor_name,
        contact_force_threshold,
        recent_contact_window,
        foot_cfg,
        approach_run_min_ahead=approach_run_min_ahead,
        approach_run_max_dist=approach_run_max_dist,
        approach_max_dist=approach_max_dist,
        approach_ball_speed_max=approach_ball_speed_max,
        approach_ball_stopped_max_speed=approach_ball_stopped_max_speed,
        approach_min_x_ahead=approach_min_x_ahead,
        approach_max_x_ahead=approach_max_x_ahead,
        close_max_dist=close_max_dist,
        close_x_min=close_x_min,
        close_x_max=close_x_max,
        seek_touch_min_steps=seek_touch_min_steps,
        seek_touch_commit_dist=seek_touch_commit_dist,
    )

    soccer_ball = env.scene["soccer_ball"]
    ball_vel_w = soccer_ball.data.root_lin_vel_w
    ball_forward = task_forward_speed(ball_vel_w)
    ball_speed_xy = torch.norm(ball_vel_w[:, :2], dim=-1)
    ball_pos_w = soccer_ball.data.root_pos_w
    x_ahead = task_forward_offset(ball_pos_w, command.robot_pelvis_pos_w)
    ball_stopped = ball_speed_xy <= approach_ball_stopped_max_speed
    approach_target_moving = (ball_forward + approach_overshoot).clamp(min=v_approach_floor)
    approach_target = torch.where(
        ball_stopped,
        torch.full_like(forward_speed, v_approach_stopped_ball),
        approach_target_moving,
    )

    target_speed = torch.full_like(forward_speed, v_approach_floor)
    target_speed = torch.where(bundle.approach, approach_target, target_speed)
    target_speed = torch.where(bundle.seek_touch, torch.full_like(target_speed, v_touch), target_speed)
    target_speed = torch.where(bundle.touch, torch.full_like(target_speed, v_touch), target_speed)

    error = (forward_speed - target_speed) ** 2
    reward = torch.exp(-error / max(std, 1e-6) ** 2)
    reward = reward * forward_dominance_gate(dominance, min_forward_dominance)

    approach_speed_gate = torch.clamp(forward_speed / max(v_approach_min, 1e-6), max=1.0)
    stopped_speed_gate = torch.clamp(forward_speed / max(v_approach_stopped_ball * 0.65, 1e-6), max=1.0)
    reward = torch.where(
        bundle.approach & ball_stopped,
        reward * stopped_speed_gate,
        torch.where(bundle.approach, reward * approach_speed_gate, reward),
    )
    ball_ahead = x_ahead >= 0.15
    backward = torch.clamp(-forward_speed, min=0.0)
    backward_pen = torch.exp(-(backward ** 2) / max(backward_penalty_scale, 1e-6) ** 2)
    reward = torch.where(bundle.approach & ball_ahead, reward * backward_pen, reward)
    return reward


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
    min_forward_dominance: float = 0.45,
) -> torch.Tensor:
    """Reward alignment between the soccer ball velocity and the robot pelvis velocity.

    A cosine-similarity style reward: when the ball moves in the same direction
    and at a similar speed as the robot, the reward is maximised.

    Optional **anti-cheese gates** (defaults preserve legacy behaviour):

    - ``pelvis_speed_min`` / ``ball_speed_min``: multiply the reward by
      ``clamp(|v_xy| / min, max=1)`` so near-zero speeds do not yield a full
      score from ``exp(0)==1``.
    - ``require_contact``: multiply by a contact gate. If
      ``recent_contact_window > 0``, contact must have occurred within the last
      N steps (shared counter with forward-progress reward); otherwise the
      current step must show contact.

    Returns a value in ``[0, 1]`` per environment.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    # Ball velocity (world frame, xy only)
    ball_vel_xy = soccer_ball.data.root_lin_vel_w[:, :2]  # (N, 2)
    # Robot pelvis velocity (world frame, xy only)
    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]  # (N, 2)

    # Squared difference as the error signal
    vel_diff = ball_vel_xy - pelvis_vel_xy
    error = torch.sum(vel_diff * vel_diff, dim=-1)  # (N,)

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
        dominance = task_velocity_forward_dominance(pelvis_vel_xy)
        gate = gate * forward_dominance_gate(dominance, min_forward_dominance)

    return base * gate


# ---------------------------------------------------------------------------
# 1c) Phase-wise ball-speed requirement (explicit target + urgency)
# ---------------------------------------------------------------------------

def dribbling_phase_ball_speed_requirement(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    recent_contact_window: int = 8,
    foot_cfg: SceneEntityCfg | None = None,
    std: float = 0.22,
    approach_ball_speed_far: float = 0.08,
    approach_ball_speed_near: float = 0.22,
    touch_ball_speed_target: float = 0.30,
    approach_near_dist: float = 0.62,
    approach_min_ratio: float = 0.75,
    touch_min_ratio: float = 0.80,
    approach_urgency_scale: float = 0.65,
    approach_run_min_ahead: float = 0.35,
    approach_run_max_dist: float = 1.85,
    approach_max_dist: float = 0.55,
    approach_ball_speed_max: float = 0.35,
    approach_ball_stopped_max_speed: float = 0.08,
    approach_min_x_ahead: float = 0.10,
    approach_max_x_ahead: float = 0.85,
    close_max_dist: float = 0.52,
    close_x_min: float = 0.15,
    close_x_max: float = 0.65,
    seek_touch_min_steps: int = 2,
    seek_touch_commit_dist: float = 0.24,
) -> torch.Tensor:
    """Explicit ball-speed objective for phase transition.

    Logic:
    - ``approach``: target speed ramps up as robot gets closer to the ball.
    - ``seek_touch``/``touch``: require a higher fixed ball speed setpoint.
    - Near-ball low speed in ``approach`` without recent contact gets extra
      urgency suppression, pushing policy to enter ``seek_touch`` and touch.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    bundle = gather_dribble_phase_bundle(
        env,
        command,
        ball_sensor_name,
        contact_force_threshold,
        recent_contact_window,
        foot_cfg,
        approach_run_min_ahead=approach_run_min_ahead,
        approach_run_max_dist=approach_run_max_dist,
        approach_max_dist=approach_max_dist,
        approach_ball_speed_max=approach_ball_speed_max,
        approach_ball_stopped_max_speed=approach_ball_stopped_max_speed,
        approach_min_x_ahead=approach_min_x_ahead,
        approach_max_x_ahead=approach_max_x_ahead,
        close_max_dist=close_max_dist,
        close_x_min=close_x_min,
        close_x_max=close_x_max,
        seek_touch_min_steps=seek_touch_min_steps,
        seek_touch_commit_dist=seek_touch_commit_dist,
    )

    soccer_ball = env.scene["soccer_ball"]
    ball_speed_xy = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    dist_xy = torch.norm(
        soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2], dim=-1
    )

    near_ratio = 1.0 - torch.clamp(dist_xy / max(approach_near_dist, 1e-6), 0.0, 1.0)
    approach_target = approach_ball_speed_far + (approach_ball_speed_near - approach_ball_speed_far) * near_ratio
    touch_target = torch.full_like(ball_speed_xy, touch_ball_speed_target)

    phase_target = torch.full_like(ball_speed_xy, approach_ball_speed_far)
    phase_target = torch.where(bundle.approach, approach_target, phase_target)
    phase_target = torch.where(bundle.seek_touch | bundle.touch, touch_target, phase_target)

    error = ball_speed_xy - phase_target
    reward = torch.exp(-(error * error) / (max(std, 1e-6) ** 2))

    min_req = torch.where(
        bundle.approach,
        approach_target * approach_min_ratio,
        touch_target * touch_min_ratio,
    ).clamp(min=1e-4)
    speed_gate = torch.clamp(ball_speed_xy / min_req, max=1.0)
    reward = reward * (0.35 + 0.65 * speed_gate)

    recent_contact = _dribbling_recent_contact_gate(
        env,
        ball_sensor_name,
        contact_force_threshold,
        recent_contact_window,
        command=command,
    )
    no_recent_contact = recent_contact <= 0.5
    speed_deficit = torch.clamp((approach_target - ball_speed_xy) / approach_target.clamp(min=1e-4), 0.0, 1.0)
    urgency = bundle.approach.to(torch.float32) * near_ratio * speed_deficit * no_recent_contact.to(torch.float32)
    urgency_scale = max(0.0, min(1.0, float(approach_urgency_scale)))
    reward = reward * (1.0 - urgency_scale * urgency)

    return reward.clamp(min=0.0, max=1.0)


def dribbling_taskframe_route_speed_requirement(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    recent_contact_window: int = 8,
    speed_std: float = 0.25,
    segment_progress_1: float = 0.45,
    segment_progress_2: float = 0.80,
    target_speed_seg1: float = 0.18,
    target_speed_seg2: float = 0.32,
    target_speed_seg3: float = 0.24,
    min_speed_ratio: float = 0.75,
    low_speed_urgency_scale: float = 0.55,
    min_route_len: float = 0.6,
) -> torch.Tensor:
    """Route-conditioned ball-speed target in task frame.

    The route is defined per env by ``initial_target_point_pos -> target_destination_pos``.
    Ball speed target is piecewise over route progress ``s in [0, 1]``:
      - segment 1: early route (e.g. from (1,1) toward (12,15))
      - segment 2: middle route
      - segment 3: late route
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    start_xy = command.initial_target_point_pos[:, :2]
    goal_xy = command.target_destination_pos[:, :2]
    ball_xy = command.target_point_pos[:, :2]
    ball_vel_xy = soccer_ball.data.root_lin_vel_w[:, :2]
    ball_speed_xy = torch.norm(ball_vel_xy, dim=-1)

    route = goal_xy - start_xy
    route_len = torch.norm(route, dim=-1)
    route_unit = route / route_len.unsqueeze(-1).clamp(min=1e-6)
    valid_route = route_len >= min_route_len

    rel = ball_xy - start_xy
    route_len_sq = torch.sum(route * route, dim=-1).clamp(min=1e-6)
    s = torch.clamp(torch.sum(rel * route, dim=-1) / route_len_sq, 0.0, 1.0)

    p1 = float(max(0.0, min(0.95, segment_progress_1)))
    p2 = float(max(p1 + 1e-3, min(0.99, segment_progress_2)))

    s12 = torch.clamp((s - p1) / max(p2 - p1, 1e-6), 0.0, 1.0)
    s23 = torch.clamp((s - p2) / max(1.0 - p2, 1e-6), 0.0, 1.0)
    v1 = torch.full_like(s, target_speed_seg1)
    v2 = torch.full_like(s, target_speed_seg2)
    v3 = torch.full_like(s, target_speed_seg3)

    target_speed = torch.where(s < p1, v1, v1 + (v2 - v1) * s12)
    target_speed = torch.where(s >= p2, v2 + (v3 - v2) * s23, target_speed)

    err = ball_speed_xy - target_speed
    reward = torch.exp(-(err * err) / (max(speed_std, 1e-6) ** 2))

    # Encourage speed along route direction, not lateral drift.
    forward_along_route = torch.sum(ball_vel_xy * route_unit, dim=-1)
    route_gate = torch.clamp(forward_along_route / ball_speed_xy.clamp(min=1e-6), min=0.0, max=1.0)
    reward = reward * (0.4 + 0.6 * route_gate)

    # When speed drops below route target and there is no recent touch, reduce reward
    # to create urgency for seek-touch/touch.
    recent_contact = _dribbling_recent_contact_gate(
        env,
        ball_sensor_name,
        contact_force_threshold,
        recent_contact_window,
        command=command,
    )
    speed_floor = (target_speed * min_speed_ratio).clamp(min=1e-4)
    deficit = torch.clamp((speed_floor - ball_speed_xy) / speed_floor, 0.0, 1.0)
    urgency = deficit * (recent_contact <= 0.5).to(deficit.dtype)
    urgency_scale = max(0.0, min(1.0, float(low_speed_urgency_scale)))
    reward = reward * (1.0 - urgency_scale * urgency)

    reward = torch.where(valid_route, reward, torch.zeros_like(reward))
    return reward.clamp(min=0.0, max=1.0)


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
        dist_xy = torch.norm(ball_pos_w[:, :2] - pelvis_pos_w[:, :2], dim=-1)
        in_corridor = (
            (x_task >= near_dist)
            & (x_task <= far_dist)
            & (torch.abs(y_task) <= zone_lateral_abs_max)
        )
        # Ball ahead and within dribble reach but no sensor touch — cut proximity hard.
        in_approach_no_touch = (x_task > 0.0) & (dist_xy <= 0.58)
        fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
        no_touch = fmag <= contact_force_threshold
        damp = torch.where(
            (in_corridor | in_approach_no_touch) & no_touch,
            torch.full_like(proximity_reward, no_contact_zone_damping),
            torch.ones_like(proximity_reward),
        )
        proximity_reward = proximity_reward * damp

    return proximity_reward


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
    min_x_ahead: float = 0.10,
) -> torch.Tensor:
    """Penalty when the ball is close in front but pelvis is nearly static and there is no contact.

    Only targets **freeze** exploits, not moderate-speed stepping needed to reach touch.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]
    dist_xy = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)
    x_ahead = task_forward_offset(soccer_ball.data.root_pos_w, command.robot_pelvis_pos_w)

    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
    pelvis_sp = torch.norm(pelvis_vel_xy, dim=-1)

    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    no_touch = fmag <= contact_force_threshold

    near = dist_xy <= max_xy_dist
    ball_in_front = x_ahead >= min_x_ahead
    slow = pelvis_sp <= pelvis_speed_max
    return (near & ball_in_front & no_touch & slow).to(torch.float32)


def dribbling_approach_touch_bridge(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    recent_contact_window: int = 8,
    std: float = 0.20,
    approach_run_min_ahead: float = 0.35,
    approach_run_max_dist: float = 1.85,
    approach_max_dist: float = 0.55,
    approach_ball_speed_max: float = 0.35,
    approach_ball_stopped_max_speed: float = 0.08,
    approach_min_x_ahead: float = 0.10,
    approach_max_x_ahead: float = 0.85,
    close_max_dist: float = 0.52,
    close_x_min: float = 0.15,
    close_x_max: float = 0.65,
    seek_touch_min_steps: int = 2,
    seek_touch_commit_dist: float = 0.24,
    approach_scale: float = 0.35,
    seek_touch_scale: float = 1.0,
) -> torch.Tensor:
    """Foot–ball shaping while running (``approach``); stronger in ``seek_touch``."""
    if foot_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command: MotionCommand = env.command_manager.get_term(command_name)
    bundle = gather_dribble_phase_bundle(
        env,
        command,
        ball_sensor_name,
        contact_force_threshold,
        recent_contact_window,
        foot_cfg,
        approach_run_min_ahead=approach_run_min_ahead,
        approach_run_max_dist=approach_run_max_dist,
        approach_max_dist=approach_max_dist,
        approach_ball_speed_max=approach_ball_speed_max,
        approach_ball_stopped_max_speed=approach_ball_stopped_max_speed,
        approach_min_x_ahead=approach_min_x_ahead,
        approach_max_x_ahead=approach_max_x_ahead,
        close_max_dist=close_max_dist,
        close_x_min=close_x_min,
        close_x_max=close_x_max,
        seek_touch_min_steps=seek_touch_min_steps,
        seek_touch_commit_dist=seek_touch_commit_dist,
    )

    soccer_ball = env.scene["soccer_ball"]
    min_dist = _min_ankle_ball_distance(env, foot_cfg, soccer_ball.data.root_pos_w)
    shaping = torch.exp(-(min_dist ** 2) / (max(std, 1e-6) ** 2))

    gate = torch.zeros(env.num_envs, device=env.device, dtype=shaping.dtype)
    in_zone = bundle.seek_touch_zone & (~bundle.seek_touch)
    zone_scale = approach_scale + (seek_touch_scale - approach_scale) * 0.65
    gate = torch.where(bundle.approach, torch.full_like(gate, approach_scale), gate)
    gate = torch.where(in_zone, torch.full_like(gate, zone_scale), gate)
    gate = torch.where(bundle.seek_touch, torch.full_like(gate, seek_touch_scale), gate)
    return shaping * gate


def dribbling_approach_touch_transition(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    recent_contact_window: int = 8,
    approach_run_min_ahead: float = 0.35,
    approach_run_max_dist: float = 1.85,
    approach_max_dist: float = 0.55,
    approach_ball_speed_max: float = 0.35,
    approach_ball_stopped_max_speed: float = 0.08,
    approach_min_x_ahead: float = 0.10,
    approach_max_x_ahead: float = 0.85,
    close_max_dist: float = 0.52,
    close_x_min: float = 0.15,
    close_x_max: float = 0.65,
    seek_touch_min_steps: int = 2,
    seek_touch_commit_dist: float = 0.24,
) -> torch.Tensor:
    """Transition shaping for ``approach -> seek_touch -> touch``.

    Provides dense guidance before commit, stronger reward during seek-touch hold,
    and a one-step completion bonus on first touch right after seek-touch.
    """
    if foot_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command: MotionCommand = env.command_manager.get_term(command_name)
    bundle = gather_dribble_phase_bundle(
        env,
        command,
        ball_sensor_name,
        contact_force_threshold,
        recent_contact_window,
        foot_cfg,
        approach_run_min_ahead=approach_run_min_ahead,
        approach_run_max_dist=approach_run_max_dist,
        approach_max_dist=approach_max_dist,
        approach_ball_speed_max=approach_ball_speed_max,
        approach_ball_stopped_max_speed=approach_ball_stopped_max_speed,
        approach_min_x_ahead=approach_min_x_ahead,
        approach_max_x_ahead=approach_max_x_ahead,
        close_max_dist=close_max_dist,
        close_x_min=close_x_min,
        close_x_max=close_x_max,
        seek_touch_min_steps=seek_touch_min_steps,
        seek_touch_commit_dist=seek_touch_commit_dist,
    )
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    # 1) Dense pre-commit shaping while already in the near-ball setup zone.
    pre_seek_zone = bundle.seek_touch_zone & (~bundle.seek_touch) & (~bundle.touch)
    min_steps = max(int(seek_touch_min_steps), 1)
    zone_progress = torch.clamp(bundle.steps_in_seek_touch_zone.to(torch.float32) / float(min_steps), 0.0, 1.0)
    reward = torch.where(pre_seek_zone, 0.35 + 0.25 * zone_progress, reward)

    # 2) Main hold reward in seek-touch (encourage stabilizing kick pose).
    seek_hold = bundle.seek_touch & (~bundle.touch)
    reward = torch.where(seek_hold, 0.70 + 0.30 * zone_progress, reward)

    # 3) Transition-event bonuses: first entry into seek-touch, then first touch.
    prev_seek_name = "_dribble_prev_seek_touch_mask"
    prev_touch_name = "_dribble_prev_touch_mask"
    prev_seek = getattr(env, prev_seek_name, None)
    prev_touch = getattr(env, prev_touch_name, None)
    if prev_seek is None or prev_seek.shape[0] != env.num_envs:
        prev_seek = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if prev_touch is None or prev_touch.shape[0] != env.num_envs:
        prev_touch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    entered_seek = seek_hold & (~prev_seek)
    just_touched = bundle.touch & (~prev_touch)
    touched_after_seek = just_touched & prev_seek

    reward = reward + entered_seek.to(torch.float32) * 0.15
    reward = reward + touched_after_seek.to(torch.float32) * 0.30
    reward = torch.clamp(reward, 0.0, 1.0)

    setattr(env, prev_seek_name, seek_hold)
    setattr(env, prev_touch_name, bundle.touch)
    return reward


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


def dribbling_ball_must_move_after_touch(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    grace_steps: int = 50,
    post_touch_grace_steps: int = 20,
    min_ball_speed: float = 0.12,
    max_stagnant_steps: int = 45,
) -> torch.Tensor:
    """Terminate if the ball stays nearly still after the robot has engaged it.

    The ball may start stationary at spawn; after the first touch, it must begin rolling.
    """
    step_buf = getattr(env, "episode_length_buf", torch.zeros(env.num_envs, device=env.device))
    past_grace = step_buf > grace_steps
    reset_m = step_buf == 0

    has_contact = soccer_ball_contact_force_magnitude(env, ball_sensor_name) > contact_force_threshold
    ever_name = "_dribbling_ever_touched_ball"
    ever = getattr(env, ever_name, None)
    if ever is None or ever.shape[0] != env.num_envs:
        ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ever = ever | has_contact
    setattr(env, ever_name, ever)

    since_touch_name = "_dribbling_steps_since_first_touch"
    since_touch = getattr(env, since_touch_name, None)
    if since_touch is None or since_touch.shape[0] != env.num_envs:
        since_touch = torch.full((env.num_envs,), 10_000, dtype=torch.int32, device=env.device)
    since_touch = torch.where(reset_m, torch.full_like(since_touch, 10_000), since_touch)
    since_touch = torch.where(has_contact & (since_touch > 5000), torch.zeros_like(since_touch), since_touch)
    since_touch = torch.where(ever, since_touch + 1, since_touch)
    setattr(env, since_touch_name, since_touch)

    soccer_ball = env.scene["soccer_ball"]
    ball_speed_xy = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    stagnant = ball_speed_xy < min_ball_speed
    enforce = past_grace & ever & (since_touch > post_touch_grace_steps) & stagnant

    cnt_name = "_dribbling_ball_stagnant_steps"
    cnt = getattr(env, cnt_name, None)
    if cnt is None or cnt.shape[0] != env.num_envs:
        cnt = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    cnt = torch.where(reset_m | has_contact, torch.zeros_like(cnt), cnt)
    cnt = torch.where(enforce, cnt + 1, torch.zeros_like(cnt))
    setattr(env, cnt_name, cnt)

    return cnt >= max_stagnant_steps


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
    cg_gated: bool = False,
) -> torch.Tensor:
    """Reward a **new** gentle ankle touch (rising edge), not sustained trapping."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    gentle = force_mag <= force_threshold
    touch = has_contact & is_ankle & gentle
    if cg_gated:
        touch = _dribbling_cg_gated_sim_contact(command, touch)

    prev_name = "_dribbling_prev_legal_foot_touch"
    prev = getattr(env, prev_name, None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    new_touch = touch & ~prev
    setattr(env, prev_name, touch.detach().clone())
    return new_touch.to(torch.float32)


def dribbling_instep_touch_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    min_alignment: float = 0.25,
    cg_gated: bool = False,
) -> torch.Tensor:
    """Bonus on a new gentle touch when the foot instep (medial side) faces the ball.

    Encourages inside/outside instep passes instead of toe pokes. Axis convention
    follows G1 ankle-roll links; verify in sim if alignment looks inverted.
    """
    if all_body_cfg is None:
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

    prev_name = "_dribbling_prev_instep_touch"
    prev = getattr(env, prev_name, None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    new_touch = touch & ~prev
    setattr(env, prev_name, touch.detach().clone())

    robot = env.scene[all_body_cfg.name]
    soccer_ball = env.scene["soccer_ball"]
    cache_name = "_dribbling_body_indices_cache"
    cached = getattr(env, cache_name, None)
    names_t = tuple(all_body_cfg.body_names)
    if cached is None or cached.get("names") != names_t:
        body_indices = torch.as_tensor(
            robot.find_bodies(all_body_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=env.device,
        )
        setattr(env, cache_name, {"names": names_t, "idx": body_indices})
    body_indices = getattr(env, cache_name)["idx"]

    num_envs = env.num_envs
    env_ids = torch.arange(num_envs, device=env.device)
    bi = body_indices[closest_idx]
    foot_quat = robot.data.body_quat_w[env_ids, bi]
    foot_pos = robot.data.body_pos_w[env_ids, bi]
    ball_pos = soccer_ball.data.root_pos_w

    is_left = torch.tensor(
        ["left" in name for name in all_body_cfg.body_names],
        device=env.device,
        dtype=torch.bool,
    )
    local_medial = torch.zeros(num_envs, 3, device=env.device, dtype=foot_quat.dtype)
    local_medial[:, 1] = torch.where(is_left[closest_idx], 1.0, -1.0)
    instep_dir = quat_apply(foot_quat, local_medial)

    to_ball = ball_pos - foot_pos
    to_ball_xy = to_ball[:, :2]
    instep_xy = instep_dir[:, :2]
    cos_align = torch.sum(instep_xy * to_ball_xy, dim=-1) / (
        torch.norm(instep_xy, dim=-1).clamp(min=1e-6) * torch.norm(to_ball_xy, dim=-1).clamp(min=1e-6)
    )
    reward = torch.clamp((cos_align - min_alignment) / (1.0 - min_alignment + 1e-6), 0.0, 1.0)
    return reward * new_touch.to(reward.dtype)


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
) -> torch.Tensor:
    """1.0 when there is contact and the closest body is **not** an ankle link."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)

    penalty = (has_contact & ~is_ankle).to(torch.float32)

    return penalty


# ---------------------------------------------------------------------------
# 4) Annotated contact-graph (dribbling) — demo ball + label consistency
# ---------------------------------------------------------------------------


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
) -> torch.Tensor:
    """1.0 when sim contact presence matches the annotated CG contact bit."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = command.motion_has_dribble_cg_label
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref = command.dribble_cg_contact_ref
    sim_c = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    agree = (ref == sim_c).to(torch.float32)
    return agree * labeled.to(torch.float32)


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
) -> torch.Tensor:
    """When the label specifies a foot during contact, reward matching closest ankle."""
    if all_body_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = command.motion_has_dribble_cg_label
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref_c = command.dribble_cg_contact_ref
    ref_f = command.dribble_cg_foot_ref
    active = labeled & ref_c & (ref_f >= 0)

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

    match = (closest == expected) & has_contact & active
    return match.to(torch.float32)


def dribbling_cg_foot_ball_distance_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.12,
    use_xy_only: bool = False,
    left_ankle_body_name: str = "left_ankle_roll_link",
    right_ankle_body_name: str = "right_ankle_roll_link",
) -> torch.Tensor:
    """Match sim foot–ball distance to demo distance from synthesized CG trajectory.

    Requires ``dribble_cg_foot_ball_dist`` in motion ``.npz`` (see
    ``scripts/rsl_rl/synthesize_dribble_ball_traj.py``). At labeled frames:

    - ``ref_dist`` = demo distance from retargeted foot to synthesized ball.
    - ``sim_dist`` = distance from the labeled foot to the **sim** ball.
    - reward = ``exp(-(sim_dist - ref_dist)^2 / std^2)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = command.motion_has_dribble_cg_foot_ball_dist_label
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref_dist = command.dribble_cg_foot_ball_dist_ref
    ref_foot = command.dribble_cg_foot_ref
    active = labeled & (ref_dist >= 0.0) & (ref_foot >= 0)

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
) -> torch.Tensor:
    """Foot imitation reward active only while the ball is **not** in contact.

    Between touches the policy should follow the reference gait (alternating
  steps) instead of freezing in a kick-ready stance with the dribble foot forward.
    """
    base = motion_relative_foot_position_error_exp(
        env, command_name, std, foot_body_names=foot_body_names
    )
    no_ball = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    return base * no_ball.to(torch.float32)


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
