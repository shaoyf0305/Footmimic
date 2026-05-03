"""Dribbling-specific reward functions.

These rewards encourage the robot to keep the ball under close control while
moving forward, as opposed to the kicking rewards which encourage striking the
ball hard in a target direction.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_inv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand

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
) -> torch.Tensor:
    """Scalar force magnitude on the ball, shape ``(num_envs,)``."""
    f = soccer_ball_contact_net_force_w(env, ball_sensor_name)
    return torch.norm(f, dim=-1)


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
) -> torch.Tensor:
    """Reward alignment between the soccer ball velocity and the robot pelvis velocity.

    A cosine-similarity style reward: when the ball moves in the same direction
    and at a similar speed as the robot, the reward is maximised.

    Optional **anti-cheese gates** (defaults preserve legacy behaviour):

    - ``pelvis_speed_min`` / ``ball_speed_min``: multiply the reward by
      ``clamp(|v_xy| / min, max=1)`` so near-zero speeds do not yield a full
      score from ``exp(0)==1``.
    - ``require_contact``: multiply by 1 only when ball contact force exceeds
      ``contact_force_threshold`` (same scale as dribbling touch rewards).

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
        fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
        gate = gate * (fmag > contact_force_threshold).to(torch.float32)

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
) -> torch.Tensor:
    """Reward keeping the ball inside a longitudinal safe-zone in front of the robot.

    The ball position is projected into the robot's local frame:
    - x_local in [near_dist, far_dist] → reward = 1.0
    - Outside that range → exponential decay with ``penalty_std``

    Lateral deviation (|y_local|) is also penalised with the same decay to
    encourage straight-ahead dribbling.

    If ``pelvis_speed_min > 0``, the reward is multiplied by
    ``clamp(|v_pelvis_xy| / pelvis_speed_min, max=1)`` so standing still in the
    safe zone is not a local optimum.

    Returns a value in [0, 1] per environment.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    # Ball position in world frame
    ball_pos_w = soccer_ball.data.root_pos_w  # (N, 3)
    # Robot pelvis position and orientation in world frame
    pelvis_pos_w = command.robot_pelvis_pos_w  # (N, 3)
    pelvis_quat_w = command.robot_pelvis_quat_w  # (N, 4)

    # Transform ball position into the robot's local frame
    delta_w = ball_pos_w - pelvis_pos_w  # (N, 3)
    delta_local = quat_apply(quat_inv(pelvis_quat_w), delta_w)  # (N, 3)

    x_local = delta_local[:, 0]  # forward axis
    y_local = delta_local[:, 1]  # lateral axis

    # Longitudinal error: distance to the nearest edge of the safe zone
    x_error = torch.where(
        x_local < near_dist,
        near_dist - x_local,    # too close → positive error
        torch.where(
            x_local > far_dist,
            x_local - far_dist,  # too far → positive error
            torch.zeros_like(x_local),  # inside safe zone → zero error
        ),
    )

    # Lateral error: absolute lateral offset
    y_error = torch.abs(y_local)

    total_error = x_error ** 2 + y_error ** 2
    proximity_reward = torch.exp(-total_error / (penalty_std ** 2))

    if pelvis_speed_min > 0.0:
        pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
        pelvis_sp = torch.norm(pelvis_vel_xy, dim=-1)
        proximity_reward = proximity_reward * torch.clamp(pelvis_sp / pelvis_speed_min, max=1.0)

    return proximity_reward


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
    force_mag = torch.norm(force_vec, dim=-1)
    has_contact = force_mag > 1.0  # minimal threshold to filter noise
    contact_force_mag = force_mag

    if not torch.any(has_contact):
        return has_contact, contact_force_mag, closest_body_idx

    # For envs with contact, find which robot body is closest to ball
    robot = env.scene[all_body_cfg.name]
    soccer_ball = env.scene["soccer_ball"]

    # Resolve body indices (cached)
    cache_name = "_dribbling_body_indices_cache"
    body_indices = getattr(env, cache_name, None)
    if body_indices is None:
        body_indices = torch.as_tensor(
            robot.find_bodies(all_body_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long, device=device,
        )
        setattr(env, cache_name, body_indices)

    # Contact envs only
    contact_env_ids = torch.nonzero(has_contact, as_tuple=False).squeeze(-1)
    body_pos = robot.data.body_pos_w[contact_env_ids][:, body_indices]  # (M, B, 3)
    ball_pos = soccer_ball.data.root_pos_w[contact_env_ids]  # (M, 3)

    dist = torch.norm(body_pos - ball_pos.unsqueeze(1), dim=-1)  # (M, B)
    closest = torch.argmin(dist, dim=-1)  # (M,) — index into body_names
    closest_body_idx[contact_env_ids] = closest

    return has_contact, contact_force_mag, closest_body_idx


def _build_side_map(body_names: list[str], device: torch.device) -> torch.Tensor:
    """Map each body name to a side: 0=left, 1=right, -1=other.

    This matches the convention in ``MotionCommand.kick_leg``:
      - 0 → left
      - 1 → right
      - -1 → unknown / no label
    """
    sides = []
    for name in body_names:
        lower = name.lower()
        if "left" in lower:
            sides.append(0)
        elif "right" in lower:
            sides.append(1)
        else:
            sides.append(-1)
    return torch.tensor(sides, dtype=torch.int8, device=device)


def _is_legal_foot(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    closest_body_idx: torch.Tensor,
    all_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Check if the closest body matches the motion's designated dribble foot.

    Uses ``command.kick_leg`` (0=left, 1=right from motion file labels) to
    determine which foot is the legal dribble foot for each environment.

    Returns:
        is_legal: (N,) bool — True if closest body is on the correct side.
    """
    cache_name = "_dribbling_side_map"
    side_map = getattr(env, cache_name, None)
    if side_map is None:
        side_map = _build_side_map(all_body_cfg.body_names, env.device)
        setattr(env, cache_name, side_map)

    # Which side did the closest body belong to? (0=left, 1=right, -1=other)
    contact_side = side_map[closest_body_idx]  # (N,) int8

    # Which side does the motion require? (0=left, 1=right, -1=unknown)
    expected_side = command.kick_leg  # (N,) int8

    # Legal if the contact side matches the expected side
    # If expected is -1 (unknown), we allow either foot
    is_legal = (contact_side == expected_side) | (expected_side < 0)

    return is_legal


# ---------------------------------------------------------------------------
# 3a) Legal Foot Gentle Touch — small positive reward
# ---------------------------------------------------------------------------

def dribbling_legal_foot_touch(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    all_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Positive reward when the LEGAL foot (from motion label) gently touches the ball.

    Which foot is "legal" is determined per-env by ``command.kick_leg``:
      - motion file ``*_left.npz``  → left foot legal
      - motion file ``*_right.npz`` → right foot legal

    Returns 1.0 for a valid gentle legal-foot touch, 0.0 otherwise.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_legal = _is_legal_foot(env, command, closest_idx, all_body_cfg)
    gentle = force_mag <= force_threshold
    reward = (has_contact & is_legal & gentle).to(torch.float32)

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
) -> torch.Tensor:
    """Moderate EMA-smoothed penalty when the LEGAL foot hits too hard.

    Only penalises legal-foot contacts exceeding ``force_threshold``.
    Non-legal-foot contacts are handled by ``dribbling_undesired_contact_penalty``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_legal = _is_legal_foot(env, command, closest_idx, all_body_cfg)

    # Only consider legal-foot hard contacts for this penalty
    legal_hard_force = torch.where(
        has_contact & is_legal,
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
) -> torch.Tensor:
    """Severe instant penalty when a NON-LEGAL body touches the ball.

    Which foot is "legal" is determined per-env by ``command.kick_leg``.
    If the motion specifies right foot, then left foot/knees/hands = penalty.

    Returns 1.0 for each env where an illegal body touched the ball.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_legal = _is_legal_foot(env, command, closest_idx, all_body_cfg)

    # Penalty = 1 when contact exists but it's NOT the legal foot
    penalty = (has_contact & ~is_legal).to(torch.float32)

    return penalty



