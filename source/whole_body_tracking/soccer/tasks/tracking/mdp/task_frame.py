"""Task frame utilities (env-local == world-parallel axes).

All dribbling / stage-1 locomotion terms use the same convention:
  +X forward (field direction), +Y lateral, +Z up.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_inv


def task_delta_xy(pos_w: torch.Tensor, ref_pos_w: torch.Tensor) -> torch.Tensor:
    """XY offset from ``ref`` to ``pos`` in task frame."""
    return pos_w[..., :2] - ref_pos_w[..., :2]


def task_forward_offset(pos_w: torch.Tensor, ref_pos_w: torch.Tensor) -> torch.Tensor:
    """Signed +X offset (ball ahead of ref when positive)."""
    return pos_w[..., 0] - ref_pos_w[..., 0]


def task_lateral_offset(pos_w: torch.Tensor, ref_pos_w: torch.Tensor) -> torch.Tensor:
    """Signed +Y offset."""
    return pos_w[..., 1] - ref_pos_w[..., 1]


def task_forward_speed(lin_vel_w: torch.Tensor, *, clamp_forward: bool = True) -> torch.Tensor:
    """+X linear speed (optionally clamped to non-negative)."""
    vx = lin_vel_w[..., 0]
    if clamp_forward:
        return torch.clamp(vx, min=0.0)
    return vx


def task_lateral_speed(lin_vel_w: torch.Tensor) -> torch.Tensor:
    """+Y linear speed."""
    return lin_vel_w[..., 1]


def task_velocity_forward_dominance(lin_vel_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Share of XY speed along task +X in ``[0, 1]`` (0 = pure lateral / backward)."""
    vx = lin_vel_w[..., 0]
    vy = lin_vel_w[..., 1]
    speed_xy = torch.sqrt(vx * vx + vy * vy + eps)
    vx_pos = torch.clamp(vx, min=0.0)
    return (vx_pos / speed_xy).clamp(0.0, 1.0)


def task_pelvis_heading_cos_world_x(pelvis_quat_w: torch.Tensor) -> torch.Tensor:
    """Cosine between pelvis forward (XY) and task +X."""
    num_envs = pelvis_quat_w.shape[0]
    ref_forward = torch.zeros(num_envs, 3, device=pelvis_quat_w.device, dtype=pelvis_quat_w.dtype)
    ref_forward[:, 0] = 1.0
    pelvis_forward = quat_apply(pelvis_quat_w, ref_forward)
    forward_xy = pelvis_forward[:, :2]
    norm = torch.norm(forward_xy, dim=-1).clamp(min=1e-6)
    return (forward_xy[:, 0] / norm).clamp(-1.0, 1.0)


def task_lateral_speed_penalty(
    lin_vel_w: torch.Tensor,
    lateral_deadzone: float = 0.12,
    lateral_scale: float = 0.4,
) -> torch.Tensor:
    """Soft squared penalty for |v_y| above ``lateral_deadzone``."""
    excess = torch.clamp(torch.abs(task_lateral_speed(lin_vel_w)) - lateral_deadzone, min=0.0)
    return torch.square(excess / max(lateral_scale, 1e-6))


def task_combined_lateral_speed_penalty(
    pelvis_lin_vel_w: torch.Tensor,
    pelvis_quat_w: torch.Tensor,
    lateral_deadzone: float = 0.06,
    lateral_scale: float = 0.28,
) -> torch.Tensor:
    """Penalise lateral drift in both task (+Y) and pelvis-local (+Y) frames."""
    pelvis_lin_vel_local = quat_apply(quat_inv(pelvis_quat_w), pelvis_lin_vel_w)
    return task_lateral_speed_penalty(pelvis_lin_vel_w, lateral_deadzone, lateral_scale) + task_lateral_speed_penalty(
        pelvis_lin_vel_local, lateral_deadzone, lateral_scale
    )


def forward_dominance_gate(dominance: torch.Tensor, min_dominance: float) -> torch.Tensor:
    """Linear ramp from 0 at ``min_dominance`` to 1 at full forward dominance."""
    if min_dominance <= 0.0:
        return torch.ones_like(dominance)
    return torch.clamp((dominance - min_dominance) / (1.0 - min_dominance + 1e-6), min=0.0, max=1.0)


def spawn_ball_ahead_env_local(
    anchor_pos: torch.Tensor,
    distance: float,
    lateral_offset: float = 0.0,
    height: float = 0.11,
) -> torch.Tensor:
    """Place ball ``distance`` along task +X from ``anchor_pos`` (env-local)."""
    ball_pos = anchor_pos.clone()
    ball_pos[..., 0] = ball_pos[..., 0] + distance
    ball_pos[..., 1] = ball_pos[..., 1] + lateral_offset
    ball_pos[..., 2] = height
    return ball_pos
