"""Task frame utilities (env-local == world-parallel axes).

All dribbling / stage-1 locomotion terms use the same convention:
  +X forward (field direction), +Y lateral, +Z up.
"""

from __future__ import annotations

import torch


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


def task_lateral_speed_penalty(
    lin_vel_w: torch.Tensor,
    lateral_deadzone: float = 0.12,
    lateral_scale: float = 0.4,
) -> torch.Tensor:
    """Soft squared penalty for |v_y| above ``lateral_deadzone``."""
    excess = torch.clamp(torch.abs(task_lateral_speed(lin_vel_w)) - lateral_deadzone, min=0.0)
    return torch.square(excess / max(lateral_scale, 1e-6))


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
