"""Task frame utilities (env-local == world-parallel axes).

All dribbling / stage-1 locomotion terms use the same convention:
  +X forward (field direction), +Y lateral, +Z up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


@dataclass
class DribblePhaseBundle:
    """Per-env dribble phase masks (mutually exclusive display priority: touch > seek_touch > chase > approach)."""

    touch: torch.Tensor
    seek_touch: torch.Tensor
    chase: torch.Tensor
    approach: torch.Tensor
    close_approach: torch.Tensor
    transition_ready: torch.Tensor
    steps_in_close_approach: torch.Tensor


def update_approach_touch_transition_steps(
    env: ManagerBasedRLEnv,
    close_approach: torch.Tensor,
    has_contact: torch.Tensor,
    *,
    buf_name: str = "_dribble_steps_in_close_approach",
) -> torch.Tensor:
    """Count consecutive steps in ``close_approach`` without ``has_contact`` (resets on touch or leaving close zone)."""
    step_buf = getattr(env, "episode_length_buf", None)
    reset_ep = step_buf == 0 if step_buf is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    cnt = getattr(env, buf_name, None)
    if cnt is None or cnt.shape[0] != env.num_envs:
        cnt = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    leave_or_touch = has_contact | (~close_approach)
    cnt = torch.where(
        reset_ep | leave_or_touch,
        torch.zeros_like(cnt),
        torch.where(close_approach, cnt + 1, cnt),
    )
    setattr(env, buf_name, cnt)
    return cnt


def compute_dribble_phase_bundle(
    has_contact: torch.Tensor,
    recent_contact: torch.Tensor,
    x_ahead: torch.Tensor,
    dist_xy: torch.Tensor,
    ball_speed_xy: torch.Tensor,
    *,
    steps_in_close_approach: torch.Tensor | None = None,
    min_ankle_ball_dist: torch.Tensor | None = None,
    chase_min_ahead: float = 0.35,
    approach_max_dist: float = 0.55,
    approach_ball_speed_max: float = 0.35,
    approach_min_x_ahead: float = 0.10,
    approach_max_x_ahead: float = 0.85,
    close_max_dist: float = 0.48,
    close_x_min: float = 0.18,
    close_x_max: float = 0.62,
    seek_touch_min_steps: int = 2,
    seek_touch_commit_dist: float = 0.24,
) -> DribblePhaseBundle:
    """Full phase logic including explicit **approach → touch** via ``seek_touch``.

    Pipeline:
      chase (ball far ahead) → approach (ball in corridor, not close) →
      close_approach (near, no sensor touch) → **seek_touch** (transition committed) →
      touch (``has_contact`` or ``recent_contact``).
    """
    touch_phase = has_contact | (recent_contact > 0.5)
    chase_phase = (~touch_phase) & (x_ahead >= chase_min_ahead) & (dist_xy > approach_max_dist)
    ball_in_corridor = (x_ahead >= approach_min_x_ahead) & (x_ahead <= approach_max_x_ahead)
    approach_wide = (
        (~touch_phase)
        & (~chase_phase)
        & ball_in_corridor
        & (
            (dist_xy <= approach_max_dist)
            | (
                (ball_speed_xy <= approach_ball_speed_max)
                & (dist_xy <= approach_max_dist + 0.20)
            )
        )
    )
    close_approach = compute_close_approach_mask(
        approach_wide,
        has_contact,
        x_ahead,
        dist_xy,
        close_max_dist=close_max_dist,
        close_x_min=close_x_min,
        close_x_max=close_x_max,
    )

    if steps_in_close_approach is None:
        steps_in_close_approach = torch.zeros_like(dist_xy, dtype=torch.int32)
    if min_ankle_ball_dist is None:
        min_ankle_ball_dist = torch.full_like(dist_xy, 999.0)

    transition_ready = close_approach & (~touch_phase) & (
        (steps_in_close_approach >= int(seek_touch_min_steps))
        | (min_ankle_ball_dist <= seek_touch_commit_dist)
    )
    seek_touch_phase = transition_ready
    approach_phase = approach_wide & (~close_approach) & (~touch_phase) & (~chase_phase)

    return DribblePhaseBundle(
        touch=touch_phase,
        seek_touch=seek_touch_phase,
        chase=chase_phase,
        approach=approach_phase,
        close_approach=close_approach,
        transition_ready=transition_ready,
        steps_in_close_approach=steps_in_close_approach,
    )


def resolve_dribble_phase_label(bundle: DribblePhaseBundle, env_index: int = 0) -> str:
    """Human-readable phase for HUD (priority: touch > seek_touch > chase > approach > idle)."""
    i = env_index
    if bool(bundle.touch[i].item()):
        return "touch"
    if bool(bundle.seek_touch[i].item()):
        return "seek_touch"
    if bool(bundle.chase[i].item()):
        return "chase"
    if bool(bundle.approach[i].item()):
        return "approach"
    return "idle"


def compute_dribble_phase_masks(
    has_contact: torch.Tensor,
    recent_contact: torch.Tensor,
    x_ahead: torch.Tensor,
    dist_xy: torch.Tensor,
    ball_speed_xy: torch.Tensor,
    *,
    steps_in_close_approach: torch.Tensor | None = None,
    min_ankle_ball_dist: torch.Tensor | None = None,
    chase_min_ahead: float = 0.35,
    approach_max_dist: float = 0.55,
    approach_ball_speed_max: float = 0.35,
    approach_min_x_ahead: float = 0.10,
    approach_max_x_ahead: float = 0.85,
    close_max_dist: float = 0.48,
    close_x_min: float = 0.18,
    close_x_max: float = 0.62,
    seek_touch_min_steps: int = 2,
    seek_touch_commit_dist: float = 0.24,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(touch, seek_touch, chase, approach)`` phase masks."""
    b = compute_dribble_phase_bundle(
        has_contact,
        recent_contact,
        x_ahead,
        dist_xy,
        ball_speed_xy,
        steps_in_close_approach=steps_in_close_approach,
        min_ankle_ball_dist=min_ankle_ball_dist,
        chase_min_ahead=chase_min_ahead,
        approach_max_dist=approach_max_dist,
        approach_ball_speed_max=approach_ball_speed_max,
        approach_min_x_ahead=approach_min_x_ahead,
        approach_max_x_ahead=approach_max_x_ahead,
        close_max_dist=close_max_dist,
        close_x_min=close_x_min,
        close_x_max=close_x_max,
        seek_touch_min_steps=seek_touch_min_steps,
        seek_touch_commit_dist=seek_touch_commit_dist,
    )
    return b.touch, b.seek_touch, b.chase, b.approach


def compute_close_approach_mask(
    approach_phase: torch.Tensor,
    has_contact: torch.Tensor,
    x_ahead: torch.Tensor,
    dist_xy: torch.Tensor,
    *,
    close_max_dist: float = 0.48,
    close_x_min: float = 0.18,
    close_x_max: float = 0.62,
) -> torch.Tensor:
    """Near the ball in approach but not yet touching — use touch-speed / foot-shaping."""
    return (
        approach_phase
        & (~has_contact)
        & (dist_xy <= close_max_dist)
        & (x_ahead >= close_x_min)
        & (x_ahead <= close_x_max)
    )


def compute_dribble_phase_target_speed(
    has_contact: torch.Tensor,
    recent_contact: torch.Tensor,
    x_ahead: torch.Tensor,
    dist_xy: torch.Tensor,
    ball_speed_xy: torch.Tensor,
    *,
    steps_in_close_approach: torch.Tensor | None = None,
    min_ankle_ball_dist: torch.Tensor | None = None,
    v_touch: float = 0.20,
    v_chase: float = 0.75,
    v_approach: float = 0.30,
    chase_min_ahead: float = 0.35,
    approach_max_dist: float = 0.55,
    approach_ball_speed_max: float = 0.35,
    close_max_dist: float = 0.48,
    close_x_min: float = 0.18,
    close_x_max: float = 0.62,
    seek_touch_min_steps: int = 2,
    seek_touch_commit_dist: float = 0.24,
) -> torch.Tensor:
    """Pelvis forward speed target: chase → approach → seek_touch → touch."""
    bundle = compute_dribble_phase_bundle(
        has_contact,
        recent_contact,
        x_ahead,
        dist_xy,
        ball_speed_xy,
        steps_in_close_approach=steps_in_close_approach,
        min_ankle_ball_dist=min_ankle_ball_dist,
        chase_min_ahead=chase_min_ahead,
        approach_max_dist=approach_max_dist,
        approach_ball_speed_max=approach_ball_speed_max,
        close_max_dist=close_max_dist,
        close_x_min=close_x_min,
        close_x_max=close_x_max,
        seek_touch_min_steps=seek_touch_min_steps,
        seek_touch_commit_dist=seek_touch_commit_dist,
    )

    target = torch.full_like(x_ahead, v_approach)
    target = torch.where(bundle.approach, torch.full_like(target, v_approach), target)
    target = torch.where(bundle.close_approach, torch.full_like(target, v_touch), target)
    target = torch.where(bundle.seek_touch, torch.full_like(target, v_touch), target)
    target = torch.where(bundle.chase, torch.full_like(target, v_chase), target)
    target = torch.where(bundle.touch, torch.full_like(target, v_touch), target)
    return target


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
