"""Anchor-based observation functions for dribbling / task-frame control.

Ball-relative observations use the **task frame** (env-local axes parallel to
world: +X forward, +Y lateral, +Z up), matching velocity rewards and spawn logic.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_inv, yaw_quat

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.task_frame import task_forward_offset, task_lateral_offset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def anchor_ball_polar(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Ball offset from pelvis in task-frame polar coordinates.

    Returns ``(distance, cos_heading, sin_heading)`` on the ground plane where
    heading is the angle from task +X to the pelvis→ball vector.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    pelvis_pos_w = command.robot_pelvis_pos_w

    dx = task_forward_offset(ball_pos_w, pelvis_pos_w)
    dy = task_lateral_offset(ball_pos_w, pelvis_pos_w)
    dist = torch.norm(torch.stack([dx, dy], dim=-1), dim=-1).clamp(min=1e-4)
    cos_heading = dx / dist
    sin_heading = dy / dist

    return torch.stack([dist, cos_heading, sin_heading], dim=-1)


def anchor_ball_pelvis_local_polar(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Ball offset in the current pelvis-yaw local polar frame.

    The output shape intentionally matches ``anchor_ball_polar`` so local
    curriculum stages preserve the 163-D actor contract while eliminating the
    non-deployable task/world-frame direction.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    offset_b = quat_apply(
        quat_inv(yaw_quat(command.robot_pelvis_quat_w)),
        soccer_ball.data.root_pos_w[:, :3] - command.robot_pelvis_pos_w,
    )
    distance = torch.norm(offset_b[:, :2], dim=-1).clamp(min=1.0e-4)
    return torch.stack((distance, offset_b[:, 0] / distance, offset_b[:, 1] / distance), dim=-1)


def anchor_ball_local(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Ball position relative to pelvis in task-frame Cartesian coordinates (x, y, z)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    pelvis_pos_w = command.robot_pelvis_pos_w

    delta = ball_pos_w - pelvis_pos_w
    return delta
