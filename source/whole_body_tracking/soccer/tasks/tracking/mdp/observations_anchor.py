"""Anchor-based observation functions for dribbling / task-frame control.

Ball-relative observations use the **task frame** (env-local axes parallel to
world: +X forward, +Y lateral, +Z up), matching velocity rewards and spawn logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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


def anchor_ball_velocity_polar_command(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    stationary_speed_threshold: float = 1.0e-4,
) -> torch.Tensor:
    """Ball XY velocity as ``[speed, cos(delta), sin(delta)]`` relative to command heading.

    This is the simulation-side velocity feedback for the learned speed loop.
    ``delta`` is the signed angle from the active locomotion direction to the
    ball velocity direction.  A stationary ball returns ``[0, 0, 0]`` so its
    undefined velocity heading cannot look like a valid forward direction.

    Stage 1 and Stage 2 expose this exact same three-dimensional term.  A
    future vision deployment can replace the simulator velocity with a tracked
    estimate without changing the policy interface.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    if hasattr(command, "locomotion_lin_vel_command_w"):
        command_vel_xy = command.locomotion_lin_vel_command_w()[:, :2]
    else:
        command_vel_xy = command.anchor_lin_vel_w[:, :2]
    command_speed = torch.norm(command_vel_xy, dim=-1)

    heading = getattr(command, "locomotion_cmd_heading", None)
    if isinstance(heading, torch.Tensor):
        fallback_direction = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    else:
        fallback_direction = torch.zeros_like(command_vel_xy)
        fallback_direction[:, 0] = 1.0
    command_direction = torch.where(
        (command_speed > stationary_speed_threshold).unsqueeze(-1),
        command_vel_xy / command_speed.unsqueeze(-1).clamp(min=stationary_speed_threshold),
        fallback_direction,
    )

    ball_vel_xy = soccer_ball.data.root_lin_vel_w[:, :2]
    ball_speed = torch.norm(ball_vel_xy, dim=-1)
    safe_speed = ball_speed.clamp(min=stationary_speed_threshold)
    cos_delta = torch.sum(ball_vel_xy * command_direction, dim=-1) / safe_speed
    sin_delta = (
        command_direction[:, 0] * ball_vel_xy[:, 1]
        - command_direction[:, 1] * ball_vel_xy[:, 0]
    ) / safe_speed
    moving = ball_speed > stationary_speed_threshold
    cos_delta = torch.where(
        moving, cos_delta.clamp(min=-1.0, max=1.0), torch.zeros_like(cos_delta)
    )
    sin_delta = torch.where(
        moving, sin_delta.clamp(min=-1.0, max=1.0), torch.zeros_like(sin_delta)
    )
    return torch.stack((ball_speed, cos_delta, sin_delta), dim=-1)


def zero_anchor_ball_velocity_polar_command(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Return a shape-preserving zero replacement for the ball-velocity input.

    The command argument is retained so this function can replace
    :func:`anchor_ball_velocity_polar_command` without changing an observation
    term's configuration or the actor/critic input dimensions.
    """
    del command_name
    return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
