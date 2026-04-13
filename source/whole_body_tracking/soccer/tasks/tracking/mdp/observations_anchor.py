"""Anchor-based observation functions for the decoupled kick architecture.

These functions provide egocentric (self-centered) observations that do NOT
depend on absolute world coordinates.  They are designed to be used alongside
the existing ``observations.py`` functions without modifying them.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_inv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def anchor_ball_polar(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Ball position relative to the robot pelvis in polar-style coordinates.

    Returns a 3-D vector per env: ``(distance, cos_heading, sin_heading)`` where
    heading is the angle between the pelvis forward direction and the
    pelvis-to-ball vector projected onto the ground plane.

    This observation is **egocentric** — it is invariant to the absolute
    position/orientation of the robot on the field.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]          # [N, 3]
    pelvis_pos_w = command.robot_pelvis_pos_w                 # [N, 3]
    pelvis_quat_w = command.robot_pelvis_quat_w               # [N, 4]

    # Delta in world frame, then rotate into pelvis-local frame.
    delta_w = ball_pos_w - pelvis_pos_w                       # [N, 3]
    delta_local = quat_apply(quat_inv(pelvis_quat_w), delta_w)  # [N, 3]

    # Polar decomposition on XY plane.
    dx = delta_local[:, 0]
    dy = delta_local[:, 1]
    dist = torch.norm(delta_local[:, :2], dim=-1).clamp(min=1e-4)
    cos_heading = dx / dist
    sin_heading = dy / dist

    return torch.stack([dist, cos_heading, sin_heading], dim=-1)  # [N, 3]


def anchor_ball_local(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Ball position relative to pelvis in local Cartesian coordinates (x, y, z).

    Same as ``constant_target_point_pos`` but explicitly decoupled from the
    original observation module for clarity.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    pelvis_pos_w = command.robot_pelvis_pos_w
    pelvis_quat_w = command.robot_pelvis_quat_w

    delta_w = ball_pos_w - pelvis_pos_w
    return quat_apply(quat_inv(pelvis_quat_w), delta_w)  # [N, 3]
