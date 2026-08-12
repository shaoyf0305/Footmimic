from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def effective_joint_action(env: ManagerBasedEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Return the normalized joint command that is actually sent by an action term."""
    action_term = env.action_manager.get_term(action_name)
    effective = getattr(action_term, "effective_raw_actions", None)
    if effective is not None:
        return effective
    return action_term.raw_actions


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_ang_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.anchor_ang_vel_w.view(env.num_envs, -1)


def motion_locomotion_polar_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Active locomotion command as ``[speed, cos(heading), sin(heading)]`` (task +X heading).

    Speed in m/s; heading radians from +X (0=forward, +pi/2=+Y). Independent of demo root vel
    when ``locomotion_command_mode`` is ``resampled`` or ``manual``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    mode = getattr(command, "locomotion_command_mode", "reference")
    if mode in {"resampled", "manual"}:
        speed = command.locomotion_cmd_speed
        heading = command.locomotion_cmd_heading
    else:
        lin = (
            command.locomotion_lin_vel_command_w()
            if hasattr(command, "locomotion_lin_vel_command_w")
            else command.anchor_lin_vel_w
        )
        speed = torch.norm(lin[:, :2], dim=-1)
        heading = torch.atan2(lin[:, 1], lin[:, 0])
    return torch.stack([speed, torch.cos(heading), torch.sin(heading)], dim=-1)
