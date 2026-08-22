from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_error_magnitude

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand, locomotion_task_state_mask
from soccer.tasks.tracking.mdp.kick_detection import KickContactTracker
from soccer.tasks.tracking.mdp.task_frame import mimic_anchor_yaw_delta_quat


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]

def action_rate_l2_clip(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward.clamp(max=100.0)


def effective_action_rate_l2_clip(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Penalize effective command changes without charging arm-reference motion.

    The recurrent observation still receives the final absolute normalized
    joint target. For reference-relative arm joints, however, smoothness is a
    property of the policy correction, not of the moving motion reference.
    """
    action_term = env.action_manager.get_term(action_name)
    current = getattr(action_term, "effective_raw_actions", action_term.raw_actions)
    previous = getattr(action_term, "prev_effective_raw_actions", env.action_manager.prev_action)
    delta = current - previous
    upper_action_ids = getattr(action_term, "_upper_action_ids", None)
    residual_action_ids = getattr(action_term, "_residual_action_ids", None)
    current_upper = getattr(action_term, "effective_upper_residual_actions", None)
    previous_upper = getattr(action_term, "prev_effective_upper_residual_actions", None)
    if (
        isinstance(upper_action_ids, torch.Tensor)
        and isinstance(residual_action_ids, torch.Tensor)
        and isinstance(current_upper, torch.Tensor)
        and isinstance(previous_upper, torch.Tensor)
    ):
        delta = delta.clone()
        delta[:, upper_action_ids] = 0.0
        delta[:, residual_action_ids] = current_upper - previous_upper
    return torch.sum(torch.square(delta), dim=1).clamp(max=100.0)


def upper_body_pre_squash_action_l2(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Penalize the eight shoulder/elbow Gaussian variables before ``tanh``.

    A post-tanh cost becomes nearly flat at the boundary and cannot reliably
    recover a saturated Gaussian mean. This unbounded quadratic acts on the
    real pre-squash variables; reference-only wrists are excluded.
    """
    action_term = env.action_manager.get_term(action_name)
    pre_squash = getattr(action_term, "upper_residual_pre_squash", None)
    residual_local_ids = getattr(action_term, "_residual_local_ids", None)
    if not isinstance(pre_squash, torch.Tensor) or not isinstance(
        residual_local_ids, torch.Tensor
    ):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return torch.mean(torch.square(pre_squash[:, residual_local_ids]), dim=1)


def _pelvis_yaw_w(command: MotionCommand, pelvis_body_name: str = "pelvis") -> torch.Tensor:
    """Return pelvis yaw in the world frame."""
    robot = command.robot
    pelvis_id = robot.body_names.index(pelvis_body_name)
    pelvis_quat_w = robot.data.body_quat_w[:, pelvis_id]
    return torch.atan2(
        2.0 * (pelvis_quat_w[:, 0] * pelvis_quat_w[:, 3] + pelvis_quat_w[:, 1] * pelvis_quat_w[:, 2]),
        1.0 - 2.0 * (torch.square(pelvis_quat_w[:, 2]) + torch.square(pelvis_quat_w[:, 3])),
    )


def _wrapped_heading_error(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Return ``target - current`` wrapped to ``[-pi, pi]``."""
    return torch.atan2(torch.sin(target - current), torch.cos(target - current))


def locomotion_heading_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    pelvis_body_name: str = "pelvis",
    std: float = 0.35,
    min_command_speed: float = 0.15,
) -> torch.Tensor:
    """Reward pelvis yaw tracking of the *effective* locomotion command.

    The command smoother exposes the heading it currently asks the policy to
    execute.  Tracking that value, rather than the final endpoint, makes the
    reward feasible throughout a large direction reversal and gives turning a
    direct training signal instead of relying on velocity tracking alone.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    pelvis_yaw = _pelvis_yaw_w(command, pelvis_body_name)
    heading_error = _wrapped_heading_error(command.locomotion_cmd_heading, pelvis_yaw)
    reward = torch.exp(-torch.square(heading_error) / max(float(std), 1.0e-6) ** 2)
    speed_gate = torch.clamp(
        command.locomotion_cmd_speed / max(float(min_command_speed), 1.0e-6), min=0.0, max=1.0
    )
    return reward * speed_gate


def _get_kick_tracker(command: MotionCommand) -> KickContactTracker:
    tracker = getattr(command, "kick_contact_tracker", None)
    if tracker is None:
        raise RuntimeError("MotionCommand is missing kick_contact_tracker; ensure command setup is up to date.")
    return tracker


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    return reward * locomotion_task_state_mask(command, active_task_states).to(reward.dtype)

def motion_relative_foot_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    foot_body_names: list[str] | None = None,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, foot_body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    return reward * locomotion_task_state_mask(command, active_task_states).to(reward.dtype)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def _motion_anchor_yaw_delta_quat(command: MotionCommand) -> torch.Tensor:
    """Yaw delta for mimic vel terms (must match command ``_update_command``)."""
    return mimic_anchor_yaw_delta_quat(
        command.anchor_quat_w,
        command.robot_anchor_quat_w,
        align_task_frame=bool(getattr(command.cfg, "mimic_align_task_frame", False)),
    )


def _locomotion_lin_vel_command_w(command: MotionCommand) -> torch.Tensor:
    if hasattr(command, "locomotion_lin_vel_command_w"):
        return command.locomotion_lin_vel_command_w()
    delta_ori_w = _motion_anchor_yaw_delta_quat(command)
    return quat_apply(delta_ori_w, command.anchor_lin_vel_w)


def motion_anchor_pos_z_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Soft reward for matching demo anchor height (Z only)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    return torch.exp(-error / max(std, 1e-6) ** 2)


def motion_anchor_lin_vel_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Match active locomotion linear-velocity command (reference or manual)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_vel = _locomotion_lin_vel_command_w(command)
    error = torch.sum(torch.square(ref_vel - command.robot_anchor_lin_vel_w), dim=-1)
    return torch.exp(-error / max(std, 1e-6) ** 2)


def motion_anchor_ang_vel_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    heading_error_gain: float = 2.0,
    max_yaw_rate: float = 1.2,
) -> torch.Tensor:
    """Track a feasible yaw-rate target generated from the heading controller.

    During a smoothed heading transition, the target contains the command
    smoother's feed-forward yaw rate.  A proportional correction based on the
    pelvis-to-effective-heading error prevents the robot from permanently
    lagging behind that moving command.  Once both heading errors vanish, the
    target naturally returns to zero.
    """
    if heading_error_gain < 0.0 or max_yaw_rate <= 0.0:
        raise ValueError("heading_error_gain must be non-negative and max_yaw_rate must be positive.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    pelvis_yaw = _pelvis_yaw_w(command)
    tracking_error = _wrapped_heading_error(command.locomotion_cmd_heading, pelvis_yaw)

    feedforward_yaw_rate = torch.zeros_like(tracking_error)
    target_heading = getattr(command, "locomotion_cmd_target_heading", None)
    heading_rate_limit = max(
        float(getattr(command.cfg, "locomotion_cmd_heading_rate_limit", 0.0)),
        0.0,
    )
    if isinstance(target_heading, torch.Tensor) and heading_rate_limit > 0.0:
        endpoint_error = _wrapped_heading_error(target_heading, command.locomotion_cmd_heading)
        step_dt = max(float(getattr(env, "step_dt", 0.02)), 1.0e-6)
        feedforward_yaw_rate = torch.clamp(
            endpoint_error / step_dt,
            min=-heading_rate_limit,
            max=heading_rate_limit,
        )

    target_yaw_rate = torch.clamp(
        feedforward_yaw_rate + float(heading_error_gain) * tracking_error,
        min=-float(max_yaw_rate),
        max=float(max_yaw_rate),
    )
    error = torch.square(target_yaw_rate - command.robot_anchor_ang_vel_w[:, 2])
    return torch.exp(-error / max(std, 1e-6) ** 2)


def motion_relative_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Linear velocity match with demo velocities yaw-aligned to the robot anchor."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    delta_ori_w = _motion_anchor_yaw_delta_quat(command)
    demo_vel = command.body_lin_vel_w[:, body_indexes]
    ref_vel = quat_apply(delta_ori_w.unsqueeze(1).expand(-1, len(body_indexes), -1), demo_vel)
    error = torch.sum(
        torch.square(ref_vel - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Angular velocity match with demo velocities yaw-aligned to the robot anchor."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    delta_ori_w = _motion_anchor_yaw_delta_quat(command)
    demo_ang = command.body_ang_vel_w[:, body_indexes]
    ref_ang = quat_apply(delta_ori_w.unsqueeze(1).expand(-1, len(body_indexes), -1), demo_ang)
    error = torch.sum(
        torch.square(ref_ang - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def foot_distance(env: ManagerBasedRLEnv, threshold: float, std: float, foot_cfg: SceneEntityCfg | None = None,) -> torch.Tensor:
    """Encourage a minimum separation between both feet to avoid crossing/overlap."""
    if foot_cfg is None:
        raise ValueError("foot_distance requires foot_cfg to identify feet.")
    robot = env.scene[foot_cfg.name]
    left_foot_idx = foot_cfg.body_ids[0]
    right_foot_idx = foot_cfg.body_ids[1]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]  # [num_envs, 3]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]  # [num_envs, 3]
    distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)  # [num_envs]
    reward = torch.where(
        distance >= threshold,
        torch.tensor(1., device=distance.device),
        1.0 * torch.exp(-((distance / threshold - 1)**2) / (std ** 2))
    )
    return reward
