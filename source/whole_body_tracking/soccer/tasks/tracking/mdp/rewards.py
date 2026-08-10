from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, quat_apply, quat_inv, quat_apply_inverse, yaw_quat

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand, locomotion_task_state_mask
from soccer.tasks.tracking.mdp.observations import get_target_point_world
from soccer.tasks.tracking.mdp.kick_detection import KickContactTracker
from soccer.tasks.tracking.mdp.task_frame import (
    forward_dominance_gate,
    mimic_anchor_yaw_delta_quat,
    task_combined_lateral_speed_penalty,
    task_forward_speed,
    task_pelvis_heading_cos_world_x,
    task_velocity_forward_dominance,
)


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
    """Penalize changes in the normalized command that the action term actually executes."""
    action_term = env.action_manager.get_term(action_name)
    current = getattr(action_term, "effective_raw_actions", action_term.raw_actions)
    previous = getattr(action_term, "prev_effective_raw_actions", env.action_manager.prev_action)
    return torch.sum(torch.square(current - previous), dim=1).clamp(max=100.0)


def upper_body_reference_overflow_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Penalize upper-body targets that exceed a reference-action safety envelope.

    The manifold action term exposes the overflow *before* it clamps a target
    to ``q_ref +/- margin``.  Penalizing that signal prevents a policy from
    hiding permanently in the downstream PCA saturation region.  The smooth
    L1-like form stays well scaled when fine-tuning a checkpoint that already
    emits very large upper-body targets.
    """
    action_term = env.action_manager.get_term(action_name)
    overflow = getattr(action_term, "manifold_reference_overflow", None)
    margin = getattr(getattr(action_term, "cfg", None), "reference_target_margin", None)
    if not isinstance(overflow, torch.Tensor) or margin is None:
        return torch.zeros(env.num_envs, device=env.device)

    normalized_overflow = overflow / max(float(margin), 1.0e-6)
    return torch.mean(torch.sqrt(1.0 + torch.square(normalized_overflow)) - 1.0, dim=1)


def upper_body_manifold_nullspace_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    scale: float = 0.10,
) -> torch.Tensor:
    """Penalize arm targets that PCA will discard, without changing the action.

    The value is measured after the v3.9 reference envelope and before PCA
    projection.  Unlike an additional clip or filter, this leaves the executed
    control path unchanged and gives PPO a continuous reason not to spend its
    14-D arm command in the manifold's null space.
    """
    action_term = env.action_manager.get_term(action_name)
    residual = getattr(action_term, "manifold_nullspace_residual", None)
    if not isinstance(residual, torch.Tensor):
        return torch.zeros(env.num_envs, device=env.device)
    normalized = residual / max(float(scale), 1.0e-6)
    return torch.sqrt(1.0 + torch.square(normalized)) - 1.0


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    target_speed: float = 0.8,
    std: float = 0.4,
    command_name: str = "motion",
    velocity_frame: str = "world",
    min_forward_dominance: float = 0.55,
) -> torch.Tensor:
    """Reward pelvis speed along the task forward axis matching ``target_speed``.

    Default ``velocity_frame="world"`` uses world +X (non-negative). When
    ``min_forward_dominance > 0``, the reward is gated so pure lateral / crab
    motion (|v_y| >> v_x) receives little or no score.
    """
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

    error = (forward_speed - target_speed) ** 2
    reward = torch.exp(-error / max(std, 1e-6) ** 2)
    return reward * forward_dominance_gate(dominance, min_forward_dominance)


def lateral_velocity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    velocity_frame: str = "world",
    lateral_deadzone: float = 0.06,
    lateral_scale: float = 0.28,
) -> torch.Tensor:
    """Soft penalty for pelvis lateral speed (task +Y and, in world mode, body +Y).

    Apply with a negative weight. Dual-frame penalty suppresses crab-walking while
    a small deadzone still allows minor dribble adjustments.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    pelvis_index = robot.body_names.index("pelvis")

    pelvis_quat_w = robot.data.body_quat_w[:, pelvis_index]
    pelvis_lin_vel_w = robot.data.body_lin_vel_w[:, pelvis_index]
    if velocity_frame == "world":
        return task_combined_lateral_speed_penalty(
            pelvis_lin_vel_w, pelvis_quat_w, lateral_deadzone, lateral_scale
        )
    if velocity_frame == "pelvis":
        pelvis_lin_vel_local = quat_apply(quat_inv(pelvis_quat_w), pelvis_lin_vel_w)
        excess = torch.clamp(torch.abs(pelvis_lin_vel_local[:, 1]) - lateral_deadzone, min=0.0)
        return torch.square(excess / max(lateral_scale, 1e-6))
    raise ValueError(f"Unsupported velocity_frame={velocity_frame!r}; use 'world' or 'pelvis'.")


def task_heading_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Reward pelvis forward (XY) aligned with task +X. Returns cos(yaw) clamped to [0, 1]."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    pelvis_index = robot.body_names.index("pelvis")
    pelvis_quat_w = robot.data.body_quat_w[:, pelvis_index]
    return task_pelvis_heading_cos_world_x(pelvis_quat_w).clamp(min=0.0, max=1.0)


def waist_action_rate_l2_clip(env: ManagerBasedRLEnv, waist_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    if waist_cfg is None:
        raise ValueError("waist_cfg cannot be None")
    robot = env.scene[waist_cfg.name]
    idx = torch.as_tensor(robot.find_joints(waist_cfg.joint_names, preserve_order=True)[0], device=env.device)
    return torch.sum(torch.square(env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]), dim=1).clamp(max=100.0)


def _get_kick_tracker(command: MotionCommand) -> KickContactTracker:
    tracker = getattr(command, "kick_contact_tracker", None)
    if tracker is None:
        raise RuntimeError("MotionCommand is missing kick_contact_tracker; ensure command setup is up to date.")
    return tracker


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


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


def _robot_anchor_twist_b(command: MotionCommand) -> torch.Tensor:
    """Current robot pelvis twist in its yaw-only local frame."""
    robot_yaw_inv = quat_inv(yaw_quat(command.robot_anchor_quat_w))
    lin_vel_b = quat_apply(robot_yaw_inv, command.robot_anchor_lin_vel_w)
    ang_vel_b = quat_apply(robot_yaw_inv, command.robot_anchor_ang_vel_w)
    return torch.stack((lin_vel_b[:, 0], lin_vel_b[:, 1], ang_vel_b[:, 2]), dim=-1)


def motion_anchor_local_lin_vel_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track the active command's forward/lateral velocity in current pelvis coordinates."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not hasattr(command, "locomotion_twist_command_b"):
        raise RuntimeError(f"motion command '{command_name}' does not expose a local twist command")
    error = torch.sum(torch.square(command.locomotion_twist_command_b()[:, :2] - _robot_anchor_twist_b(command)[:, :2]), dim=-1)
    return torch.exp(-error / max(std, 1e-6) ** 2)


def motion_anchor_local_ang_vel_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track the active command's yaw rate in current pelvis coordinates."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not hasattr(command, "locomotion_twist_command_b"):
        raise RuntimeError(f"motion command '{command_name}' does not expose a local twist command")
    error = torch.square(command.locomotion_twist_command_b()[:, 2] - _robot_anchor_twist_b(command)[:, 2])
    return torch.exp(-error / max(std, 1e-6) ** 2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


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


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
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


def target_point_proximity(env: ManagerBasedRLEnv, std: float, command_name: str = "motion",) -> torch.Tensor:
    """Reward proximity to the target point (ball) and freeze at first kick contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    
    # Compute current proximity reward.
    base_xy = command.robot_anchor_pos_w[..., :2]
    target = get_target_point_world(env, command_name).to(device=base_xy.device, dtype=base_xy.dtype)
    diff_xy = base_xy - target[..., :2]
    error = torch.sum(diff_xy * diff_xy, dim=-1)
    proximity_reward = torch.exp(-error / std**2)
    
    # Query kick-contact status.
    contact_awarded = tracker.get_contact_awarded()
    frozen_reward = tracker.get_frozen_proximity_reward()
    
    # Freeze reward for environments that just kicked this step.
    new_kick_mask = contact_awarded & (frozen_reward == 0.0)
    if torch.any(new_kick_mask):
        new_kick_ids = torch.nonzero(new_kick_mask, as_tuple=False).squeeze(-1)
        tracker.freeze_proximity_reward(new_kick_ids, proximity_reward[new_kick_ids])
        frozen_reward = tracker.get_frozen_proximity_reward()
    
    # Return frozen reward after contact; otherwise return current reward.
    reward = torch.where(contact_awarded, frozen_reward, proximity_reward)
    return reward


def pelvis_orientation(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Penalize pelvis pitch/roll tilt to keep the robot upright."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    gravity_vec_w = robot.data.GRAVITY_VEC_W

    # Project gravity vector to pelvis local frame.
    pelvis_proj_gravity = quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
    # print("pelvis_proj_gravity:", gravity_vec_w, pelvis_proj_gravity)
    return torch.sum(torch.square(pelvis_proj_gravity[:, :2]), dim=1)
