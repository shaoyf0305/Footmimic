from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ReferenceResidualJointPositionAction(JointPositionAction):
    """Execute bounded arm corrections around the live motion reference.

    The policy interface remains one action per controlled robot joint. Lower-
    body actions keep the standard Isaac Lab joint-position semantics. The
    selected arm actions are dimensionless residuals: zero follows the live
    reference exactly and ``+/-1`` requests the configured per-joint margin.

    This term deliberately does not project or temporally filter the residual.
    It applies the sole tanh to PPO's Gaussian pre-squash arm variable.
    Simulator soft joint limits remain as the final physical safety guard
    before the target is sent to the existing position controller.
    """

    cfg: ReferenceResidualJointPositionActionCfg

    def __init__(self, cfg: ReferenceResidualJointPositionActionCfg, env: ManagerBasedEnv):
        if cfg.reference_target_margin <= 0.0:
            raise ValueError("reference_target_margin must be positive.")

        super().__init__(cfg, env)
        self._residual_env = env

        robot_joint_ids, found_names = self._asset.find_joints(
            cfg.upper_body_joint_names, preserve_order=True
        )
        if len(robot_joint_ids) != len(cfg.upper_body_joint_names):
            raise ValueError(
                "Could not resolve every reference-residual upper-body joint: "
                f"expected {cfg.upper_body_joint_names}, found {found_names}."
            )

        if isinstance(self._joint_ids, slice):
            controlled_robot_ids = list(range(self._asset.num_joints))
        else:
            controlled_robot_ids = [int(index) for index in self._joint_ids]
        self._controlled_robot_ids = tuple(controlled_robot_ids)
        robot_to_action = {
            robot_id: action_id for action_id, robot_id in enumerate(controlled_robot_ids)
        }
        try:
            upper_action_ids = [robot_to_action[int(robot_id)] for robot_id in robot_joint_ids]
        except KeyError as exc:
            raise ValueError(
                f"The joint-position action does not control upper-body robot joint id {exc.args[0]}."
            ) from exc

        self._upper_robot_ids = torch.as_tensor(
            robot_joint_ids, dtype=torch.long, device=self.device
        )
        self._upper_action_ids = torch.as_tensor(
            upper_action_ids, dtype=torch.long, device=self.device
        )
        self._upper_joint_names = tuple(found_names)
        self._upper_residual_margins = torch.full(
            (1, len(found_names)),
            float(cfg.reference_target_margin),
            dtype=self.raw_actions.dtype,
            device=self.device,
        )

        upper_shape = (self.num_envs, len(upper_action_ids))
        self.upper_reference_target = torch.zeros(upper_shape, device=self.device)
        self.upper_raw_target = torch.zeros(upper_shape, device=self.device)
        self.upper_executed_target = torch.zeros(upper_shape, device=self.device)
        self.upper_residual_pre_squash = torch.zeros(upper_shape, device=self.device)
        self.upper_residual_policy = torch.zeros(upper_shape, device=self.device)
        self.upper_residual_commanded = torch.zeros(upper_shape, device=self.device)
        self.upper_residual_executed = torch.zeros(upper_shape, device=self.device)
        self.upper_residual_actor_boundary_fraction = torch.zeros(
            self.num_envs, device=self.device
        )
        self.upper_residual_joint_limit_fraction = torch.zeros(
            self.num_envs, device=self.device
        )

        # The recurrent policy observes the final normalized absolute target so
        # Stage 1 and Stage 2 retain the same 29-D action-feedback interface.
        self.effective_raw_actions = torch.zeros_like(self.raw_actions)
        self.prev_effective_raw_actions = torch.zeros_like(self.raw_actions)

        # The action-rate reward uses only the arm correction for these joints;
        # a moving reference is therefore not incorrectly charged as policy
        # jitter. Soft-limit clipping is included because it changes what the
        # controller can actually execute.
        self.effective_upper_residual_actions = torch.zeros(upper_shape, device=self.device)
        self.prev_effective_upper_residual_actions = torch.zeros_like(
            self.effective_upper_residual_actions
        )

    @property
    def uses_reference_relative_upper_body_residual(self) -> bool:
        return True

    @property
    def uses_bounded_upper_body_policy_action(self) -> bool:
        return True

    def policy_action_ids_for_robot_joint_ids(self, robot_joint_ids: Sequence[int]) -> list[int]:
        robot_to_action = {
            int(robot_id): action_id
            for action_id, robot_id in enumerate(self._controlled_robot_ids)
        }
        try:
            return [robot_to_action[int(robot_id)] for robot_id in robot_joint_ids]
        except KeyError as exc:
            raise ValueError(
                f"Robot joint id {exc.args[0]} is not controlled by this action term."
            ) from exc

    def _scale_tensor(self) -> torch.Tensor:
        if isinstance(self._scale, torch.Tensor):
            return self._scale
        return torch.full_like(self.raw_actions, float(self._scale))

    def _offset_tensor(self) -> torch.Tensor:
        if isinstance(self._offset, torch.Tensor):
            return self._offset
        return torch.full_like(self.raw_actions, float(self._offset))

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)

        action_ids = self._upper_action_ids
        reference_target = self._residual_env.command_manager.get_term(
            self.cfg.command_name
        ).joint_pos[:, self._upper_robot_ids]
        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._upper_robot_ids]

        # PPO stores and evaluates the Gaussian pre-squash variable. Applying
        # tanh exactly once here avoids inverse-tanh reconstruction of float32
        # actions at +/-1 while preserving the same bounded physical residual.
        pre_squash_residual = self.raw_actions[:, action_ids]
        policy_residual = torch.tanh(pre_squash_residual)
        commanded_residual = policy_residual * self._upper_residual_margins
        raw_target = reference_target + commanded_residual
        executed_target = torch.clamp(
            raw_target, min=soft_limits[..., 0], max=soft_limits[..., 1]
        )
        executed_residual = executed_target - reference_target
        self._processed_actions[:, action_ids] = executed_target

        scale = self._scale_tensor()
        offset = self._offset_tensor()
        upper_scale = scale[:, action_ids]
        safe_upper_scale = torch.where(
            torch.abs(upper_scale) < 1.0e-8,
            torch.ones_like(upper_scale),
            upper_scale,
        )
        self.prev_effective_raw_actions[:] = self.effective_raw_actions
        self.effective_raw_actions[:] = self.raw_actions
        self.effective_raw_actions[:, action_ids] = (
            executed_target - offset[:, action_ids]
        ) / safe_upper_scale

        self.prev_effective_upper_residual_actions[:] = self.effective_upper_residual_actions
        self.effective_upper_residual_actions[:] = (
            executed_residual / self._upper_residual_margins
        )

        self.upper_reference_target[:] = reference_target
        self.upper_raw_target[:] = raw_target
        self.upper_executed_target[:] = executed_target
        self.upper_residual_pre_squash[:] = pre_squash_residual
        self.upper_residual_policy[:] = policy_residual
        self.upper_residual_commanded[:] = commanded_residual
        self.upper_residual_executed[:] = executed_residual
        self.upper_residual_actor_boundary_fraction[:] = (
            torch.abs(policy_residual) >= 0.95
        ).float().mean(dim=1)
        self.upper_residual_joint_limit_fraction[:] = (
            torch.abs(executed_target - raw_target) > 1.0e-6
        ).float().mean(dim=1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self.upper_reference_target[env_ids] = 0.0
        self.upper_raw_target[env_ids] = 0.0
        self.upper_executed_target[env_ids] = 0.0
        self.upper_residual_pre_squash[env_ids] = 0.0
        self.upper_residual_policy[env_ids] = 0.0
        self.upper_residual_commanded[env_ids] = 0.0
        self.upper_residual_executed[env_ids] = 0.0
        self.upper_residual_actor_boundary_fraction[env_ids] = 0.0
        self.upper_residual_joint_limit_fraction[env_ids] = 0.0
        self.effective_raw_actions[env_ids] = 0.0
        self.prev_effective_raw_actions[env_ids] = 0.0
        self.effective_upper_residual_actions[env_ids] = 0.0
        self.prev_effective_upper_residual_actions[env_ids] = 0.0


@configclass
class ReferenceResidualJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for direct reference-relative upper-body residuals."""

    class_type: type[ActionTerm] = ReferenceResidualJointPositionAction
    upper_body_joint_names: list[str] = MISSING
    command_name: str = "motion"
    reference_target_margin: float = 0.25
