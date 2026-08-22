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
    """Control lower joints directly, shoulder/elbow joints as residuals, and wrists by reference.

    The parent action term retains a full 29-joint internal target buffer. The
    external Stage-2 policy exposes only the 15 non-arm actions followed by
    eight shoulder/elbow pre-squash residual variables. Six wrist targets come
    directly from the live motion reference and therefore cannot form
    unsupervised policy null directions.

    Shoulder/elbow residuals use exactly one physical transform:
    ``q_target = q_reference + margin * tanh(z)``. There is no PCA projection,
    actor-mean squash, temporal filter, or second residual envelope.
    """

    cfg: ReferenceResidualJointPositionActionCfg

    @property
    def action_dim(self) -> int:
        # JointPositionAction must construct its internal buffers in the full
        # controlled-joint layout. The reduced public dimension becomes active
        # only after ``super().__init__`` has finished.
        if getattr(self, "_use_reduced_policy_action_layout", False):
            return self._policy_action_dim
        return super().action_dim

    def __init__(self, cfg: ReferenceResidualJointPositionActionCfg, env: ManagerBasedEnv):
        if cfg.reference_target_margin <= 0.0:
            raise ValueError("reference_target_margin must be positive.")

        self._use_reduced_policy_action_layout = False
        super().__init__(cfg, env)
        self._residual_env = env

        upper_robot_ids, upper_names = self._asset.find_joints(
            cfg.upper_body_joint_names, preserve_order=True
        )
        if len(upper_robot_ids) != len(cfg.upper_body_joint_names):
            raise ValueError(
                "Could not resolve every upper-body reference joint: "
                f"expected {cfg.upper_body_joint_names}, found {upper_names}."
            )
        residual_robot_ids, residual_names = self._asset.find_joints(
            cfg.residual_joint_names, preserve_order=True
        )
        if len(residual_robot_ids) != len(cfg.residual_joint_names):
            raise ValueError(
                "Could not resolve every shoulder/elbow residual joint: "
                f"expected {cfg.residual_joint_names}, found {residual_names}."
            )

        upper_robot_id_set = {int(index) for index in upper_robot_ids}
        if any(int(index) not in upper_robot_id_set for index in residual_robot_ids):
            raise ValueError("Every residual joint must also be listed in upper_body_joint_names.")
        if len(set(int(index) for index in residual_robot_ids)) != len(residual_robot_ids):
            raise ValueError("residual_joint_names must not contain duplicates.")

        if isinstance(self._joint_ids, slice):
            controlled_robot_ids = list(range(self._asset.num_joints))
        else:
            controlled_robot_ids = [int(index) for index in self._joint_ids]
        self._controlled_robot_ids = tuple(controlled_robot_ids)
        robot_to_full_action = {
            robot_id: action_id for action_id, robot_id in enumerate(controlled_robot_ids)
        }
        try:
            upper_action_ids = [
                robot_to_full_action[int(robot_id)] for robot_id in upper_robot_ids
            ]
            residual_action_ids = [
                robot_to_full_action[int(robot_id)] for robot_id in residual_robot_ids
            ]
        except KeyError as exc:
            raise ValueError(
                f"The joint-position action does not control upper-body robot joint id {exc.args[0]}."
            ) from exc

        residual_robot_id_set = {int(index) for index in residual_robot_ids}
        reference_only_local_ids = [
            local_id
            for local_id, robot_id in enumerate(upper_robot_ids)
            if int(robot_id) not in residual_robot_id_set
        ]
        upper_local_by_robot = {
            int(robot_id): local_id for local_id, robot_id in enumerate(upper_robot_ids)
        }
        residual_local_ids = [
            upper_local_by_robot[int(robot_id)] for robot_id in residual_robot_ids
        ]

        self._full_action_dim = int(super().action_dim)
        upper_action_id_set = set(upper_action_ids)
        lower_action_ids = [
            action_id
            for action_id in range(self._full_action_dim)
            if action_id not in upper_action_id_set
        ]
        lower_robot_ids = [controlled_robot_ids[action_id] for action_id in lower_action_ids]
        lower_count = len(lower_action_ids)
        residual_count = len(residual_action_ids)
        self._policy_action_dim = lower_count + residual_count

        self._upper_robot_ids = torch.as_tensor(
            upper_robot_ids, dtype=torch.long, device=self.device
        )
        self._upper_action_ids = torch.as_tensor(
            upper_action_ids, dtype=torch.long, device=self.device
        )
        self._upper_joint_names = tuple(upper_names)
        self._residual_robot_ids = torch.as_tensor(
            residual_robot_ids, dtype=torch.long, device=self.device
        )
        self._residual_action_ids = torch.as_tensor(
            residual_action_ids, dtype=torch.long, device=self.device
        )
        self._residual_local_ids = torch.as_tensor(
            residual_local_ids, dtype=torch.long, device=self.device
        )
        self._residual_joint_names = tuple(residual_names)
        self._reference_only_local_ids = torch.as_tensor(
            reference_only_local_ids, dtype=torch.long, device=self.device
        )
        self._reference_only_joint_names = tuple(
            upper_names[index] for index in reference_only_local_ids
        )
        self._lower_action_ids = torch.as_tensor(
            lower_action_ids, dtype=torch.long, device=self.device
        )
        self._lower_robot_ids = torch.as_tensor(
            lower_robot_ids, dtype=torch.long, device=self.device
        )
        self._residual_policy_action_ids = torch.arange(
            lower_count,
            self._policy_action_dim,
            dtype=torch.long,
            device=self.device,
        )
        self._upper_residual_margins = torch.full(
            (1, residual_count),
            float(cfg.reference_target_margin),
            dtype=self.raw_actions.dtype,
            device=self.device,
        )

        upper_shape = (self.num_envs, len(upper_action_ids))
        residual_shape = (self.num_envs, residual_count)
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

        # Both stages expose a full 29-D final normalized target as recurrent
        # action feedback, even though Stage 2 produces only 23 policy actions.
        self.effective_raw_actions = torch.zeros_like(self.raw_actions)
        self.prev_effective_raw_actions = torch.zeros_like(self.raw_actions)

        # Smoothness applies only to the eight executed shoulder/elbow
        # residuals. The moving reference and reference-only wrists are not
        # charged as policy jitter.
        self.effective_upper_residual_actions = torch.zeros(
            residual_shape, device=self.device
        )
        self.prev_effective_upper_residual_actions = torch.zeros_like(
            self.effective_upper_residual_actions
        )

        self._use_reduced_policy_action_layout = True

    @property
    def uses_reference_relative_upper_body_residual(self) -> bool:
        return True

    @property
    def uses_bounded_upper_body_policy_action(self) -> bool:
        return True

    @property
    def residual_policy_action_ids(self) -> torch.Tensor:
        return self._residual_policy_action_ids

    def policy_action_ids_for_robot_joint_ids(self, robot_joint_ids: Sequence[int]) -> list[int]:
        robot_to_policy_action = {
            int(robot_id): action_id
            for action_id, robot_id in enumerate(self._lower_robot_ids.tolist())
        }
        robot_to_policy_action.update(
            {
                int(robot_id): int(policy_id)
                for robot_id, policy_id in zip(
                    self._residual_robot_ids.tolist(),
                    self._residual_policy_action_ids.tolist(),
                )
            }
        )
        try:
            return [robot_to_policy_action[int(robot_id)] for robot_id in robot_joint_ids]
        except KeyError as exc:
            raise ValueError(
                "Reference-only wrist joints do not have policy-action indices "
                f"(robot joint id {exc.args[0]})."
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
        if actions.shape[1] != self._policy_action_dim:
            raise ValueError(
                f"Expected {self._policy_action_dim} Stage-2 actions "
                f"(lower body + shoulder/elbow residuals), got {actions.shape[1]}."
            )

        lower_count = self._lower_action_ids.numel()
        pre_squash_residual = actions[:, self._residual_policy_action_ids]
        full_actions = torch.zeros(
            actions.shape[0],
            self._full_action_dim,
            dtype=actions.dtype,
            device=actions.device,
        )
        full_actions[:, self._lower_action_ids] = actions[:, :lower_count]
        super().process_actions(full_actions)

        reference_target = self._residual_env.command_manager.get_term(
            self.cfg.command_name
        ).joint_pos[:, self._upper_robot_ids]
        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._upper_robot_ids]

        policy_residual = torch.tanh(pre_squash_residual)
        residual_command = policy_residual * self._upper_residual_margins
        commanded_residual = torch.zeros_like(reference_target)
        commanded_residual[:, self._residual_local_ids] = residual_command
        raw_target = reference_target + commanded_residual
        executed_target = torch.clamp(
            raw_target, min=soft_limits[..., 0], max=soft_limits[..., 1]
        )
        executed_residual = executed_target - reference_target
        self._processed_actions[:, self._upper_action_ids] = executed_target

        scale = self._scale_tensor()
        offset = self._offset_tensor()
        upper_scale = scale[:, self._upper_action_ids]
        safe_upper_scale = torch.where(
            torch.abs(upper_scale) < 1.0e-8,
            torch.ones_like(upper_scale),
            upper_scale,
        )
        self.prev_effective_raw_actions[:] = self.effective_raw_actions
        self.effective_raw_actions[:] = self.raw_actions
        self.effective_raw_actions[:, self._upper_action_ids] = (
            executed_target - offset[:, self._upper_action_ids]
        ) / safe_upper_scale

        self.prev_effective_upper_residual_actions[:] = self.effective_upper_residual_actions
        self.effective_upper_residual_actions[:] = (
            executed_residual[:, self._residual_local_ids]
            / self._upper_residual_margins
        )

        pre_squash_all = torch.zeros_like(reference_target)
        policy_residual_all = torch.zeros_like(reference_target)
        pre_squash_all[:, self._residual_local_ids] = pre_squash_residual
        policy_residual_all[:, self._residual_local_ids] = policy_residual

        self.upper_reference_target[:] = reference_target
        self.upper_raw_target[:] = raw_target
        self.upper_executed_target[:] = executed_target
        self.upper_residual_pre_squash[:] = pre_squash_all
        self.upper_residual_policy[:] = policy_residual_all
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
    """Configuration for shoulder/elbow residuals plus reference-only wrists."""

    class_type: type[ActionTerm] = ReferenceResidualJointPositionAction
    upper_body_joint_names: list[str] = MISSING
    residual_joint_names: list[str] = MISSING
    command_name: str = "motion"
    reference_target_margin: float = 0.25
