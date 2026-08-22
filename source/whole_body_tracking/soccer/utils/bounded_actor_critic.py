"""RSL-RL support for Stage-2 shoulder/elbow reference residuals.

PPO stores and evaluates the unconstrained Gaussian variable. The environment
applies the sole physical ``tanh`` when converting the eight learned residuals
to joint targets. A pre-squash reward supplies the restoring gradient that the
old nearly-flat post-tanh penalty could not provide.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class BoundedUpperBodyActorCriticRecurrent(ActorCriticRecurrent):
    """Recurrent actor-critic with stable pre-squash Gaussian PPO actions.

    The environment applies exactly one action-to-residual tanh. PPO remains
    in the unconstrained Gaussian space and never reconstructs a saturated
    pre-image with ``atanh``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "_bounded_action_indices",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self._minimum_scalar_std = 0.05
        self._std_floor_warning_emitted = False
        # Inference-only diagnostics.  These tensors are deliberately not
        # buffers and therefore never enter a checkpoint state dict.
        self.last_inference_actor_raw_mean: torch.Tensor | None = None
        self.last_inference_actor_action: torch.Tensor | None = None

    @property
    def bounded_action_indices(self) -> torch.Tensor:
        return self._bounded_action_indices

    def set_bounded_action_indices(self, action_indices: Sequence[int]) -> None:
        unique_indices = sorted({int(index) for index in action_indices})
        std_parameter = getattr(self, "std", getattr(self, "log_std", None))
        if not isinstance(std_parameter, torch.Tensor):
            raise RuntimeError("The actor-critic does not expose a trainable action-noise tensor.")
        if unique_indices and (
            unique_indices[0] < 0 or unique_indices[-1] >= int(std_parameter.numel())
        ):
            raise ValueError(
                f"Bounded action indices {unique_indices} are outside the policy action dimension "
                f"{int(std_parameter.numel())}."
            )
        self._bounded_action_indices = torch.as_tensor(
            unique_indices,
            dtype=torch.long,
            device=std_parameter.device,
        )

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(observations)
        if not torch.isfinite(mean).all():
            invalid_ids = torch.nonzero(
                ~torch.isfinite(mean).all(dim=0), as_tuple=False
            ).flatten()
            raise RuntimeError(
                "Actor raw mean contains NaN/Inf at action indices "
                f"{invalid_ids.detach().cpu().tolist()}."
            )
        if self.noise_std_type == "scalar":
            if not torch.isfinite(self.std).all():
                invalid_ids = torch.nonzero(~torch.isfinite(self.std), as_tuple=False).flatten()
                raise RuntimeError(
                    "Actor exploration std contains NaN/Inf at action indices "
                    f"{invalid_ids.detach().cpu().tolist()}."
                )
            minimum_before_projection = float(self.std.detach().min().item())
            if minimum_before_projection < self._minimum_scalar_std:
                # RSL-RL's scalar-noise parameterization optimizes std itself,
                # so Adam can move it below zero. Project the parameter back to
                # the valid domain before constructing Normal. This preserves
                # the checkpoint parameterization and gives it a normal
                # gradient on the next update, unlike clamping only a temporary
                # tensor in the computation graph.
                with torch.no_grad():
                    self.std.clamp_(min=self._minimum_scalar_std)
                if not self._std_floor_warning_emitted:
                    print(
                        "[WARN] Projected actor scalar std to the safety floor "
                        f"{self._minimum_scalar_std:.3f}; minimum before projection was "
                        f"{minimum_before_projection:.6f}."
                    )
                    self._std_floor_warning_emitted = True
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).clamp_min(self._minimum_scalar_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. "
                "Expected 'scalar' or 'log'."
            )
        self.distribution = torch.distributions.Normal(mean, std)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        input_a = self.memory_a(observations)
        raw_mean = self.actor(input_a.squeeze(0))
        diagnostic_action = raw_mean.clone()
        if self._bounded_action_indices.numel() > 0:
            diagnostic_action[..., self._bounded_action_indices] = torch.tanh(
                raw_mean[..., self._bounded_action_indices]
            )
        self.last_inference_actor_raw_mean = raw_mean.detach()
        self.last_inference_actor_action = diagnostic_action.detach()
        return raw_mean


def register_bounded_actor_critic() -> None:
    """Expose the custom class to RSL-RL 2.x's ``eval(class_name)`` loader."""
    import rsl_rl.modules as rsl_modules
    import rsl_rl.runners.on_policy_runner as rsl_runner_module

    setattr(rsl_modules, BoundedUpperBodyActorCriticRecurrent.__name__, BoundedUpperBodyActorCriticRecurrent)
    setattr(
        rsl_runner_module,
        BoundedUpperBodyActorCriticRecurrent.__name__,
        BoundedUpperBodyActorCriticRecurrent,
    )


def _stage2_bounded_action_indices(env) -> list[int]:
    base_env = getattr(env, "unwrapped", env)
    try:
        action_term = base_env.action_manager.get_term("joint_pos")
    except (AttributeError, KeyError):
        return []
    if not getattr(action_term, "uses_bounded_upper_body_policy_action", False):
        return []
    residual_policy_ids = getattr(action_term, "residual_policy_action_ids", None)
    if not isinstance(residual_policy_ids, torch.Tensor):
        raise RuntimeError("Bounded upper-body action term does not expose its policy action indices.")
    return [int(index) for index in residual_policy_ids.detach().cpu().tolist()]


class BoundedOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner that binds Stage-2 arm indices after env construction."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu") -> None:
        register_bounded_actor_critic()
        super().__init__(env, train_cfg, log_dir, device)
        policy = self.alg.policy
        if not isinstance(policy, BoundedUpperBodyActorCriticRecurrent):
            raise TypeError(
                "The active runner configuration must use "
                "BoundedUpperBodyActorCriticRecurrent."
            )
        bounded_indices = _stage2_bounded_action_indices(env)
        policy.set_bounded_action_indices(bounded_indices)
        if bounded_indices:
            print(
                "[INFO] Stage-2 shoulder/elbow policy: unconstrained Gaussian variables "
                f"with a single environment tanh on indices {bounded_indices}."
            )


# Register at import time as well so registry-driven callers that instantiate
# RSL-RL's base runner can still resolve the policy class by name.
register_bounded_actor_critic()
