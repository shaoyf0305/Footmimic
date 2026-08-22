"""RSL-RL policy support for Stage-2 upper-body reference residuals.

PPO stores the Gaussian pre-squash variable for every action dimension. The
The arm Gaussian mean is kept inside the numerically informative pre-squash
range, then the Stage-2 action term applies the sole physical ``tanh`` before
assigning the residual its meaning. For a fixed bijection the tanh Jacobian
cancels in PPO's new/old probability ratio, avoiding the numerically fragile
inverse-tanh path for saturated float32 actions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


DEFAULT_ARM_ACTION_MEAN_LIMIT = float(math.atanh(0.95))


class BoundedUpperBodyActorCriticRecurrent(ActorCriticRecurrent):
    """Recurrent actor-critic with stable pre-squash Gaussian PPO actions.

    The module layout and state-dict keys are identical to
    ``ActorCriticRecurrent``. The selected arm Gaussian means use a smooth
    finite guard; the environment then applies exactly one action-to-residual
    tanh. PPO never reconstructs a saturated pre-image with ``atanh``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "_bounded_action_indices",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        # Keep the Gaussian mean inside the numerically informative part of
        # the sole environment-side tanh.  This is a distribution-parameter
        # guard, not a second action-to-residual transform.
        self._bounded_mean_limit = DEFAULT_ARM_ACTION_MEAN_LIMIT
        self._minimum_scalar_std = 0.05
        self._std_floor_warning_emitted = False
        # Inference-only diagnostics.  These tensors are deliberately not
        # buffers and therefore never enter a checkpoint state dict.
        self.last_inference_actor_raw_mean: torch.Tensor | None = None
        self.last_inference_actor_bounded_mean: torch.Tensor | None = None
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

    def _distribution_mean(self, raw_mean: torch.Tensor) -> torch.Tensor:
        """Smoothly bound only the selected Gaussian mean components."""
        if self._bounded_action_indices.numel() == 0:
            return raw_mean
        mean = raw_mean.clone()
        selected = raw_mean[..., self._bounded_action_indices]
        mean[..., self._bounded_action_indices] = self._bounded_mean_limit * torch.tanh(
            selected / self._bounded_mean_limit
        )
        return mean

    def update_distribution(self, observations: torch.Tensor) -> None:
        raw_mean = self.actor(observations)
        if not torch.isfinite(raw_mean).all():
            invalid_ids = torch.nonzero(
                ~torch.isfinite(raw_mean).all(dim=0), as_tuple=False
            ).flatten()
            raise RuntimeError(
                "Actor raw mean contains NaN/Inf at action indices "
                f"{invalid_ids.detach().cpu().tolist()}."
            )
        mean = self._distribution_mean(raw_mean)
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
        bounded_mean = self._distribution_mean(raw_mean)
        diagnostic_action = bounded_mean.clone()
        if self._bounded_action_indices.numel() > 0:
            diagnostic_action[..., self._bounded_action_indices] = torch.tanh(
                bounded_mean[..., self._bounded_action_indices]
            )
        self.last_inference_actor_raw_mean = raw_mean.detach()
        self.last_inference_actor_bounded_mean = bounded_mean.detach()
        self.last_inference_actor_action = diagnostic_action.detach()
        # The environment owns the sole arm tanh. Lower and upper dimensions
        # therefore share one numerically stable Gaussian PPO action space.
        return bounded_mean


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
    upper_ids = getattr(action_term, "_upper_action_ids", None)
    if not isinstance(upper_ids, torch.Tensor):
        raise RuntimeError("Bounded upper-body action term does not expose its policy action indices.")
    return [int(index) for index in upper_ids.detach().cpu().tolist()]


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
                "[INFO] Stage-2 arm policy: smoothly bounded Gaussian means with a single "
                f"environment tanh on indices {bounded_indices}; pre-squash mean limit "
                f"{policy._bounded_mean_limit:.6f}."
            )


# Register at import time as well so registry-driven callers that instantiate
# RSL-RL's base runner can still resolve the policy class by name.
register_bounded_actor_critic()
