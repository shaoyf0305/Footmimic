"""RSL-RL policy support for bounded Stage-2 upper-body residual actions.

The lower-body action dimensions retain the standard Gaussian policy.  Only
the Stage-2 arm residual dimensions use a tanh-transformed Gaussian, including
the change-of-variables correction in ``log_prob``.  This keeps PPO's sampled
actions and likelihood in the same action space instead of relying on an
environment-side clamp that the policy cannot observe probabilistically.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class _PartiallySquashedNormal:
    """Independent Normal with tanh transforms on selected dimensions.

    ``mean`` and ``stddev`` intentionally expose the pre-transform Normal
    parameters.  RSL-RL stores these values for its analytic adaptive-KL
    schedule.  Samples and log probabilities, however, live in the actual
    mixed action space consumed by the environment.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        bounded_action_indices: torch.Tensor,
        epsilon: float,
    ) -> None:
        self._normal = torch.distributions.Normal(mean, std)
        self._bounded_action_indices = bounded_action_indices
        self._epsilon = float(epsilon)

    @property
    def mean(self) -> torch.Tensor:
        return self._normal.mean

    @property
    def stddev(self) -> torch.Tensor:
        return self._normal.stddev

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        pre_squash = self._normal.sample(sample_shape)
        if self._bounded_action_indices.numel() == 0:
            return pre_squash
        actions = pre_squash.clone()
        actions[..., self._bounded_action_indices] = torch.tanh(
            pre_squash[..., self._bounded_action_indices]
        )
        return actions

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self._bounded_action_indices.numel() == 0:
            return self._normal.log_prob(actions)

        pre_squash = actions.clone()
        bounded_actions = torch.clamp(
            actions[..., self._bounded_action_indices],
            min=-1.0 + self._epsilon,
            max=1.0 - self._epsilon,
        )
        pre_squash[..., self._bounded_action_indices] = torch.atanh(bounded_actions)
        log_prob = self._normal.log_prob(pre_squash)
        # dy/dx = 1 - tanh(x)^2.  Subtract log|dy/dx| to express
        # the likelihood in the bounded action space used by PPO storage.
        log_prob[..., self._bounded_action_indices] -= torch.log(
            torch.clamp(1.0 - bounded_actions.square(), min=self._epsilon)
        )
        return log_prob

    def entropy(self) -> torch.Tensor:
        # A tanh-Normal has no simple analytic entropy.  RSL-RL's existing
        # Gaussian entropy is retained as the exploration surrogate, while the
        # PPO probability ratio above uses the exact transformed log-probability.
        return self._normal.entropy()


class BoundedUpperBodyActorCriticRecurrent(ActorCriticRecurrent):
    """Recurrent actor-critic with bounded arm residual action dimensions.

    The module layout and state-dict keys are identical to
    ``ActorCriticRecurrent``.  A bounded pre-squash mean prevents a migrated arm
    head from recreating the old tens-of-radians Gaussian mean, while the tanh
    distribution guarantees that the environment receives values in ``(-1, 1)``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "_bounded_action_indices",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self._bounded_mean_limit = float(math.atanh(0.95))
        self._squash_epsilon = 1.0e-6

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
        if self._bounded_action_indices.numel() == 0:
            return raw_mean
        mean = raw_mean.clone()
        selected = raw_mean[..., self._bounded_action_indices]
        mean[..., self._bounded_action_indices] = self._bounded_mean_limit * torch.tanh(
            selected / self._bounded_mean_limit
        )
        return mean

    def _squash_deterministic_actions(self, pre_squash_mean: torch.Tensor) -> torch.Tensor:
        if self._bounded_action_indices.numel() == 0:
            return pre_squash_mean
        actions = pre_squash_mean.clone()
        actions[..., self._bounded_action_indices] = torch.tanh(
            pre_squash_mean[..., self._bounded_action_indices]
        )
        return actions

    def update_distribution(self, observations: torch.Tensor) -> None:
        raw_mean = self.actor(observations)
        mean = self._distribution_mean(raw_mean)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. "
                "Expected 'scalar' or 'log'."
            )
        self.distribution = _PartiallySquashedNormal(
            mean,
            std,
            self._bounded_action_indices,
            self._squash_epsilon,
        )

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        input_a = self.memory_a(observations)
        raw_mean = self.actor(input_a.squeeze(0))
        return self._squash_deterministic_actions(self._distribution_mean(raw_mean))


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
                "[INFO] Stage-2 bounded policy: tanh-Normal arm actions with corrected "
                f"log-probability on indices {bounded_indices}."
            )


# Register at import time as well so registry-driven callers that instantiate
# RSL-RL's base runner can still resolve the policy class by name.
register_bounded_actor_critic()
