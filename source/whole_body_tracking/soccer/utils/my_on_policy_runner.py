import os
from functools import wraps

import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from soccer.utils.checkpoint_loading import (
    S2_CURRICULUM_INFO_KEY,
    capture_s2_curriculum_state,
    load_checkpoint_with_obs_expand,
    set_s2_training_iteration,
)
from soccer.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        # These two project-local runner options are deliberately removed
        # before the upstream RSL-RL constructor sees the configuration.
        # They bound Gaussian exploration after every PPO update, including
        # when an S1 checkpoint with a larger learned std is resumed into S2.
        runner_cfg = dict(train_cfg)
        self.policy_std_min = runner_cfg.pop("policy_std_min", None)
        self.policy_std_max = runner_cfg.pop("policy_std_max", None)
        self.s2_warm_start_reset_optimizer = runner_cfg.pop(
            "s2_warm_start_reset_optimizer", False
        )
        self.s2_warm_start_policy_std = runner_cfg.pop(
            "s2_warm_start_policy_std", None
        )
        if (
            self.policy_std_min is not None
            and self.policy_std_max is not None
            and float(self.policy_std_min) > float(self.policy_std_max)
        ):
            raise ValueError("policy_std_min cannot be greater than policy_std_max")
        super().__init__(env, runner_cfg, log_dir, device)
        self.registry_name = registry_name
        self._install_s2_iteration_bridge()
        self._install_policy_std_clamp()

    def _policy_module(self):
        if hasattr(self.alg, "policy"):
            return self.alg.policy
        if hasattr(self.alg, "actor_critic"):
            return self.alg.actor_critic
        return None

    def _clamp_policy_std(self) -> None:
        if self.policy_std_min is None and self.policy_std_max is None:
            return
        policy = self._policy_module()
        if policy is None:
            return
        parameters = dict(policy.named_parameters())
        std = parameters.get("std")
        if isinstance(std, torch.Tensor):
            lower = float(self.policy_std_min) if self.policy_std_min is not None else 0.0
            upper = float(self.policy_std_max) if self.policy_std_max is not None else float("inf")
            with torch.no_grad():
                std.clamp_(min=lower, max=upper)
            return
        log_std = parameters.get("log_std")
        if isinstance(log_std, torch.Tensor):
            lower = (
                torch.log(torch.tensor(float(self.policy_std_min), device=log_std.device)).item()
                if self.policy_std_min is not None
                else -float("inf")
            )
            upper = (
                torch.log(torch.tensor(float(self.policy_std_max), device=log_std.device)).item()
                if self.policy_std_max is not None
                else float("inf")
            )
            with torch.no_grad():
                log_std.clamp_(min=lower, max=upper)

    def _install_policy_std_clamp(self) -> None:
        if self.policy_std_min is None and self.policy_std_max is None:
            return
        self._clamp_policy_std()
        original_update = self.alg.update

        @wraps(original_update)
        def update_and_clamp(*args, **kwargs):
            result = original_update(*args, **kwargs)
            self._clamp_policy_std()
            return result

        self.alg.update = update_and_clamp

    def _sync_s2_training_iteration(self) -> bool:
        return set_s2_training_iteration(
            self,
            int(getattr(self, "current_learning_iteration", 0)),
        )

    def _install_s2_iteration_bridge(self) -> None:
        """Publish the exact runner iteration before every rollout action."""
        if not self._sync_s2_training_iteration():
            return
        original_act = self.alg.act

        @wraps(original_act)
        def act_with_iteration(*args, **kwargs):
            self._sync_s2_training_iteration()
            return original_act(*args, **kwargs)

        self.alg.act = act_with_iteration

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        curriculum_state = capture_s2_curriculum_state(self)
        checkpoint_infos = infos
        if curriculum_state is not None:
            checkpoint_infos = dict(infos) if isinstance(infos, dict) else {}
            checkpoint_infos[S2_CURRICULUM_INFO_KEY] = curriculum_state
        super().save(path, checkpoint_infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

    def load(self, path: str, *args, **kwargs):
        """Resume training; auto-expand obs when loading forward ckpt into follow/control."""
        result = load_checkpoint_with_obs_expand(self, path, **kwargs)
        self._clamp_policy_std()
        return result
