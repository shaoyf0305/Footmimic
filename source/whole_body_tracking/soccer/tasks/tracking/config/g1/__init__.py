"""Gym registrations for the active two-stage Footmimic training pipeline."""

import gymnasium as gym

from . import (
    agents,
    soccer_dribbling_ablation_env_cfg,
    soccer_dribbling_env_cfg,
    soccer_flat_env_cfg,
)


# Stage 1: motion-mimic pretraining with the same observation layout as Stage 2.
gym.register(
    id="Tracking-CG-G1-Motion-RNN-mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatMotionPretrainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)


def _register_ablation_task(task_id: str, env_cfg_class: type) -> None:
    """Register one configuration-controlled Essay13 ablation task."""
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_class,
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg"
            ),
        },
    )


_ABLATION_TASKS = {
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-Full": (
        soccer_dribbling_ablation_env_cfg.G1Essay13AblationFullEnvCfg
    ),
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoBallVelocity": (
        soccer_dribbling_ablation_env_cfg.G1Essay13NoExplicitBallVelocityEnvCfg
    ),
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoBallVelocityObservation": (
        soccer_dribbling_ablation_env_cfg.G1Essay13NoBallVelocityObservationEnvCfg
    ),
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoBallVelocityReward": (
        soccer_dribbling_ablation_env_cfg.G1Essay13NoBallVelocityRewardEnvCfg
    ),
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoRecovery": (
        soccer_dribbling_ablation_env_cfg.G1Essay13NoRecoveryBlendingEnvCfg
    ),
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoStage1": (
        soccer_dribbling_ablation_env_cfg.G1Essay13NoStage1InitializationEnvCfg
    ),
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoDenseDistance": (
        soccer_dribbling_ablation_env_cfg.G1Essay13NoDenseDistanceEnvCfg
    ),
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoTouchTiming": (
        soccer_dribbling_ablation_env_cfg.G1Essay13NoTouchTimingEnvCfg
    ),
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoInteractionReference": (
        soccer_dribbling_ablation_env_cfg.G1Essay13NoInteractionReferenceEnvCfg
    ),
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoBodyFootReference": (
        soccer_dribbling_ablation_env_cfg.G1Essay13NoBodyFootReferenceEnvCfg
    ),
}

for _task_id, _env_cfg_class in _ABLATION_TASKS.items():
    _register_ablation_task(_task_id, _env_cfg_class)


# Stage 2: continuous speed/heading/duration dribbling control.
gym.register(
    id="Tracking-CG-G1-Dribbling-RNN-control",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatCGDribblingControlEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)
