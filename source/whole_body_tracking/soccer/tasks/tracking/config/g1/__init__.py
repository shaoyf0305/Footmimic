"""Gym registrations for the active two-stage Footmimic training pipeline."""

import gymnasium as gym

from . import agents, soccer_dribbling_env_cfg, soccer_flat_env_cfg


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
