"""G1 environments retained for the current dribbling training line."""

import gymnasium as gym

from . import agents, soccer_dribbling_env_cfg, soccer_flat_env_cfg


def _register(env_id: str, env_cfg_entry_point, runner_cfg: str) -> None:
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg}",
        },
    )


# Minimal flat motion and kick environments retained as the shared base line.
_register(
    "Tracking-Flat-G1-Motion-RNN-v0",
    soccer_flat_env_cfg.G1FlatMotionPretrainEnvCfg,
    "G1FlatRecurrentPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-SoccerDestination-v0",
    soccer_flat_env_cfg.G1FlatKickEnvCfg,
    "G1FlatPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-SoccerDestination-RNN-v0",
    soccer_flat_env_cfg.G1FlatKickEnvCfg,
    "G1FlatRecurrentPPORunnerCfg",
)

# Non-CG dribbling baseline.
_register(
    "Tracking-Flat-G1-Dribbling-v0",
    soccer_dribbling_env_cfg.G1FlatDribblingEnvCfg,
    "G1DribblingPPORunnerCfg",
)
_register(
    "Tracking-Flat-G1-Dribbling-RNN-v0",
    soccer_dribbling_env_cfg.G1FlatDribblingEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)

# Frozen Stage-2 baselines and the new polar-only unified control task.
_register(
    "Tracking-CG-G1-Dribbling-RNN-control",
    soccer_dribbling_env_cfg.G1FlatCGDribblingControlEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)
_register(
    "Tracking-CG-G1-Dribbling-RNN-full-control",
    soccer_dribbling_env_cfg.G1FlatCGDribblingFullControlEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)
_register(
    "Tracking-CG-G1-Dribbling-RNN-unified-control",
    soccer_dribbling_env_cfg.G1FlatCGDribblingUnifiedControlEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)

# CG Stage-1 variants.  Only unified-mimic has the exact unified-control input layout.
_register(
    "Tracking-CG-G1-Motion-RNN-strict",
    soccer_dribbling_env_cfg.G1FlatMotionCGPretrainStrictEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)
_register(
    "Tracking-CG-G1-Motion-RNN-mimic",
    soccer_dribbling_env_cfg.G1FlatMotionCGPretrainMimicEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)
_register(
    "Tracking-CG-G1-Motion-RNN-unified-mimic",
    soccer_dribbling_env_cfg.G1FlatMotionCGPretrainUnifiedMimicEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)
