"""The deployment-oriented G1 three-stage dribbling curriculum."""

import gymnasium as gym

from . import agents, soccer_dribbling_env_cfg


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


# All stages share the deployment-oriented 163-D observation and 29-D action
# contract.  S1/S2 use the reference local twist; S3 uses the task local twist.
_register(
    "Tracking-CG-G1-Motion-RNN-unified-s1-local-strict",
    soccer_dribbling_env_cfg.G1FlatMotionCGPretrainUnifiedS1LocalStrictEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)
_register(
    "Tracking-CG-G1-Dribbling-RNN-unified-s2-local-reference",
    soccer_dribbling_env_cfg.G1FlatCGDribblingUnifiedS2LocalReferenceEnvCfg,
    "G1DribblingS2RecurrentPPORunnerCfg",
)
_register(
    "Tracking-CG-G1-Dribbling-RNN-unified-s3-local-task",
    soccer_dribbling_env_cfg.G1FlatCGDribblingUnifiedS3LocalTaskEnvCfg,
    "G1DribblingRecurrentPPORunnerCfg",
)
