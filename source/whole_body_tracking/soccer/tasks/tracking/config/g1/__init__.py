import gymnasium as gym

from . import agents, flat_env_cfg
from . import soccer_flat_env_cfg
from . import soccer_dribbling_env_cfg
from . import soccer_anchor_env_cfg

##
# Register Gym environments.
##

## Motion tracking environments
gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
    },
)


## Soccer environments
###  Stage 1
# Terrain
gym.register(
    id="Tracking-Terrain-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1TerrainMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Terrain-G1-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1TerrainMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)
# Flat
gym.register(
    id="Tracking-Flat-G1-Motion-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatMotionPretrainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Motion-RNN-task",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatMotionTaskPretrainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


###  Stage 2
gym.register(
    id="Tracking-Flat-G1-SoccerDestination-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-SoccerDestination-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-SoccerMoving-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatKickMovingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)




## Advanced Soccer environments

# Only-vision
gym.register(
    id="Tracking-Flat-G1-SoccerBlind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSoccerBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-SoccerBlind-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSoccerBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-SuperSoccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSuperSoccerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-Soccer-Distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSoccerStudentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatStudentTeacherPPORunnerCfg",
    },
)


## Dribbling environments
gym.register(
    id="Tracking-Flat-G1-Dribbling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatDribblingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Dribbling-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatDribblingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-CG-G1-Dribbling-RNN-forward",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatCGDribblingForwardEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-CG-G1-Dribbling-RNN-follow",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatCGDribblingFollowEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-CG-G1-Dribbling-RNN-control",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatCGDribblingControlEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-CG-G1-Dribbling-RNN-full-control",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatCGDribblingFullControlEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

# Alias for older scripts / checkpoints (same env as -forward).
gym.register(
    id="Tracking-CG-G1-Dribbling-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatCGDribblingForwardEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

# CG progressive Stage 1: obs-compatible with G1FlatCGDribblingEnvCfg (anchor_ball_polar).
#   -strict: raw-reference full-body replay, without task-frame/reset perturbations
#   -task:  original Stage-1 (gym id ``...-v0``): forward/lateral/heading task terms
#   -mimic: later split (gym id ``...-v1``): layered mimic-only pretrain
gym.register(
    id="Tracking-CG-G1-Motion-RNN-strict",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatMotionCGPretrainStrictEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-CG-G1-Motion-RNN-mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatMotionCGPretrainMimicEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-CG-G1-Motion-RNN-task",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatMotionCGPretrainTaskEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

# Historical gym ids (same obs layout; reward recipe differs task vs mimic).
gym.register(
    id="Tracking-CG-G1-Motion-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatMotionCGPretrainTaskEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-CG-G1-Motion-RNN-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatMotionCGPretrainMimicEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-CG-Heuristic-G1-Dribbling-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1FlatDribblingCGHeuristicEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)

# Dribbling Stage 1: ankle disturbance mode
gym.register(
    id="Tracking-Flat-G1-Dribbling-AnkleDisturb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1TerrainDribblingAnkleDisturbEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Dribbling-AnkleDisturb-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_dribbling_env_cfg.G1TerrainDribblingAnkleDisturbEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DribblingRecurrentPPORunnerCfg",
    },
)


## Anchor-based kick environments (Sprint 2 — isolated from baseline)
# Stage 1: egocentric observation + velocity downweight
gym.register(
    id="Anchor-Kick-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-Kick-G1-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Stage 2: egocentric obs + ankle masking + kick rewards
gym.register(
    id="Anchor-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Stage 2 CG: Soft Contact Graph kick (Sprint 4)
gym.register(
    id="Anchor-CG-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorCGKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-CG-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorCGKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Stage 2 SM: state-machine kick (APPROACH/STRIKE distance trigger)
gym.register(
    id="Anchor-SM-Kick-G1-Soccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorStateMachineKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Anchor-SM-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AnchorStateMachineKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

## Ablation tests (隔离测试)
# Test 2: xyz obs + velocity/ankle changes
gym.register(
    id="Ablation-Xyz-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AblationXyzKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

# Test 3: polar obs only, keep velocity/ankle as baseline
gym.register(
    id="Ablation-Polar-Kick-G1-Soccer-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_anchor_env_cfg.G1AblationPolarOnlyKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)
