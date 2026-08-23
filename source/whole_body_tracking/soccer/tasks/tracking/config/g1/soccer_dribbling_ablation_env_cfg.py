"""Single-factor Essay13 ablation configurations.

Every class starts from the frozen Stage-II environment and changes only the
factor named by the class.  The explicit Full class is the re-trained control
group used by the ablation study.
"""

from isaaclab.utils import configclass

from soccer.tasks.tracking.mdp import observations_anchor as obs_anchor

from .soccer_dribbling_env_cfg import G1FlatCGDribblingControlEnvCfg


_CLOSED_LOOP_REWARD_NAMES = (
    "motion_anchor_lin_vel",
    "dribbling_dynamic_proximity",
    "dribbling_ball_velocity_tracking",
    "dribbling_useful_foot_touch",
)


def _zero_ball_velocity_observations(cfg) -> None:
    """Zero actor and critic velocity terms without changing their dimensions."""
    for group_name in ("policy", "critic"):
        group = getattr(cfg.observations, group_name)
        term = group.anchor_ball_velocity_polar_cmd
        term.func = obs_anchor.zero_anchor_ball_velocity_polar_command
        term.params = {"command_name": "motion"}


def _disable_ball_velocity_tracking_reward(cfg) -> None:
    """Remove only the direct command-frame ball-velocity reward."""
    cfg.rewards.dribbling_ball_velocity_tracking = None


def _disable_recovery_blending(cfg) -> None:
    """Keep recovery diagnostics but remove both of its control effects."""
    for reward_name in _CLOSED_LOOP_REWARD_NAMES:
        term = getattr(cfg.rewards, reward_name)
        if term is None:
            raise ValueError(
                f"Recovery ablation requires active reward term {reward_name!r}."
            )
        term.params["recovery_target_blending_enabled"] = False

    # A value of one makes the existing factor identically one while the raw
    # and filtered recovery gates continue to be calculated and logged.
    cfg.rewards.dribbling_ball_velocity_tracking.params[
        "minimum_controllability_gate"
    ] = 1.0


def _disable_dense_interaction_distance(cfg) -> None:
    cfg.rewards.dribbling_cg_foot_ball_distance = None


def _disable_touch_timing_scaling(cfg) -> None:
    # Window-aligned and off-window touches receive the same score.  Touch
    # quality and delayed control-improvement scoring remain unchanged.
    cfg.rewards.dribbling_useful_foot_touch.params[
        "off_window_reward_scale"
    ] = 1.0


@configclass
class G1Essay13AblationFullEnvCfg(G1FlatCGDribblingControlEnvCfg):
    """Re-trained Essay13 control with no MDP factor removed."""

    ablation_variant: str = "full"
    requires_stage1_initialization: bool = True


@configclass
class G1Essay13NoBallVelocityObservationEnvCfg(G1Essay13AblationFullEnvCfg):
    """Zero only the policy and critic ball-velocity observations."""

    ablation_variant: str = "no_ball_velocity_observation"

    def __post_init__(self):
        super().__post_init__()
        _zero_ball_velocity_observations(self)


@configclass
class G1Essay13NoBallVelocityRewardEnvCfg(G1Essay13AblationFullEnvCfg):
    """Remove only the direct ball-velocity tracking reward."""

    ablation_variant: str = "no_ball_velocity_reward"

    def __post_init__(self):
        super().__post_init__()
        _disable_ball_velocity_tracking_reward(self)


@configclass
class G1Essay13NoExplicitBallVelocityEnvCfg(G1Essay13AblationFullEnvCfg):
    """Remove explicit ball-velocity feedback from both input and reward."""

    ablation_variant: str = "no_explicit_ball_velocity"

    def __post_init__(self):
        super().__post_init__()
        _zero_ball_velocity_observations(self)
        _disable_ball_velocity_tracking_reward(self)


@configclass
class G1Essay13NoRecoveryBlendingEnvCfg(G1Essay13AblationFullEnvCfg):
    """Measure recovery gates without letting them alter policy objectives."""

    ablation_variant: str = "no_recovery_blending"

    def __post_init__(self):
        super().__post_init__()
        _disable_recovery_blending(self)


@configclass
class G1Essay13NoStage1InitializationEnvCfg(G1Essay13AblationFullEnvCfg):
    """Use the complete Stage-II MDP but require random network initialization."""

    ablation_variant: str = "no_stage1_initialization"
    requires_stage1_initialization: bool = False


@configclass
class G1Essay13NoDenseDistanceEnvCfg(G1Essay13AblationFullEnvCfg):
    """Remove only the synthesized reference foot--ball distance reward."""

    ablation_variant: str = "no_dense_distance"

    def __post_init__(self):
        super().__post_init__()
        _disable_dense_interaction_distance(self)


@configclass
class G1Essay13NoTouchTimingEnvCfg(G1Essay13AblationFullEnvCfg):
    """Remove only reference-window timing modulation from useful touches."""

    ablation_variant: str = "no_touch_timing"

    def __post_init__(self):
        super().__post_init__()
        _disable_touch_timing_scaling(self)


@configclass
class G1Essay13NoInteractionReferenceEnvCfg(G1Essay13AblationFullEnvCfg):
    """Jointly remove dense distance and touch-window reference signals."""

    ablation_variant: str = "no_interaction_reference"

    def __post_init__(self):
        super().__post_init__()
        _disable_dense_interaction_distance(self)
        _disable_touch_timing_scaling(self)


@configclass
class G1Essay13NoBodyFootReferenceEnvCfg(G1Essay13AblationFullEnvCfg):
    """Optional single-factor removal of Stage-II body/foot regularization."""

    ablation_variant: str = "no_body_foot_reference"

    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_body_pos = None
        self.rewards.motion_foot_pos = None
