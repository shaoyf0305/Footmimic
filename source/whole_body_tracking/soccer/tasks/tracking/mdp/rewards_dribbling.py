"""Reward functions used by the active Stage-2 dribbling controller."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import (
    TASK_STATE_STOP,
    MotionCommand,
    locomotion_task_state_mask,
)
from soccer.tasks.tracking.mdp.rewards import motion_relative_foot_position_error_exp
from soccer.tasks.tracking.mdp.task_frame import (
    forward_dominance_gate,
    task_pelvis_heading_cos_world_x,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Shared ball-contact signals
# ---------------------------------------------------------------------------


def soccer_ball_contact_net_force_w(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
) -> torch.Tensor:
    """Net contact force on the soccer ball body, shape ``(num_envs, 3)``.

    Zeros are returned when the sensor has no usable data (same convention as
    ``_identify_contact_body``).
    """
    device = env.device
    num_envs = env.num_envs
    zero = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)

    contact_sensor: ContactSensor = env.scene.sensors[ball_sensor_name]
    forces_data = contact_sensor.data

    forces = None
    if hasattr(forces_data, "net_forces_w_history"):
        fh = forces_data.net_forces_w_history
        if fh is not None and fh.numel() > 0:
            forces = fh.to(device)
            if forces.ndim >= 4:
                forces = forces.max(dim=1).values
    if forces is None:
        if hasattr(forces_data, "net_forces_w"):
            f = forces_data.net_forces_w
            if f is not None and f.numel() > 0:
                forces = f.to(device)

    if forces is None or forces.ndim < 2:
        return zero

    if forces.ndim == 3:
        return forces[:, 0, :]
    if forces.shape[-1] >= 3:
        return forces[:, :3]
    return zero


def _soccer_ball_filtered_forces_w(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str,
) -> tuple[torch.Tensor | None, tuple[str, ...]]:
    """Return per-filter ball contact forces and their robot body names.

    Newer IsaacLab versions expose ``force_matrix_w`` for the explicit filter
    paths configured on the ball sensor.  The ball is the single sensor body,
    so its matrix is reduced to ``(num_envs, num_filtered_bodies, 3)``.  A
    ``None`` tensor signals that the runtime only supports the legacy net force.
    """
    contact_sensor: ContactSensor = env.scene.sensors[ball_sensor_name]
    matrix = getattr(contact_sensor.data, "force_matrix_w", None)
    if not isinstance(matrix, torch.Tensor) or matrix.numel() == 0:
        return None, ()

    matrix = torch.nan_to_num(matrix.to(env.device))
    if matrix.ndim == 4:
        # Expected ContactSensorData layout: (N, sensor_bodies, filters, 3).
        matrix = matrix.sum(dim=1)
    elif matrix.ndim == 2 and matrix.shape[-1] == 3:
        matrix = matrix.unsqueeze(1)
    if matrix.ndim != 3 or matrix.shape[-1] < 3:
        return None, ()
    matrix = matrix[..., :3]

    filter_expr = getattr(contact_sensor.cfg, "filter_prim_paths_expr", ()) or ()
    names = tuple(str(path).rstrip("/").rsplit("/", 1)[-1] for path in filter_expr)
    usable = min(matrix.shape[1], len(names))
    if usable <= 0:
        return None, ()
    return matrix[:, :usable], names[:usable]


def soccer_ball_body_contact_force_magnitudes(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    body_names: tuple[str, ...] | list[str] | None = None,
    *,
    horizontal_only: bool = True,
) -> tuple[torch.Tensor, tuple[str, ...], bool]:
    """Return per-body robot-ball force magnitudes.

    The boolean return value is true when filtered contact-pair data was
    available.  When ``body_names`` is provided, output columns follow that
    exact order and missing filters are filled with zero.
    """
    filtered_forces, filtered_names = _soccer_ball_filtered_forces_w(env, ball_sensor_name)
    requested_names = filtered_names if body_names is None else tuple(body_names)
    if filtered_forces is None:
        return (
            torch.zeros(env.num_envs, len(requested_names), device=env.device, dtype=torch.float32),
            requested_names,
            False,
        )

    components = filtered_forces[..., :2] if horizontal_only else filtered_forces
    filtered_magnitudes = torch.norm(components, dim=-1)
    if body_names is None:
        return filtered_magnitudes, filtered_names, True

    output = torch.zeros(
        env.num_envs,
        len(requested_names),
        device=env.device,
        dtype=filtered_magnitudes.dtype,
    )
    filter_index = {name: index for index, name in enumerate(filtered_names)}
    for output_index, name in enumerate(requested_names):
        source_index = filter_index.get(name)
        if source_index is not None:
            output[:, output_index] = filtered_magnitudes[:, source_index]
    return output, requested_names, True


def soccer_ball_contact_force_magnitude(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    *,
    horizontal_only: bool = True,
) -> torch.Tensor:
    """Magnitude of the ball's net contact force for diagnostics.

    The net signal also includes ball-ground contact/friction, so it must not
    be used as a robot-contact predicate.  Robot-contact decisions use
    :func:`soccer_ball_robot_contact`; raw safety terms use
    :func:`soccer_ball_max_link_contact_force_magnitude`.
    """
    f = soccer_ball_contact_net_force_w(env, ball_sensor_name)
    if horizontal_only:
        return torch.norm(f[:, :2], dim=-1)
    return torch.norm(f, dim=-1)


def soccer_ball_max_link_contact_force_magnitude(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    *,
    horizontal_only: bool = True,
) -> torch.Tensor:
    """Maximum robot-link force on the ball for anti-trapping and safety.

    Opposing link forces cannot cancel in this channel.  On an older runtime
    without ``force_matrix_w``, fall back to the v5 global net-force signal so
    the environment remains usable, albeit without link-level discrimination.
    """
    body_magnitudes, _body_names, filtered_available = soccer_ball_body_contact_force_magnitudes(
        env,
        ball_sensor_name,
        horizontal_only=horizontal_only,
    )
    if filtered_available and body_magnitudes.shape[1] > 0:
        return body_magnitudes.max(dim=1).values
    return soccer_ball_contact_force_magnitude(
        env,
        ball_sensor_name,
        horizontal_only=horizontal_only,
    )


def soccer_ball_robot_contact(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    *,
    contact_force_threshold: float = 1.0,
    hold_steps: int = 2,
) -> torch.Tensor:
    """Temporally stable robot-ball contact from filtered per-link forces.

    ``soccer_ball_contact_force_magnitude`` intentionally remains available as
    net-force telemetry, but it includes ball-ground friction and therefore is
    not a reliable robot-contact predicate.  This function thresholds the
    maximum filtered robot-link force and keeps a detected contact active for
    ``hold_steps`` additional control steps.  The short hold recovers the
    temporal stability of the legacy three-frame signal without reintroducing
    non-robot contacts or force cancellation.

    Results are cached by episode step and parameter tuple so reward terms,
    terminations, and diagnostics all observe exactly the same contact state
    without advancing the hold counter multiple times in one control step.
    """
    if hold_steps < 0:
        raise ValueError("hold_steps must be non-negative.")

    raw_force = soccer_ball_max_link_contact_force_magnitude(env, ball_sensor_name)
    raw_contact = raw_force > float(contact_force_threshold)
    step_buf = getattr(env, "episode_length_buf", None)
    if not isinstance(step_buf, torch.Tensor) or step_buf.shape[0] != env.num_envs:
        step_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    else:
        step_buf = step_buf.to(device=env.device, dtype=torch.long)

    cache_name = "_soccer_ball_robot_contact_cache"
    cache = getattr(env, cache_name, None)
    if not isinstance(cache, dict):
        cache = {}
    key = (str(ball_sensor_name), float(contact_force_threshold), int(hold_steps))
    state = cache.get(key)
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("last_step"), torch.Tensor)
        or state["last_step"].shape[0] != env.num_envs
    ):
        state = {
            "last_step": torch.full_like(step_buf, -1),
            "steps_since_contact": torch.full(
                (env.num_envs,),
                fill_value=int(hold_steps) + 1,
                device=env.device,
                dtype=torch.int32,
            ),
        }

    last_step = state["last_step"]
    steps_since = state["steps_since_contact"]
    update_mask = last_step != step_buf
    next_steps_since = torch.where(
        raw_contact,
        torch.zeros_like(steps_since),
        torch.clamp(steps_since + 1, max=int(hold_steps) + 1),
    )
    # Never carry a contact hold across an environment reset.  Contact at the
    # reset pose can be detected normally from the following control step.
    next_steps_since = torch.where(
        step_buf == 0,
        torch.full_like(next_steps_since, int(hold_steps) + 1),
        next_steps_since,
    )
    steps_since = torch.where(update_mask, next_steps_since, steps_since)
    last_step = torch.where(update_mask, step_buf, last_step)
    contact = steps_since <= int(hold_steps)

    cache[key] = {
        "last_step": last_step,
        "steps_since_contact": steps_since,
        "contact": contact,
    }
    setattr(env, cache_name, cache)
    setattr(env, "_soccer_ball_max_link_contact_force", raw_force)
    setattr(env, "_soccer_ball_raw_link_contact", raw_contact)
    setattr(env, "_soccer_ball_robot_contact", contact)
    setattr(env, "_soccer_ball_steps_since_link_contact", steps_since)
    return contact


def _dribbling_sim_contact(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str,
    contact_force_threshold: float,
) -> torch.Tensor:
    """Unified robot-contact truth: filtered per-link force with 2-step hold."""
    return soccer_ball_robot_contact(
        env,
        ball_sensor_name,
        contact_force_threshold=contact_force_threshold,
        hold_steps=2,
    )


def _dribbling_cg_gated_sim_contact(
    command: MotionCommand,
    sim_contact: torch.Tensor,
) -> torch.Tensor:
    """On CG-labeled clips, only count touches on annotated contact frames."""
    if not hasattr(command, "motion_has_dribble_cg_label"):
        return sim_contact
    labeled = command.motion_has_dribble_cg_label
    if not torch.any(labeled):
        return sim_contact
    ref = command.dribble_cg_contact_ref
    return torch.where(labeled, sim_contact & ref, sim_contact)


def _dribbling_recent_contact_gate(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str,
    contact_force_threshold: float,
    recent_contact_window: int,
    buf_name: str = "_dribbling_steps_since_contact",
    command: MotionCommand | None = None,
    cg_gated: bool = False,
) -> torch.Tensor:
    """Per-env gate in ``[0, 1]``: 1 iff robot-ball contact within the last N steps."""
    sim_contact = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    if cg_gated and command is not None:
        has_contact = _dribbling_cg_gated_sim_contact(command, sim_contact)
    else:
        has_contact = sim_contact

    if recent_contact_window <= 0:
        return has_contact.to(torch.float32)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        step_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    reset_mask = step_buf == 0

    cnt = getattr(env, buf_name, None)
    if cnt is None or cnt.shape[0] != env.num_envs:
        cnt = torch.full(
            (env.num_envs,),
            fill_value=recent_contact_window + 1,
            device=env.device,
            dtype=torch.int32,
        )

    cnt = torch.where(
        reset_mask,
        torch.full_like(cnt, recent_contact_window + 1),
        torch.where(has_contact, torch.zeros_like(cnt), cnt + 1),
    )
    setattr(env, buf_name, cnt)
    return (cnt <= int(recent_contact_window)).to(torch.float32)


def _command_direction_xy(command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the active locomotion direction and speed in world XY.

    The task frame is fixed to world ``+X`` for the legacy forward task.  In
    control, however, the direction must come from the active external command
    so that reward geometry rotates together with a requested heading.
    """
    if hasattr(command, "locomotion_lin_vel_command_w"):
        command_vel_xy = command.locomotion_lin_vel_command_w()[:, :2]
    else:
        command_vel_xy = command.anchor_lin_vel_w[:, :2]
    speed = torch.norm(command_vel_xy, dim=-1)
    # At zero speed (IDLE/STOP), keep the last commanded heading as the ball
    # corridor.  Falling back unconditionally to +X would make a stopped
    # turning run suddenly evaluate its ball geometry in the wrong direction.
    heading = getattr(command, "locomotion_cmd_heading", None)
    if isinstance(heading, torch.Tensor):
        fallback = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    else:
        fallback = torch.zeros_like(command_vel_xy)
        fallback[:, 0] = 1.0
    direction = torch.where(
        (speed > 1.0e-4).unsqueeze(-1),
        command_vel_xy / speed.unsqueeze(-1).clamp(min=1.0e-4),
        fallback,
    )
    return direction, speed


def _command_frame_components(vector_xy: torch.Tensor, direction_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project XY vectors onto the commanded forward and lateral axes."""
    forward = torch.sum(vector_xy * direction_xy, dim=-1)
    lateral_axis = torch.stack((-direction_xy[:, 1], direction_xy[:, 0]), dim=-1)
    lateral = torch.sum(vector_xy * lateral_axis, dim=-1)
    return forward, lateral


def dribbling_stop_settle_state(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    settled_pelvis_speed: float = 0.05,
    settled_pelvis_angular_speed: float = 0.20,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return the STOP success predicate and its underlying physical measurements.

    The reward and successful-STOP termination call this same helper so an
    episode cannot reset under criteria different from those it was trained to
    optimize.  STOP is intentionally robot-only: the ball may continue rolling
    or leave the control corridor after the final dribble touch.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    pelvis_id = robot.body_names.index("pelvis")
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    forward_offset, lateral_offset = _command_frame_components(offset_xy, direction_xy)
    ball_speed = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    pelvis_speed = torch.norm(robot.data.body_lin_vel_w[:, pelvis_id, :2], dim=-1)
    pelvis_angular_speed = torch.norm(robot.data.body_ang_vel_w[:, pelvis_id], dim=-1)
    active = locomotion_task_state_mask(command, (TASK_STATE_STOP,))
    settled = (
        active
        & (pelvis_speed <= float(settled_pelvis_speed))
        & (pelvis_angular_speed <= float(settled_pelvis_angular_speed))
    )
    return active, pelvis_speed, pelvis_angular_speed, ball_speed, forward_offset, lateral_offset, settled


# ---------------------------------------------------------------------------
# Dynamic proximity: keep the ball in a safe command-frame corridor.
# ---------------------------------------------------------------------------


def dribbling_command_dynamic_proximity(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    near_dist: float = 0.2,
    far_dist: float = 0.5,
    penalty_std: float = 0.15,
    pelvis_speed_min: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    no_contact_zone_damping: float = 1.0,
    zone_lateral_abs_max: float = 0.18,
) -> torch.Tensor:
    """Reward the ball only while it is inside the safe forward corridor.

    Earlier versions returned a decayed but still positive reward when the ball
    was closer than ``near_dist``.  That made a trapped ball profitable.  The
    near side is now handled by ``dribbling_ball_too_close_penalty`` and the far
    side by chase/lost-ball terms, so this reward is exactly zero outside the
    corridor.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    forward_offset, lateral_offset = _command_frame_components(offset_xy, direction_xy)

    in_forward_corridor = (forward_offset >= near_dist) & (forward_offset <= far_dist)
    reward = torch.exp(-lateral_offset.square() / max(penalty_std, 1.0e-6) ** 2)
    reward = reward * in_forward_corridor.to(reward.dtype)

    if pelvis_speed_min > 0.0:
        pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
        reward = reward * torch.clamp(pelvis_speed / pelvis_speed_min, max=1.0)

    if no_contact_zone_damping < 1.0 - 1.0e-6:
        in_corridor = in_forward_corridor & (torch.abs(lateral_offset) <= zone_lateral_abs_max)
        no_touch = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
        reward = torch.where(in_corridor & no_touch, reward * no_contact_zone_damping, reward)
    return reward


def dribbling_ball_too_close_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_dist: float = 0.28,
    full_penalty_dist: float = 0.14,
) -> torch.Tensor:
    """Continuous anti-trap penalty for a ball too close along the command axis.

    The penalty is zero at and beyond ``min_forward_dist`` and reaches one at
    ``full_penalty_dist``.  It is independent of contact detection, so force
    cancellation or intermittent sensor contact cannot hide a squeezed ball.
    """
    if full_penalty_dist >= min_forward_dist:
        raise ValueError("full_penalty_dist must be smaller than min_forward_dist.")
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    forward_offset, lateral_offset = _command_frame_components(offset_xy, direction_xy)
    penalty = torch.clamp(
        (float(min_forward_dist) - forward_offset)
        / max(float(min_forward_dist) - float(full_penalty_dist), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    setattr(env, "_dribbling_ball_forward_offset", forward_offset)
    setattr(env, "_dribbling_ball_lateral_offset", lateral_offset)
    setattr(env, "_dribbling_ball_too_close_penalty", penalty)
    return penalty


# ---------------------------------------------------------------------------
# Helper: identify which robot body caused the ball contact
# ---------------------------------------------------------------------------

def _identify_contact_body(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    ball_sensor_name: str,
    all_body_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify which robot body caused the ball contact.

    Returns:
        has_contact: (N,) bool — whether a robot link contacts the ball
        contact_force_mag: (N,) float — maximum robot-link force magnitude
        closest_body_idx: (N,) long — index into ``all_body_cfg.body_names``;
            selected by maximum filtered force, or by nearest body on the
            legacy fallback path
    """
    if all_body_cfg is None:
        raise ValueError("all_body_cfg is required to identify the robot body touching the ball.")

    device = env.device
    num_envs = env.num_envs
    closest_body_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

    requested_names = tuple(all_body_cfg.body_names)
    body_force_mag, _body_names, filtered_available = soccer_ball_body_contact_force_magnitudes(
        env,
        ball_sensor_name,
        requested_names,
    )
    filtered_peak, filtered_closest = body_force_mag.max(dim=1)
    if filtered_available:
        # Do not mix the ball's ground-friction force back into a valid
        # robot-link matrix. The net force is only a compatibility fallback.
        contact_force_mag = filtered_peak
        has_contact = filtered_peak > 1.0
        closest_body_idx = torch.where(has_contact, filtered_closest, closest_body_idx)
        return has_contact, contact_force_mag, closest_body_idx

    contact_force_mag = torch.norm(
        soccer_ball_contact_net_force_w(env, ball_sensor_name)[:, :2], dim=-1
    )
    has_contact = contact_force_mag > 1.0
    if not torch.any(has_contact):
        return has_contact, contact_force_mag, closest_body_idx

    # Compatibility fallback for runtimes without filtered contact matrices.
    robot = env.scene[all_body_cfg.name]
    soccer_ball = env.scene["soccer_ball"]

    # Resolve body indices (cached; invalidate if body list changes)
    cache_name = "_dribbling_body_indices_cache"
    cached = getattr(env, cache_name, None)
    names_t = tuple(all_body_cfg.body_names)
    if cached is None or cached.get("names") != names_t:
        body_indices = torch.as_tensor(
            robot.find_bodies(all_body_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long, device=device,
        )
        setattr(env, cache_name, {"names": names_t, "idx": body_indices})
    body_indices = getattr(env, cache_name)["idx"]

    contact_env_ids = torch.nonzero(has_contact, as_tuple=False).squeeze(-1)
    body_pos = robot.data.body_pos_w[contact_env_ids][:, body_indices]  # (M, B, 3)
    ball_pos = soccer_ball.data.root_pos_w[contact_env_ids]  # (M, 3)

    dist = torch.norm(body_pos - ball_pos.unsqueeze(1), dim=-1)  # (M, B)
    closest = torch.argmin(dist, dim=-1)  # (M,) — index into body_names
    closest_body_idx[contact_env_ids] = closest

    return has_contact, contact_force_mag, closest_body_idx


def _is_dribble_legal_ankle_contact(closest_body_idx: torch.Tensor, num_ankle_links: int) -> torch.Tensor:
    """True when the closest link index is one of the leading ankle entries."""
    if num_ankle_links <= 0:
        return torch.zeros_like(closest_body_idx, dtype=torch.bool)
    return closest_body_idx < num_ankle_links


def dribbling_micro_contact_filter(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    max_penalty: float = 2.0,
    ema_alpha: float = 0.4,
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
) -> torch.Tensor:
    """EMA-smoothed penalty when a legal ankle hits the ball too hard."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg
    )
    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    legal_hard_force = torch.where(
        has_contact & is_ankle,
        force_mag,
        torch.zeros_like(force_mag),
    )

    buf_name = "_dribbling_contact_ema"
    ema = getattr(env, buf_name, None)
    if ema is None or ema.shape[0] != env.num_envs:
        ema = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    ema = float(ema_alpha) * legal_hard_force + (1.0 - float(ema_alpha)) * ema

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is not None:
        ema = torch.where(step_buf == 0, torch.zeros_like(ema), ema)
    setattr(env, buf_name, ema)

    excess = torch.clamp(ema - float(force_threshold), min=0.0)
    return torch.clamp(excess / max(float(force_threshold), 1.0e-6), max=float(max_penalty))


def dribbling_undesired_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
) -> torch.Tensor:
    """Return one when the ball contact is caused by a non-ankle body."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, _force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg
    )
    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    return (has_contact & ~is_ankle).to(torch.float32)


# ---------------------------------------------------------------------------
# Pelvis orientation vs motion reference.
# ---------------------------------------------------------------------------


def dribbling_pelvis_quat_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.45,
) -> torch.Tensor:
    """Reward matching motion pelvis orientation (same frame as body tracking)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    names = command.cfg.body_names
    if "pelvis" not in names:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    pi = names.index("pelvis")
    ref = command.body_quat_relative_w[:, pi]
    rob = command.robot_body_quat_w[:, pi]
    err = quat_error_magnitude(ref, rob)
    return torch.exp(-(err ** 2) / (std ** 2))


# ---------------------------------------------------------------------------
# Penalize excessive horizontal ball speed.
# ---------------------------------------------------------------------------


def dribbling_ball_xy_speed_excess_penalty(
    env: ManagerBasedRLEnv,
    speed_cap: float = 3.5,
    linear_scale: float = 1.5,
) -> torch.Tensor:
    """Penalty in ``[0, 1]`` for ``|v_ball,xy|`` above ``speed_cap``."""
    soccer_ball = env.scene["soccer_ball"]
    sp = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    excess = torch.relu(sp - speed_cap)
    return torch.clamp(excess / linear_scale, max=1.0)


def dribbling_ball_coast_without_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    speed_threshold: float = 0.35,
    speed_scale: float = 0.45,
    max_close_xy_dist: float = 0.55,
    recent_contact_grace_steps: int = 0,
) -> torch.Tensor:
    """Penalty when the ball slides fast **near** the robot with no contact.

    A short grace period after each real touch leaves the intended kick-release
    phase unpenalized.  Only prolonged nearby coasting is treated as loss of
    control.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    ball_sp = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    no_touch = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)

    steps_name = "_dribbling_coast_steps_since_contact"
    steps_since_contact = getattr(env, steps_name, None)
    if steps_since_contact is None or steps_since_contact.shape[0] != env.num_envs:
        steps_since_contact = torch.full(
            (env.num_envs,),
            fill_value=int(recent_contact_grace_steps) + 1,
            device=env.device,
            dtype=torch.int32,
        )
    step_buf = getattr(env, "episode_length_buf", None)
    reset_mask = step_buf == 0 if step_buf is not None else torch.zeros_like(no_touch)
    steps_since_contact = torch.where(
        reset_mask,
        torch.full_like(steps_since_contact, int(recent_contact_grace_steps) + 1),
        torch.where(no_touch, steps_since_contact + 1, torch.zeros_like(steps_since_contact)),
    )
    setattr(env, steps_name, steps_since_contact)
    grace_finished = steps_since_contact > int(recent_contact_grace_steps)

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]
    dist_xy = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)
    close = dist_xy <= max_close_xy_dist

    excess = torch.relu(ball_sp - speed_threshold)
    penalty = torch.clamp(excess / max(speed_scale, 1e-6), max=1.0)
    return penalty * (no_touch & close & grace_finished).to(torch.float32)


# ---------------------------------------------------------------------------
# Ball progress along the active command direction.
# ---------------------------------------------------------------------------


def dribbling_command_ball_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_speed: float = 0.2,
    command_speed_ratio: float = 0.50,
    speed_scale: float = 0.25,
    lateral_ratio_max: float = 0.70,
    pelvis_speed_min: float = 0.06,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    require_recent_contact: bool = True,
    recent_contact_window: int = 10,
    cg_gated_contact: bool = False,
) -> torch.Tensor:
    """Reward recent-contact ball progress along the active command direction.

    ``command_speed_ratio`` makes high-speed commands demand proportionally
    faster ball progress without asking the ball to exactly match pelvis speed.
    A lateral gate prevents a sideways kick from scoring as forward dribbling.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, command_speed = _command_direction_xy(command)
    forward_speed, lateral_speed = _command_frame_components(soccer_ball.data.root_lin_vel_w[:, :2], direction_xy)

    required_speed = torch.maximum(
        torch.full_like(command_speed, float(min_forward_speed)),
        command_speed * float(command_speed_ratio),
    )
    progress = torch.clamp(
        (forward_speed - required_speed) / max(speed_scale, 1.0e-6), min=0.0, max=1.0
    )
    lateral_ratio = torch.abs(lateral_speed) / forward_speed.clamp(min=1.0e-4)
    direction_gate = torch.clamp(
        1.0 - lateral_ratio / max(lateral_ratio_max, 1.0e-6), min=0.0, max=1.0
    )

    pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
    gate = torch.clamp(pelvis_speed / max(pelvis_speed_min, 1.0e-6), max=1.0)
    if require_recent_contact:
        gate = gate * _dribbling_recent_contact_gate(
            env,
            ball_sensor_name,
            contact_force_threshold,
            recent_contact_window,
            command=command,
            cg_gated=cg_gated_contact,
        )
    return progress * direction_gate * gate


# ---------------------------------------------------------------------------
# Discourage circling around the ball without touching it.
# ---------------------------------------------------------------------------


def dribbling_orbiting_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    orbit_radius_max: float = 0.9,
    tangential_deadzone: float = 0.08,
    tangential_scale: float = 0.35,
) -> torch.Tensor:
    """Penalty in ``[0,1]`` for tangential pelvis motion around the ball.

    The penalty is active only when the pelvis is near the ball (within
    ``orbit_radius_max`` in XY) and ball contact force is weak
    (``<= contact_force_threshold``). This directly suppresses the common
    local optimum of \"one foot hovering, circling around the ball\".
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]

    r = pelvis_pos_xy - ball_pos_xy
    r_norm = torch.norm(r, dim=-1)
    r_hat = r / (r_norm.unsqueeze(-1) + 1e-6)

    # Tangent unit vector around the ball (CCW): [-y, x]
    t_hat = torch.stack((-r_hat[:, 1], r_hat[:, 0]), dim=-1)
    v_tan = torch.abs(torch.sum(pelvis_vel_xy * t_hat, dim=-1))

    weak_contact = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    near_ball = r_norm <= orbit_radius_max

    core = torch.clamp((v_tan - tangential_deadzone) / tangential_scale, min=0.0, max=1.0)
    return core * (weak_contact & near_ball).to(torch.float32)


# ---------------------------------------------------------------------------
# Legal gentle foot touch.
# ---------------------------------------------------------------------------

def dribbling_legal_foot_touch(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    cg_gated: bool = False,
    min_pelvis_heading: float = 0.0,
) -> torch.Tensor:
    """Reward a **new** gentle ankle touch (rising edge), not sustained trapping."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    gentle = force_mag <= force_threshold
    touch = has_contact & is_ankle & gentle
    if cg_gated:
        touch = _dribbling_cg_gated_sim_contact(command, touch)

    prev_name = "_dribbling_prev_legal_foot_touch"
    prev = getattr(env, prev_name, None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    new_touch = touch & ~prev
    setattr(env, prev_name, touch.detach().clone())

    reward = new_touch.to(torch.float32)
    if min_pelvis_heading > 0.0:
        heading = task_pelvis_heading_cos_world_x(command.robot_pelvis_quat_w).clamp(min=0.0, max=1.0)
        reward = reward * forward_dominance_gate(heading, min_pelvis_heading)
    return reward


# ---------------------------------------------------------------------------
# Annotated contact-graph terms.
# ---------------------------------------------------------------------------


def _dribbling_cg_contact_window_ref(
    command: MotionCommand,
    tolerance_steps: int,
) -> torch.Tensor:
    """Return the current CG contact window, dilated in reference time."""
    if tolerance_steps < 0:
        raise ValueError("tolerance_steps must be non-negative.")

    labels = command.motion.dribble_cg_contact
    phase = command.time_steps
    motion_idx = command.motion_idx
    motion_length = command.motion_length
    max_frame = labels.shape[1] - 1
    window = torch.zeros_like(phase, dtype=torch.bool)
    for offset in range(-int(tolerance_steps), int(tolerance_steps) + 1):
        frame = phase + offset
        valid = (frame >= 0) & (frame < motion_length)
        safe_frame = frame.clamp(min=0, max=max_frame)
        window |= valid & labels[motion_idx, safe_frame].to(torch.bool)
    return window


def dribbling_cg_contact_consistency(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    contact_window_tolerance_steps: int = 2,
    premature_contact_penalty: float = 1.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Score CG contact as an event inside a short annotated time window.

    The former frame-wise equality score was dominated by the much more common
    no-contact frames, so a policy could obtain most of this reward without
    touching the ball.  The event score has no true-negative reward:

    - inside a dilated CG contact window, score 1 after the first robot touch;
    - before the first touch in the window, score 0;
    - outside all windows, raw premature robot contact scores a negative value.

    A successful hit stays latched only until that contact window ends.  The
    smoothed robot-contact signal is used to detect the hit, while premature
    contact deliberately uses the raw per-link signal so its penalty is not
    extended by the temporal hold.
    """
    if premature_contact_penalty < 0.0:
        raise ValueError("premature_contact_penalty must be non-negative.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = command.motion_has_dribble_cg_label
    active = labeled & locomotion_task_state_mask(command, active_task_states)
    window = _dribbling_cg_contact_window_ref(command, contact_window_tolerance_steps) & active
    sim_contact = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    raw_contact = (
        soccer_ball_max_link_contact_force_magnitude(env, ball_sensor_name)
        > float(contact_force_threshold)
    )

    prev_window = getattr(env, "_dribbling_cg_contact_window_prev", None)
    window_hit = getattr(env, "_dribbling_cg_contact_window_hit", None)
    prev_phase = getattr(env, "_dribbling_cg_contact_window_prev_phase", None)
    if not isinstance(prev_window, torch.Tensor) or prev_window.shape[0] != env.num_envs:
        prev_window = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if not isinstance(window_hit, torch.Tensor) or window_hit.shape[0] != env.num_envs:
        window_hit = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if not isinstance(prev_phase, torch.Tensor) or prev_phase.shape[0] != env.num_envs:
        prev_phase = torch.full_like(command.time_steps, -2)

    step_buf = getattr(env, "episode_length_buf", None)
    reset = (
        step_buf == 0
        if isinstance(step_buf, torch.Tensor)
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )
    contiguous_phase = command.time_steps == (prev_phase + 1)
    window_start = window & (reset | ~prev_window | ~contiguous_phase)
    continued_hit = window_hit | sim_contact
    window_hit = torch.where(
        window,
        torch.where(window_start, sim_contact, continued_hit),
        torch.zeros_like(window_hit),
    )

    premature = active & ~window & raw_contact
    score = window_hit.to(torch.float32) * window.to(torch.float32)
    score -= float(premature_contact_penalty) * premature.to(torch.float32)

    setattr(env, "_dribbling_cg_contact_window_prev", window.detach().clone())
    setattr(env, "_dribbling_cg_contact_window_hit", window_hit.detach().clone())
    setattr(env, "_dribbling_cg_contact_window_prev_phase", command.time_steps.detach().clone())
    setattr(env, "_dribbling_cg_contact_window_active", window)
    setattr(env, "_dribbling_cg_premature_contact", premature)
    setattr(env, "_dribbling_cg_contact_event_score", score)
    return score


def dribbling_sustained_contact_penalty(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    ema_window_steps: int = 20,
    duty_threshold: float = 0.25,
    full_penalty_duty: float = 0.60,
) -> torch.Tensor:
    """Penalize excessive recent contact duty instead of only consecutive steps.

    A contact EMA cannot be reset by inserting one force-free frame between
    repeated touches, closing the intermittent-contact loophole used by the
    trapped-ball policy.
    """
    if ema_window_steps <= 0:
        raise ValueError("ema_window_steps must be positive.")
    if not 0.0 <= duty_threshold < full_penalty_duty <= 1.0:
        raise ValueError("Expected 0 <= duty_threshold < full_penalty_duty <= 1.")
    # Sustained trapping is deliberately the exception to the v5 global
    # contact channel: opposing links can squeeze the ball while their net
    # forces cancel, so contact duty must use the maximum per-link force.
    link_force = soccer_ball_max_link_contact_force_magnitude(env, ball_sensor_name)
    in_contact = link_force > contact_force_threshold
    buf_name = "_dribbling_contact_duty_ema"
    duty = getattr(env, buf_name, None)
    if duty is None or duty.shape[0] != env.num_envs:
        duty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    step_buf = getattr(env, "episode_length_buf", None)
    reset_mask = step_buf == 0 if step_buf is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    alpha = 2.0 / (float(ema_window_steps) + 1.0)
    duty = (1.0 - alpha) * duty + alpha * in_contact.to(torch.float32)
    duty = torch.where(reset_mask, torch.zeros_like(duty), duty)
    penalty = torch.clamp(
        (duty - float(duty_threshold))
        / max(float(full_penalty_duty) - float(duty_threshold), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    setattr(env, buf_name, duty)
    setattr(env, "_dribbling_contact_duty_penalty", penalty)
    return penalty


def dribbling_ball_bounce_penalty(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    vz_threshold: float = 0.32,
) -> torch.Tensor:
    """Penalize pinch-bounces with excessive vertical ball speed during contact."""
    soccer_ball = env.scene["soccer_ball"]
    # Safety telemetry must use the instantaneous per-link force.  Extending a
    # contact via the normal 2-step gate would penalize free-flight frames
    # after release and would blur the actual pinch/bounce event.
    link_force = soccer_ball_max_link_contact_force_magnitude(env, ball_sensor_name)
    in_contact = link_force > float(contact_force_threshold)
    vertical_speed = torch.abs(soccer_ball.data.root_lin_vel_w[:, 2])
    return (in_contact & (vertical_speed > float(vz_threshold))).to(torch.float32)


def dribbling_cg_foot_ball_distance_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.12,
    use_xy_only: bool = False,
    left_ankle_body_name: str = "left_ankle_roll_link",
    right_ankle_body_name: str = "right_ankle_roll_link",
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Match sim foot–ball distance to demo distance from synthesized CG trajectory.

    Requires ``dribble_cg_foot_ball_dist`` in motion ``.npz`` (see
    ``scripts/dribble/synthesize_dribble_ball_traj.py``). Active on every frame
    with a stitched demo ball trajectory (contact **and** non-contact approach
    gaps), not only during annotated contact segments.

    - ``ref_dist`` = demo XY distance from the reference foot to synthesized ball.
    - Reference foot comes from ``dribble_cg_dist_foot`` when present, else
      falls back to ``dribble_cg_foot`` (legacy contact-only labels).
    - ``sim_dist`` = distance from that foot to the **sim** ball.
    - reward = ``exp(-(sim_dist - ref_dist)^2 / std^2)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = command.motion_has_dribble_cg_foot_ball_dist_label
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref_dist = command.dribble_cg_foot_ball_dist_ref
    ref_foot_dist = command.dribble_cg_dist_foot_ref
    ref_foot = torch.where(
        ref_foot_dist >= 0,
        ref_foot_dist,
        command.dribble_cg_foot_ref,
    )
    active = labeled & (ref_dist >= 0.0) & (ref_foot >= 0) & locomotion_task_state_mask(
        command, active_task_states
    )

    robot = env.scene[command.cfg.asset_name]
    soccer_ball = env.scene["soccer_ball"]
    ball_pos = soccer_ball.data.root_pos_w[:, :3]

    li = robot.body_names.index(left_ankle_body_name)
    ri = robot.body_names.index(right_ankle_body_name)
    left_pos = robot.data.body_pos_w[:, li]
    right_pos = robot.data.body_pos_w[:, ri]

    foot_pos = torch.where(
        (ref_foot == 0).unsqueeze(-1),
        left_pos,
        torch.where((ref_foot == 1).unsqueeze(-1), right_pos, left_pos),
    )

    delta = foot_pos - ball_pos
    if use_xy_only:
        sim_dist = torch.norm(delta[:, :2], dim=-1)
    else:
        sim_dist = torch.norm(delta, dim=-1)

    err = (sim_dist - ref_dist) ** 2
    rew = torch.exp(-err / max(std, 1e-6) ** 2)
    return rew * active.to(torch.float32)


def dribbling_gait_foot_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.3,
    foot_body_names: list[str] | None = None,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Foot imitation reward active only while the ball is **not** in contact.

    Between touches the policy should follow the reference gait (alternating
  steps) instead of freezing in a kick-ready stance with the dribble foot forward.
    """
    base = motion_relative_foot_position_error_exp(
        env, command_name, std, foot_body_names=foot_body_names
    )
    no_ball = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    command: MotionCommand = env.command_manager.get_term(command_name)
    active = no_ball & locomotion_task_state_mask(command, active_task_states)
    return base * active.to(torch.float32)


def dribbling_command_chase_ball_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    min_ball_ahead: float = 0.25,
    max_chase_xy_dist: float = 1.1,
    pelvis_forward_speed_min: float = 0.22,
    forward_speed_scale: float = 0.45,
) -> torch.Tensor:
    """Reward closing on a nearby ball ahead along the active command direction."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    ball_ahead, _ = _command_frame_components(offset_xy, direction_xy)
    dist_xy = torch.norm(offset_xy, dim=-1)
    pelvis_forward, _ = _command_frame_components(command.robot_anchor_lin_vel_w[:, :2], direction_xy)

    speed_reward = torch.clamp(
        (pelvis_forward - pelvis_forward_speed_min) / max(forward_speed_scale, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    no_ball = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    active = no_ball & (ball_ahead >= min_ball_ahead) & (dist_xy <= max_chase_xy_dist)
    return speed_reward * active.to(torch.float32)


def dribbling_rapid_retouch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    min_steps_between_touches: int = 22,
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    cg_gated: bool = False,
) -> torch.Tensor:
    """Penalty for a **new** legal gentle touch sooner than ``min_steps_between_touches``.

    Encourages kick → chase → kick cadence instead of tapping the ball every step.
    """
    if min_steps_between_touches <= 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )
    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    gentle = force_mag <= force_threshold
    touch = has_contact & is_ankle & gentle
    if cg_gated:
        touch = _dribbling_cg_gated_sim_contact(command, touch)

    prev_touch_name = "_dribbling_prev_retouch_touch"
    steps_name = "_dribbling_steps_since_legal_touch"
    prev_touch = getattr(env, prev_touch_name, None)
    if prev_touch is None or prev_touch.shape[0] != env.num_envs:
        prev_touch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    steps_since = getattr(env, steps_name, None)
    if steps_since is None or steps_since.shape[0] != env.num_envs:
        steps_since = torch.full(
            (env.num_envs,),
            fill_value=min_steps_between_touches + 1,
            device=env.device,
            dtype=torch.int32,
        )

    step_buf = getattr(env, "episode_length_buf", None)
    reset_mask = step_buf == 0 if step_buf is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    steps_since = torch.where(reset_mask, torch.full_like(steps_since, min_steps_between_touches + 1), steps_since + 1)

    new_touch = touch & ~prev_touch
    too_soon = new_touch & (steps_since < int(min_steps_between_touches))
    steps_since = torch.where(new_touch, torch.zeros_like(steps_since), steps_since)

    setattr(env, prev_touch_name, touch.detach().clone())
    setattr(env, steps_name, steps_since)
    return too_soon.to(torch.float32)


def dribbling_command_face_ball(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_distance: float = 0.05,
) -> torch.Tensor:
    """Reward ball placement and pelvis yaw aligned with the requested heading."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    ball_forward, _ = _command_frame_components(offset_xy, direction_xy)
    distance = torch.norm(offset_xy, dim=-1)
    ball_ahead = torch.where(
        distance > float(min_distance),
        (ball_forward / distance.clamp(min=1.0e-4)).clamp(min=0.0, max=1.0),
        torch.ones_like(distance),
    )

    local_forward = torch.zeros(env.num_envs, 3, device=env.device, dtype=offset_xy.dtype)
    local_forward[:, 0] = 1.0
    pelvis_forward_xy = quat_apply(command.robot_pelvis_quat_w, local_forward)[:, :2]
    pelvis_forward_xy = pelvis_forward_xy / torch.norm(pelvis_forward_xy, dim=-1, keepdim=True).clamp(min=1.0e-4)
    heading_alignment = torch.sum(pelvis_forward_xy * direction_xy, dim=-1).clamp(min=0.0, max=1.0)
    return ball_ahead * heading_alignment
