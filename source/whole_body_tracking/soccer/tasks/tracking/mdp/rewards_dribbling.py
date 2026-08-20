"""Reward functions used by the active Stage-2 dribbling controller."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import (
    TASK_STATE_DRIBBLE,
    TASK_STATE_STOP,
    MotionCommand,
    locomotion_task_state_mask,
)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Shared ball-contact signals
# ---------------------------------------------------------------------------


_RESET_CONTACT_GUARD_STEPS = 1
_BALL_CONTACT_COLLISION_LABELS = {
    # The URDF attaches the complete lower-leg collision cylinder to the knee
    # link.  Calling this a knee contact is visually misleading.
    "left_knee_link": "left_shin_collision",
    "right_knee_link": "right_shin_collision",
}


def soccer_ball_contact_collision_names(body_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return collision-semantic labels for robot body filter names."""
    return tuple(_BALL_CONTACT_COLLISION_LABELS.get(name, name) for name in body_names)


def _zero_reset_contact_samples(env: ManagerBasedRLEnv, values: torch.Tensor) -> torch.Tensor:
    """Remove the stale PhysX contact sample immediately following reset."""
    step_buf = getattr(env, "episode_length_buf", None)
    if not isinstance(step_buf, torch.Tensor) or step_buf.shape[0] != env.num_envs:
        return values
    reset_guard = step_buf.to(device=values.device) <= _RESET_CONTACT_GUARD_STEPS
    while reset_guard.ndim < values.ndim:
        reset_guard = reset_guard.unsqueeze(-1)
    return torch.where(reset_guard, torch.zeros_like(values), values)


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
        return _zero_reset_contact_samples(env, forces[:, 0, :])
    if forces.shape[-1] >= 3:
        return _zero_reset_contact_samples(env, forces[:, :3])
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
    filter_expr = getattr(contact_sensor.cfg, "filter_prim_paths_expr", ()) or ()
    names = tuple(str(path).rstrip("/").rsplit("/", 1)[-1] for path in filter_expr)
    expected_filter_count = len(names)
    matrix = getattr(contact_sensor.data, "force_matrix_w", None)
    if not isinstance(matrix, torch.Tensor) or matrix.numel() == 0:
        setattr(env, "_soccer_ball_contact_filter_count", 0)
        setattr(env, "_soccer_ball_contact_expected_filter_count", expected_filter_count)
        setattr(env, "_soccer_ball_contact_filter_mapping_valid", False)
        return None, names

    matrix = torch.nan_to_num(matrix.to(env.device))
    if matrix.ndim == 4:
        # Expected ContactSensorData layout: (N, sensor_bodies, filters, 3).
        matrix = matrix.sum(dim=1)
    elif matrix.ndim == 2 and matrix.shape[-1] == 3:
        matrix = matrix.unsqueeze(1)
    if matrix.ndim != 3 or matrix.shape[-1] < 3:
        setattr(env, "_soccer_ball_contact_filter_count", 0)
        setattr(env, "_soccer_ball_contact_expected_filter_count", expected_filter_count)
        setattr(env, "_soccer_ball_contact_filter_mapping_valid", False)
        return None, names
    matrix = matrix[..., :3]

    actual_filter_count = int(matrix.shape[1])
    mapping_valid = actual_filter_count == expected_filter_count and expected_filter_count > 0
    setattr(env, "_soccer_ball_contact_filter_count", actual_filter_count)
    setattr(env, "_soccer_ball_contact_expected_filter_count", expected_filter_count)
    setattr(env, "_soccer_ball_contact_filter_mapping_valid", mapping_valid)
    if not mapping_valid:
        raise RuntimeError(
            "Soccer-ball contact filter mapping is ambiguous: PhysX returned "
            f"{actual_filter_count} columns for {expected_filter_count} configured body filters. "
            "Refusing to assign link names or train with a geometry guess. Verify the active "
            "ContactSensor filter layout. The compatibility fallback is reserved for runtimes "
            "that do not expose force_matrix_w at all."
        )
    return _zero_reset_contact_samples(env, matrix), names


def soccer_ball_contact_filter_status(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
) -> tuple[int, int, bool]:
    """Return runtime/expected filter counts and whether column labels are safe."""
    _soccer_ball_filtered_forces_w(env, ball_sensor_name)
    return (
        int(getattr(env, "_soccer_ball_contact_filter_count", 0)),
        int(getattr(env, "_soccer_ball_contact_expected_filter_count", 0)),
        bool(getattr(env, "_soccer_ball_contact_filter_mapping_valid", False)),
    )


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
    body_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """Temporally stable robot-ball contact from filtered per-link forces.

    ``soccer_ball_contact_force_magnitude`` intentionally remains available as
    net-force telemetry, but it includes ball-ground friction and therefore is
    not a reliable robot-contact predicate.  This function thresholds the
    maximum filtered robot-link force, optionally restricted to ``body_names``,
    and keeps a detected contact active for ``hold_steps`` additional control
    steps.  The short hold recovers the temporal stability of the legacy
    three-frame signal without reintroducing non-robot contacts or force
    cancellation.

    Results are cached by episode step and parameter tuple so reward terms,
    terminations, and diagnostics all observe exactly the same contact state
    without advancing the hold counter multiple times in one control step.
    """
    if hold_steps < 0:
        raise ValueError("hold_steps must be non-negative.")

    requested_names = None if body_names is None else tuple(body_names)
    if requested_names is None:
        raw_force = soccer_ball_max_link_contact_force_magnitude(env, ball_sensor_name)
    else:
        body_forces, _names, filtered_available = soccer_ball_body_contact_force_magnitudes(
            env,
            ball_sensor_name,
            requested_names,
        )
        if filtered_available and body_forces.shape[1] > 0:
            raw_force = body_forces.max(dim=1).values
        else:
            # Older runtimes cannot distinguish links.  Retain the documented
            # net-force compatibility fallback rather than disabling the task.
            raw_force = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
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
    key = (str(ball_sensor_name), float(contact_force_threshold), int(hold_steps), requested_names)
    state = cache.get(key)
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("last_step"), torch.Tensor)
        or not isinstance(state.get("steps_since_contact"), torch.Tensor)
        or not isinstance(state.get("contact"), torch.Tensor)
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
    # Never carry a contact hold or a stale PhysX initialization impulse across
    # an environment reset.  Contact is detected normally after the guard.
    next_steps_since = torch.where(
        step_buf <= _RESET_CONTACT_GUARD_STEPS,
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
    if requested_names is None:
        setattr(env, "_soccer_ball_max_link_contact_force", raw_force)
        setattr(env, "_soccer_ball_raw_link_contact", raw_contact)
        setattr(env, "_soccer_ball_robot_contact", contact)
        setattr(env, "_soccer_ball_steps_since_link_contact", steps_since)
    return contact


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


def _low_pass_ball_velocity_xy(
    env: ManagerBasedRLEnv,
    ball_velocity_xy: torch.Tensor,
    window_steps: int,
    buf_name: str = "_dribbling_ball_velocity_ema_state",
) -> torch.Tensor:
    """Return a reset-safe EMA of ball XY velocity, updated once per control step.

    ``alpha = 1 / window_steps`` gives the filter an approximately
    ``window_steps``-long time constant.  With the active 50 Hz controller,
    ten steps correspond to 0.2 s: long enough to reject the impact spike from
    one kick, but short enough to close the speed loop within a gait cycle.
    """
    if window_steps <= 0:
        raise ValueError("window_steps must be positive.")

    step_buf = getattr(env, "episode_length_buf", None)
    if not isinstance(step_buf, torch.Tensor):
        step_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    state = getattr(env, buf_name, None)
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("last_step"), torch.Tensor)
        or state["last_step"].shape[0] != env.num_envs
        or not isinstance(state.get("velocity_xy"), torch.Tensor)
        or state["velocity_xy"].shape != ball_velocity_xy.shape
        or int(state.get("window_steps", -1)) != int(window_steps)
    ):
        state = {
            "last_step": torch.full_like(step_buf, -1),
            "velocity_xy": ball_velocity_xy.detach().clone(),
            "window_steps": int(window_steps),
        }

    last_step = state["last_step"]
    filtered_velocity_xy = state["velocity_xy"]
    update_mask = last_step != step_buf
    # A decreasing episode counter means this environment was reset since its
    # last reward evaluation.  ``step == 0`` covers explicit reset-time calls.
    reset_mask = (step_buf == 0) | (step_buf < last_step)
    alpha = 1.0 / float(window_steps)
    next_velocity_xy = filtered_velocity_xy + alpha * (
        ball_velocity_xy - filtered_velocity_xy
    )
    next_velocity_xy = torch.where(
        reset_mask.unsqueeze(-1), ball_velocity_xy, next_velocity_xy
    )
    filtered_velocity_xy = torch.where(
        update_mask.unsqueeze(-1), next_velocity_xy, filtered_velocity_xy
    )
    last_step = torch.where(update_mask, step_buf, last_step)

    state["last_step"] = last_step
    state["velocity_xy"] = filtered_velocity_xy.detach().clone()
    setattr(env, buf_name, state)
    return filtered_velocity_xy


def _dribbling_closed_loop_control_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    desired_forward_offset: float,
    prediction_horizon: float,
    position_forward_std: float,
    position_lateral_std: float,
    recovery_forward_half_width: float,
    recovery_lateral_half_width: float,
    recovery_transition_width: float,
    recovery_gate_filter_steps: int,
    position_correction_gain: float,
    max_position_correction_speed: float,
    max_recovery_target_speed: float,
    velocity_ema_window_steps: int,
) -> dict[str, torch.Tensor]:
    """Build the shared position--velocity state for control and recovery.

    The predicted ball position is evaluated in the active command frame.  A
    smooth recovery gate rises only when that prediction leaves the normal
    controllable region.  The recovery velocity is the measured ball velocity
    plus a bounded correction that drives the ball back to the desired
    pelvis-relative offset.  The same correction naturally speeds the pelvis
    up for a far ball and slows it down for a ball that is too close.
    """
    if desired_forward_offset <= 0.0 or prediction_horizon < 0.0:
        raise ValueError("desired offset must be positive and prediction horizon non-negative.")
    if position_forward_std <= 0.0 or position_lateral_std <= 0.0:
        raise ValueError("position tracking std values must be positive.")
    if recovery_forward_half_width < 0.0 or recovery_lateral_half_width < 0.0:
        raise ValueError("recovery half widths must be non-negative.")
    if recovery_transition_width <= 0.0 or recovery_gate_filter_steps <= 0:
        raise ValueError("recovery transition/filter parameters must be positive.")
    if position_correction_gain < 0.0:
        raise ValueError("position_correction_gain must be non-negative.")
    if max_position_correction_speed <= 0.0 or max_recovery_target_speed <= 0.0:
        raise ValueError("recovery speed limits must be positive.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, command_speed = _command_direction_xy(command)

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]
    filtered_ball_vel_xy = _low_pass_ball_velocity_xy(
        env,
        soccer_ball.data.root_lin_vel_w[:, :2],
        velocity_ema_window_steps,
    )
    offset_xy = ball_pos_xy - pelvis_pos_xy
    relative_velocity_xy = filtered_ball_vel_xy - pelvis_vel_xy
    predicted_offset_xy = offset_xy + float(prediction_horizon) * relative_velocity_xy
    predicted_forward, predicted_lateral = _command_frame_components(
        predicted_offset_xy, direction_xy
    )

    forward_overrun = torch.relu(
        torch.abs(predicted_forward - float(desired_forward_offset))
        - float(recovery_forward_half_width)
    )
    lateral_overrun = torch.relu(
        torch.abs(predicted_lateral) - float(recovery_lateral_half_width)
    )
    raw_recovery_gate = torch.clamp(
        torch.maximum(forward_overrun, lateral_overrun)
        / float(recovery_transition_width),
        min=0.0,
        max=1.0,
    )
    dribble_active = locomotion_task_state_mask(command, (TASK_STATE_DRIBBLE,))
    raw_recovery_gate = raw_recovery_gate * dribble_active.to(raw_recovery_gate.dtype)

    step_buf = getattr(env, "episode_length_buf", None)
    if not isinstance(step_buf, torch.Tensor):
        step_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    gate_state_name = "_dribbling_recovery_gate_state"
    gate_state = getattr(env, gate_state_name, None)
    parameter_signature = (
        float(desired_forward_offset),
        float(prediction_horizon),
        float(position_forward_std),
        float(position_lateral_std),
        float(recovery_forward_half_width),
        float(recovery_lateral_half_width),
        float(recovery_transition_width),
        int(recovery_gate_filter_steps),
        float(position_correction_gain),
        float(max_position_correction_speed),
        float(max_recovery_target_speed),
        int(velocity_ema_window_steps),
    )
    valid_gate_state = (
        isinstance(gate_state, dict)
        and isinstance(gate_state.get("last_step"), torch.Tensor)
        and gate_state["last_step"].shape[0] == env.num_envs
        and isinstance(gate_state.get("gate"), torch.Tensor)
        and gate_state["gate"].shape[0] == env.num_envs
    )
    if valid_gate_state and gate_state.get("parameters") != parameter_signature:
        raise ValueError(
            "All closed-loop reward terms must use identical control-state parameters."
        )
    if not valid_gate_state:
        gate_state = {
            "last_step": torch.full_like(step_buf, -1),
            "gate": raw_recovery_gate.detach().clone(),
            "parameters": parameter_signature,
        }

    last_step = gate_state["last_step"]
    recovery_gate = gate_state["gate"]
    update_mask = last_step != step_buf
    reset_mask = (step_buf == 0) | (step_buf < last_step)
    gate_alpha = 2.0 / (float(recovery_gate_filter_steps) + 1.0)
    next_gate = recovery_gate + gate_alpha * (raw_recovery_gate - recovery_gate)
    next_gate = torch.where(reset_mask, raw_recovery_gate, next_gate)
    recovery_gate = torch.where(update_mask, next_gate, recovery_gate)
    last_step = torch.where(update_mask, step_buf, last_step)
    gate_state["last_step"] = last_step
    gate_state["gate"] = recovery_gate.detach().clone()
    setattr(env, gate_state_name, gate_state)

    desired_offset_xy = direction_xy * float(desired_forward_offset)
    position_correction_xy = float(position_correction_gain) * (
        predicted_offset_xy - desired_offset_xy
    )
    correction_speed = torch.norm(position_correction_xy, dim=-1)
    correction_scale = torch.clamp(
        float(max_position_correction_speed)
        / correction_speed.clamp(min=1.0e-6),
        max=1.0,
    )
    position_correction_xy = position_correction_xy * correction_scale.unsqueeze(-1)
    recovery_target_velocity_xy = filtered_ball_vel_xy + position_correction_xy
    recovery_target_speed = torch.norm(recovery_target_velocity_xy, dim=-1)
    recovery_speed_scale = torch.clamp(
        float(max_recovery_target_speed)
        / recovery_target_speed.clamp(min=1.0e-6),
        max=1.0,
    )
    recovery_target_velocity_xy = (
        recovery_target_velocity_xy * recovery_speed_scale.unsqueeze(-1)
    )
    command_target_velocity_xy = direction_xy * command_speed.unsqueeze(-1)
    blended_target_velocity_xy = (
        (1.0 - recovery_gate).unsqueeze(-1) * command_target_velocity_xy
        + recovery_gate.unsqueeze(-1) * recovery_target_velocity_xy
    )

    position_error_norm = torch.sqrt(
        ((predicted_forward - float(desired_forward_offset)) / float(position_forward_std)).square()
        + (predicted_lateral / float(position_lateral_std)).square()
    )
    radial_direction_xy = offset_xy / torch.norm(offset_xy, dim=-1, keepdim=True).clamp(min=1.0e-6)
    closing_speed = torch.sum(
        (pelvis_vel_xy - filtered_ball_vel_xy) * radial_direction_xy,
        dim=-1,
    )

    state = {
        "direction_xy": direction_xy,
        "command_speed": command_speed,
        "offset_xy": offset_xy,
        "relative_velocity_xy": relative_velocity_xy,
        "predicted_offset_xy": predicted_offset_xy,
        "predicted_forward": predicted_forward,
        "predicted_lateral": predicted_lateral,
        "position_error_norm": position_error_norm,
        "raw_recovery_gate": raw_recovery_gate,
        "recovery_gate": recovery_gate,
        "filtered_ball_velocity_xy": filtered_ball_vel_xy,
        "command_target_velocity_xy": command_target_velocity_xy,
        "position_correction_xy": position_correction_xy,
        "recovery_target_velocity_xy": recovery_target_velocity_xy,
        "blended_target_velocity_xy": blended_target_velocity_xy,
        "closing_speed": closing_speed,
        "dribble_active": dribble_active,
    }
    setattr(env, "_dribbling_predicted_ball_offset_xy", predicted_offset_xy)
    setattr(env, "_dribbling_predicted_ball_forward_offset", predicted_forward)
    setattr(env, "_dribbling_predicted_ball_lateral_offset", predicted_lateral)
    setattr(env, "_dribbling_position_error_norm", position_error_norm)
    setattr(env, "_dribbling_recovery_gate_raw", raw_recovery_gate)
    setattr(env, "_dribbling_recovery_gate", recovery_gate)
    setattr(env, "_dribbling_command_target_velocity_xy", command_target_velocity_xy)
    setattr(env, "_dribbling_position_correction_velocity_xy", position_correction_xy)
    setattr(env, "_dribbling_recovery_target_velocity_xy", recovery_target_velocity_xy)
    setattr(env, "_dribbling_blended_pelvis_target_velocity_xy", blended_target_velocity_xy)
    setattr(env, "_dribbling_ball_closing_speed", closing_speed)
    return state


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
    desired_forward_offset: float = 0.45,
    prediction_horizon: float = 0.20,
    position_forward_std: float = 0.22,
    position_lateral_std: float = 0.18,
    recovery_forward_half_width: float = 0.16,
    recovery_lateral_half_width: float = 0.12,
    recovery_transition_width: float = 0.16,
    recovery_gate_filter_steps: int = 4,
    position_correction_gain: float = 1.5,
    max_position_correction_speed: float = 0.45,
    max_recovery_target_speed: float = 2.20,
    velocity_ema_window_steps: int = 10,
) -> torch.Tensor:
    """Smoothly track the predicted command-frame ball position.

    Unlike the old hard corridor, this term supplies a gradient on both sides
    of the target and does not depend on a contact/possession timer.  Predicting
    the relative position by a short horizon makes a fast escaping ball enter
    recovery before its current position alone crosses the boundary.
    """
    state = _dribbling_closed_loop_control_state(
        env,
        command_name,
        desired_forward_offset,
        prediction_horizon,
        position_forward_std,
        position_lateral_std,
        recovery_forward_half_width,
        recovery_lateral_half_width,
        recovery_transition_width,
        recovery_gate_filter_steps,
        position_correction_gain,
        max_position_correction_speed,
        max_recovery_target_speed,
        velocity_ema_window_steps,
    )
    reward = torch.exp(-state["position_error_norm"].square())
    reward = reward * state["dribble_active"].to(reward.dtype)
    setattr(env, "_dribbling_control_position_reward", reward)
    return reward


def dribbling_closed_loop_pelvis_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.8,
    desired_forward_offset: float = 0.45,
    prediction_horizon: float = 0.20,
    position_forward_std: float = 0.22,
    position_lateral_std: float = 0.18,
    recovery_forward_half_width: float = 0.16,
    recovery_lateral_half_width: float = 0.12,
    recovery_transition_width: float = 0.16,
    recovery_gate_filter_steps: int = 4,
    position_correction_gain: float = 1.5,
    max_position_correction_speed: float = 0.45,
    max_recovery_target_speed: float = 2.20,
    velocity_ema_window_steps: int = 10,
) -> torch.Tensor:
    """Track command velocity in control and a ball-relative target in recovery."""
    if std <= 0.0:
        raise ValueError("std must be positive.")
    state = _dribbling_closed_loop_control_state(
        env,
        command_name,
        desired_forward_offset,
        prediction_horizon,
        position_forward_std,
        position_lateral_std,
        recovery_forward_half_width,
        recovery_lateral_half_width,
        recovery_transition_width,
        recovery_gate_filter_steps,
        position_correction_gain,
        max_position_correction_speed,
        max_recovery_target_speed,
        velocity_ema_window_steps,
    )
    command: MotionCommand = env.command_manager.get_term(command_name)
    target_velocity_w = torch.zeros_like(command.robot_anchor_lin_vel_w)
    target_velocity_w[:, :2] = state["blended_target_velocity_xy"]
    error = torch.sum(
        torch.square(target_velocity_w - command.robot_anchor_lin_vel_w), dim=-1
    )
    reward = torch.exp(-error / float(std) ** 2)
    setattr(env, "_dribbling_closed_loop_pelvis_velocity_reward", reward)
    return reward


def dribbling_ball_too_close_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_xy_dist: float = 0.28,
    full_penalty_dist: float = 0.14,
) -> torch.Tensor:
    """Continuous anti-trap penalty based on physical pelvis--ball XY distance.

    Command-axis projection made a ball safely beside the robot look maximally
    trapped during a turn.  Radial distance measures the actual near-body
    geometry instead: the penalty is zero at and beyond ``min_xy_dist`` and
    reaches one at ``full_penalty_dist``.  It remains independent of contact
    detection, so force cancellation cannot hide a genuinely squeezed ball.
    """
    if full_penalty_dist >= min_xy_dist:
        raise ValueError("full_penalty_dist must be smaller than min_xy_dist.")
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    forward_offset, lateral_offset = _command_frame_components(offset_xy, direction_xy)
    distance_xy = torch.norm(offset_xy, dim=-1)
    penalty = torch.clamp(
        (float(min_xy_dist) - distance_xy)
        / max(float(min_xy_dist) - float(full_penalty_dist), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    setattr(env, "_dribbling_ball_forward_offset", forward_offset)
    setattr(env, "_dribbling_ball_lateral_offset", lateral_offset)
    setattr(env, "_dribbling_ball_too_close_distance_xy", distance_xy)
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


def dribbling_command_ball_velocity_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    underspeed_tolerance: float = 0.05,
    overspeed_tolerance: float = 0.20,
    speed_error_std: float = 0.30,
    lateral_speed_std: float = 0.35,
    minimum_controllability_gate: float = 0.10,
    desired_forward_offset: float = 0.45,
    prediction_horizon: float = 0.20,
    position_forward_std: float = 0.22,
    position_lateral_std: float = 0.18,
    recovery_forward_half_width: float = 0.16,
    recovery_lateral_half_width: float = 0.12,
    recovery_transition_width: float = 0.16,
    recovery_gate_filter_steps: int = 4,
    position_correction_gain: float = 1.5,
    max_position_correction_speed: float = 0.45,
    max_recovery_target_speed: float = 2.20,
    velocity_ema_window_steps: int = 10,
) -> torch.Tensor:
    """Track commanded ball velocity with a soft physical controllability gate.

    This single term replaces the former progress lower bound and coast
    penalty.  It rewards both forward speed and direction directly.  When the
    predicted ball position leaves the controllable region, its positive score
    is reduced so the pelvis can first recover the ball instead of optimizing
    an unchangeable free-rolling velocity.
    """
    if underspeed_tolerance < 0.0 or overspeed_tolerance < 0.0:
        raise ValueError("Speed tolerances must be non-negative.")
    if speed_error_std <= 0.0 or lateral_speed_std <= 0.0:
        raise ValueError("speed_error_std and lateral_speed_std must be positive.")
    if not 0.0 <= minimum_controllability_gate <= 1.0:
        raise ValueError("minimum_controllability_gate must be in [0, 1].")

    state = _dribbling_closed_loop_control_state(
        env,
        command_name,
        desired_forward_offset,
        prediction_horizon,
        position_forward_std,
        position_lateral_std,
        recovery_forward_half_width,
        recovery_lateral_half_width,
        recovery_transition_width,
        recovery_gate_filter_steps,
        position_correction_gain,
        max_position_correction_speed,
        max_recovery_target_speed,
        velocity_ema_window_steps,
    )
    target_speed = state["command_speed"]
    ball_vel_xy = state["filtered_ball_velocity_xy"]
    forward_speed, lateral_speed = _command_frame_components(
        ball_vel_xy, state["direction_xy"]
    )

    signed_speed_error = forward_speed - target_speed
    underspeed_error = torch.relu(-signed_speed_error - float(underspeed_tolerance))
    overspeed_error = torch.relu(signed_speed_error - float(overspeed_tolerance))
    # The two errors are mutually exclusive.  A narrow lower tolerance asks
    # the policy to replace rolling losses, while the wider upper tolerance
    # permits a short post-touch speed pulse without redefining the command.
    error_outside_band = underspeed_error + overspeed_error
    forward_score = torch.exp(-error_outside_band.square() / float(speed_error_std) ** 2)
    lateral_score = torch.exp(-lateral_speed.square() / float(lateral_speed_std) ** 2)

    controllability_gate = 1.0 - (
        1.0 - float(minimum_controllability_gate)
    ) * state["recovery_gate"]
    reward = forward_score * lateral_score * controllability_gate
    reward = reward * state["dribble_active"].to(reward.dtype)
    setattr(env, "_dribbling_ball_speed_target", target_speed)
    setattr(env, "_dribbling_ball_speed_error", signed_speed_error)
    setattr(env, "_dribbling_ball_velocity_underspeed_error", underspeed_error)
    setattr(env, "_dribbling_ball_velocity_overspeed_error", overspeed_error)
    setattr(env, "_dribbling_ball_velocity_underspeed_tolerance", float(underspeed_tolerance))
    setattr(env, "_dribbling_ball_velocity_overspeed_tolerance", float(overspeed_tolerance))
    setattr(env, "_dribbling_filtered_ball_velocity_xy", ball_vel_xy)
    setattr(env, "_dribbling_filtered_ball_forward_speed", forward_speed)
    setattr(env, "_dribbling_filtered_ball_lateral_speed", lateral_speed)
    setattr(env, "_dribbling_filtered_ball_speed_error", signed_speed_error)
    setattr(env, "_dribbling_ball_velocity_ema_window_steps", int(velocity_ema_window_steps))
    setattr(env, "_dribbling_ball_velocity_controllability_gate", controllability_gate)
    setattr(
        env,
        "_dribbling_ball_velocity_heading_error",
        torch.atan2(lateral_speed, forward_speed),
    )
    setattr(env, "_dribbling_ball_velocity_tracking_reward", reward)
    return reward


def dribbling_ball_xy_speed_excess_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    speed_margin: float = 0.30,
    min_speed_cap: float = 0.35,
    huber_scale: float = 0.45,
    max_penalty: float = 6.0,
) -> torch.Tensor:
    """Command-relative, non-saturating safety penalty for excessive ball speed.

    The legacy fixed-cap penalty saturated at one, giving zero gradient above
    2.55 m/s under the active configuration.  A Huber tail keeps a finite,
    direct speed-reduction gradient throughout the relevant high-speed range,
    while ``max_penalty`` bounds the worst early-training contribution.
    """
    if huber_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("huber_scale and max_penalty must be positive.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    _, target_speed = _command_direction_xy(command)
    ball_speed = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    speed_cap = torch.maximum(
        torch.full_like(target_speed, float(min_speed_cap)),
        target_speed + float(speed_margin),
    )
    normalized_excess = torch.relu(ball_speed - speed_cap) / float(huber_scale)
    penalty = torch.where(
        normalized_excess <= 1.0,
        0.5 * normalized_excess.square(),
        normalized_excess - 0.5,
    )
    penalty = torch.clamp(penalty, max=float(max_penalty))
    active = locomotion_task_state_mask(command, (TASK_STATE_DRIBBLE,))
    penalty = penalty * active.to(penalty.dtype)

    setattr(env, "_dribbling_ball_speed_target", target_speed)
    setattr(env, "_dribbling_ball_speed_cap", speed_cap)
    setattr(env, "_dribbling_ball_speed_excess_penalty", penalty)
    return penalty


# ---------------------------------------------------------------------------
# Useful gentle foot touch.
# ---------------------------------------------------------------------------

def _event_score_to_reward_rate(env: ManagerBasedRLEnv, score: torch.Tensor) -> torch.Tensor:
    """Convert a dimensionless event score to a reward rate.

    IsaacLab multiplies every reward term by the control-step duration.  Event
    terms occur on one step only, so returning ``score / step_dt`` makes their
    integrated contribution independent of control frequency.
    """
    step_dt = max(float(getattr(env, "step_dt", 0.02)), 1.0e-6)
    return score / step_dt


def dribbling_useful_foot_touch(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    evaluation_delay_steps: int = 5,
    base_touch_score: float = 0.25,
    improvement_scale: float = 0.50,
    contact_window_tolerance_steps: int = 2,
    off_window_reward_scale: float = 0.35,
    touch_speed_error_std: float = 0.30,
    touch_lateral_speed_std: float = 0.35,
    desired_forward_offset: float = 0.45,
    prediction_horizon: float = 0.20,
    position_forward_std: float = 0.22,
    position_lateral_std: float = 0.18,
    recovery_forward_half_width: float = 0.16,
    recovery_lateral_half_width: float = 0.12,
    recovery_transition_width: float = 0.16,
    recovery_gate_filter_steps: int = 4,
    position_correction_gain: float = 1.5,
    max_position_correction_speed: float = 0.45,
    max_recovery_target_speed: float = 2.20,
    velocity_ema_window_steps: int = 10,
) -> torch.Tensor:
    """Reward a gentle right-foot touch, with an extra control-improvement bonus.

    A new touch stores the position--velocity error from the preceding control
    step and immediately receives ``base_touch_score``.  After a short
    impact-settling delay, the remaining score budget is proportional to the
    decrease in that error.  A neutral light touch is therefore valid and
    rewarded, while a touch that improves control still receives the full
    score.  The base and improvement components sum to at most one per touch.
    The score is converted to a reward rate so its integrated per-event return
    is not accidentally reduced by ``step_dt``.  CG timing remains a soft
    style multiplier, not a separate reward.
    """
    if evaluation_delay_steps <= 0 or improvement_scale <= 0.0:
        raise ValueError("touch delay and improvement scale must be positive.")
    if not 0.0 <= base_touch_score <= 1.0:
        raise ValueError("base_touch_score must be in [0, 1].")
    if not 0.0 <= off_window_reward_scale <= 1.0:
        raise ValueError("off_window_reward_scale must be in [0, 1].")
    if touch_speed_error_std <= 0.0 or touch_lateral_speed_std <= 0.0:
        raise ValueError("touch velocity error std values must be positive.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    gentle = force_mag <= force_threshold
    touch = has_contact & is_ankle & gentle

    control_state = _dribbling_closed_loop_control_state(
        env,
        command_name,
        desired_forward_offset,
        prediction_horizon,
        position_forward_std,
        position_lateral_std,
        recovery_forward_half_width,
        recovery_lateral_half_width,
        recovery_transition_width,
        recovery_gate_filter_steps,
        position_correction_gain,
        max_position_correction_speed,
        max_recovery_target_speed,
        velocity_ema_window_steps,
    )
    forward_speed, lateral_speed = _command_frame_components(
        control_state["filtered_ball_velocity_xy"], control_state["direction_xy"]
    )
    velocity_error_norm = torch.sqrt(
        ((forward_speed - control_state["command_speed"]) / float(touch_speed_error_std)).square()
        + (lateral_speed / float(touch_lateral_speed_std)).square()
    )
    control_error = torch.sqrt(
        control_state["position_error_norm"].square()
        + 0.5 * velocity_error_norm.square()
    )

    active = control_state["dribble_active"]
    labeled = command.motion_has_dribble_cg_label
    cg_window = _dribbling_cg_contact_window_ref(
        command, contact_window_tolerance_steps
    ) & active & labeled
    cg_aligned = ~labeled | cg_window
    event_scale = torch.where(
        cg_aligned,
        torch.ones_like(control_error),
        torch.full_like(control_error, float(off_window_reward_scale)),
    )

    step_buf = getattr(env, "episode_length_buf", None)
    if not isinstance(step_buf, torch.Tensor):
        step_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    state_name = "_dribbling_useful_touch_state"
    state = getattr(env, state_name, None)
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("last_step"), torch.Tensor)
        or state["last_step"].shape[0] != env.num_envs
    ):
        state = {
            "last_step": torch.full_like(step_buf, -1),
            "previous_touch": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
            "pending": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
            "remaining": torch.zeros(env.num_envs, device=env.device, dtype=torch.int32),
            "pre_error": control_error.detach().clone(),
            "last_error": control_error.detach().clone(),
            "event_scale": torch.ones_like(control_error),
            "base_reward": torch.zeros_like(control_error),
            "improvement_reward": torch.zeros_like(control_error),
            "reward": torch.zeros_like(control_error),
        }

    update_mask = state["last_step"] != step_buf
    reset_mask = (step_buf == 0) | (step_buf < state["last_step"])
    new_touch = touch & ~state["previous_touch"] & active & ~reset_mask
    matured = state["pending"] & (state["remaining"] <= 1) & ~reset_mask
    improvement = torch.relu(state["pre_error"] - control_error)
    immediate_base_reward = (
        float(base_touch_score)
        * event_scale
        * new_touch.to(control_error.dtype)
    )
    delayed_improvement_reward = (
        (1.0 - float(base_touch_score))
        * torch.clamp(improvement / float(improvement_scale), max=1.0)
        * state["event_scale"]
        * matured.to(control_error.dtype)
    )
    candidate_reward = immediate_base_reward + delayed_improvement_reward
    base_reward = torch.where(update_mask, immediate_base_reward, state["base_reward"])
    improvement_reward = torch.where(
        update_mask,
        delayed_improvement_reward,
        state["improvement_reward"],
    )
    reward = torch.where(update_mask, candidate_reward, state["reward"])

    next_pending = state["pending"] & ~matured
    next_remaining = torch.where(
        state["pending"],
        torch.clamp(state["remaining"] - 1, min=0),
        state["remaining"],
    )
    next_pending = torch.where(new_touch, torch.ones_like(next_pending), next_pending)
    next_remaining = torch.where(
        new_touch,
        torch.full_like(next_remaining, int(evaluation_delay_steps)),
        next_remaining,
    )
    next_pre_error = torch.where(new_touch, state["last_error"], state["pre_error"])
    next_event_scale = torch.where(new_touch, event_scale, state["event_scale"])

    next_pending = torch.where(reset_mask, torch.zeros_like(next_pending), next_pending)
    next_remaining = torch.where(reset_mask, torch.zeros_like(next_remaining), next_remaining)
    next_pre_error = torch.where(reset_mask, control_error, next_pre_error)
    next_event_scale = torch.where(reset_mask, torch.ones_like(next_event_scale), next_event_scale)
    base_reward = torch.where(reset_mask, torch.zeros_like(base_reward), base_reward)
    improvement_reward = torch.where(
        reset_mask,
        torch.zeros_like(improvement_reward),
        improvement_reward,
    )
    reward = torch.where(reset_mask, torch.zeros_like(reward), reward)

    state["last_step"] = torch.where(update_mask, step_buf, state["last_step"])
    state["previous_touch"] = torch.where(update_mask, touch, state["previous_touch"])
    state["pending"] = torch.where(update_mask, next_pending, state["pending"])
    state["remaining"] = torch.where(update_mask, next_remaining, state["remaining"])
    state["pre_error"] = torch.where(update_mask, next_pre_error, state["pre_error"])
    state["last_error"] = torch.where(update_mask, control_error, state["last_error"])
    state["event_scale"] = torch.where(update_mask, next_event_scale, state["event_scale"])
    state["base_reward"] = base_reward.detach().clone()
    state["improvement_reward"] = improvement_reward.detach().clone()
    state["reward"] = reward.detach().clone()
    setattr(env, state_name, state)

    setattr(env, "_dribbling_cg_contact_window_active", cg_window)
    setattr(env, "_dribbling_useful_touch_new_event", new_touch)
    setattr(env, "_dribbling_useful_touch_evaluated", matured)
    setattr(env, "_dribbling_useful_touch_pending", state["pending"])
    setattr(env, "_dribbling_useful_touch_pre_error", state["pre_error"])
    setattr(env, "_dribbling_useful_touch_current_error", control_error)
    setattr(env, "_dribbling_useful_touch_improvement", improvement)
    setattr(env, "_dribbling_useful_touch_cg_aligned", cg_aligned)
    setattr(env, "_dribbling_useful_touch_base_reward", base_reward)
    setattr(env, "_dribbling_useful_touch_improvement_reward", improvement_reward)
    reward_rate = _event_score_to_reward_rate(env, reward)
    setattr(env, "_dribbling_useful_touch_event_score", reward)
    setattr(env, "_dribbling_useful_touch_reward", reward_rate)
    return reward_rate


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
    reset_mask = (
        step_buf == 0
        if step_buf is not None
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
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
    max_reference_distance: float = 0.55,
    left_ankle_body_name: str = "left_ankle_roll_link",
    right_ankle_body_name: str = "right_ankle_roll_link",
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Match sim foot–ball distance to demo distance from synthesized CG trajectory.

    Requires ``dribble_cg_foot_ball_dist`` in motion ``.npz`` (see
    ``scripts/dribble/synthesize_dribble_ball_traj.py``). It is active only in
    the reference approach/contact window.  Long free-flight gaps no longer
    reward the foot for hovering at a demo-specific distance while the closed
    loop is trying to recover the ball.

    - ``ref_dist`` = demo XY distance from the reference foot to synthesized ball.
    - Reference foot comes from ``dribble_cg_dist_foot`` when present, else
      falls back to ``dribble_cg_foot`` (legacy contact-only labels).
    - ``sim_dist`` = distance from that foot to the **sim** ball.
    - reward = ``exp(-(sim_dist - ref_dist)^2 / std^2)``.
    """
    if std <= 0.0 or max_reference_distance <= 0.0:
        raise ValueError("std and max_reference_distance must be positive.")
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
    active = (
        labeled
        & (ref_dist >= 0.0)
        & (ref_dist <= float(max_reference_distance))
        & (ref_foot >= 0)
        & locomotion_task_state_mask(command, active_task_states)
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


def dribbling_rapid_retouch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    min_steps_between_touches: int = 22,
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
) -> torch.Tensor:
    """Penalty for a **new** legal gentle touch sooner than ``min_steps_between_touches``.

    Encourages kick → chase → kick cadence instead of tapping the ball every
    step.  The one-step event score is converted to a reward rate so the
    integrated penalty does not depend on the control-step duration.
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
    reset_mask = (
        step_buf == 0
        if step_buf is not None
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    steps_since = torch.where(
        reset_mask,
        torch.full_like(steps_since, min_steps_between_touches + 1),
        steps_since + 1,
    )

    new_touch = touch & ~prev_touch
    too_soon = new_touch & (steps_since < int(min_steps_between_touches))
    steps_since = torch.where(new_touch, torch.zeros_like(steps_since), steps_since)

    event_score = too_soon.to(torch.float32)
    reward_rate = _event_score_to_reward_rate(env, event_score)
    setattr(env, prev_touch_name, touch.detach().clone())
    setattr(env, steps_name, steps_since)
    setattr(env, "_dribbling_rapid_retouch_event", too_soon)
    setattr(env, "_dribbling_rapid_retouch_reward", reward_rate)
    return reward_rate
