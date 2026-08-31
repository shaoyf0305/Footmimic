"""Compatibility loader for checkpoints created before the unified input interface."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner as BaseOnPolicyRunner


UPPER_BODY_ACTION_SEMANTICS_KEY = "upper_body_action_semantics"
UPPER_BODY_REFERENCE_RESIDUAL_V1 = "reference_residual_v1"
UPPER_BODY_BOUNDED_REFERENCE_RESIDUAL_V2 = "bounded_reference_residual_v2"


def _get_alg_policy(runner):
    """Resolve policy module across rsl_rl versions (``alg.policy`` or legacy ``alg.actor_critic``)."""
    alg = runner.alg
    if hasattr(alg, "policy"):
        return alg.policy
    if hasattr(alg, "actor_critic"):
        return alg.actor_critic
    raise AttributeError(
        f"Unsupported algorithm type {type(alg)!r}: expected ``policy`` or ``actor_critic`` attribute."
    )


def _reference_upper_residual_layout(runner) -> tuple[list[int], int] | None:
    """Return arm action rows when the environment uses the 29-D residual interface."""
    env = getattr(runner, "env", None)
    base_env = getattr(env, "unwrapped", env)
    try:
        action_term = base_env.action_manager.get_term("joint_pos")
    except (AttributeError, KeyError):
        return None
    if not getattr(action_term, "uses_reference_relative_upper_body_residual", False):
        return None
    upper_ids = getattr(action_term, "_upper_action_ids", None)
    if not isinstance(upper_ids, torch.Tensor):
        return None
    return [int(index) for index in upper_ids.detach().cpu().tolist()], int(action_term.action_dim)


def _checkpoint_upper_body_action_semantics(loaded_dict: dict[str, Any]) -> str | None:
    semantics = loaded_dict.get(UPPER_BODY_ACTION_SEMANTICS_KEY)
    if isinstance(semantics, str):
        return semantics
    infos = loaded_dict.get("infos")
    if isinstance(infos, dict):
        semantics = infos.get(UPPER_BODY_ACTION_SEMANTICS_KEY)
        if isinstance(semantics, str):
            return semantics
    return None


def checkpoint_infos_with_action_semantics(runner, infos):
    """Stamp newly saved residual-action checkpoints for safe future resumes."""
    if _reference_upper_residual_layout(runner) is None:
        return infos
    if infos is None:
        stamped_infos: dict[str, Any] = {}
    elif isinstance(infos, dict):
        stamped_infos = dict(infos)
    else:
        raise TypeError(f"Checkpoint infos must be a dict or None, got {type(infos)!r}.")
    stamped_infos[UPPER_BODY_ACTION_SEMANTICS_KEY] = UPPER_BODY_BOUNDED_REFERENCE_RESIDUAL_V2
    return stamped_infos


# Current observations retain polar ball position, polar locomotion command,
# and command-relative polar ball velocity after the stable actor/critic base
# blocks. Each legacy layout below lists the exact source ranges whose semantics
# still exist; missing current values are appended with neutral normalizer
# statistics and zero model weights.
_LEGACY_OBSERVATION_KEEP_RANGES: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    # Previous active actor layout (160) appends ball_velocity_polar(3)
    # through the generic expansion path. Older layouts additionally need
    # obsolete Cartesian fields removed before the velocity tail is appended.
    (163, 160): ((0, 154), (160, 163)),
    (169, 160): ((0, 154), (157, 160), (166, 169)),
    (172, 160): ((0, 154), (160, 163), (169, 172)),
    (169, 163): ((0, 154), (157, 160), (166, 169)),
    (172, 163): ((0, 154), (160, 163), (169, 172)),
    # Previous active critic layout (292) appends ball_velocity_polar(3)
    # generically; these entries cover older Cartesian layouts.
    (295, 292): ((0, 286), (292, 295)),
    (301, 292): ((0, 286), (289, 292), (298, 301)),
    (304, 292): ((0, 286), (292, 295), (301, 304)),
    (301, 295): ((0, 286), (289, 292), (298, 301)),
    (304, 295): ((0, 286), (292, 295), (301, 304)),
}


def _migrate_obs_axis(old: torch.Tensor, cur: torch.Tensor, *, axis: int, fill: float) -> torch.Tensor:
    """Select still-active legacy observations and append any newly introduced terms."""
    if old.shape == cur.shape:
        return old
    if old.ndim != cur.ndim or axis >= old.ndim:
        raise ValueError(f"Cannot migrate observation tensor {old.shape} -> {cur.shape}")
    for index, (old_size, cur_size) in enumerate(zip(old.shape, cur.shape)):
        if index != axis and old_size != cur_size:
            raise ValueError(f"Cannot migrate observation tensor {old.shape} -> {cur.shape}")

    old_dim = old.shape[axis]
    cur_dim = cur.shape[axis]
    keep_ranges = _LEGACY_OBSERVATION_KEEP_RANGES.get((old_dim, cur_dim))
    migrated = old
    if keep_ranges is not None:
        migrated = torch.cat(
            [old.narrow(axis, start, end - start) for start, end in keep_ranges],
            dim=axis,
        )
    elif cur_dim <= old_dim:
        raise ValueError(f"Cannot migrate observation tensor {old.shape} -> {cur.shape}")

    pad = cur_dim - migrated.shape[axis]
    if pad < 0:
        raise ValueError(f"Cannot migrate observation tensor {old.shape} -> {cur.shape}")
    if pad:
        tail_shape = list(migrated.shape)
        tail_shape[axis] = pad
        tail = torch.full(tail_shape, fill, dtype=old.dtype, device=old.device)
        migrated = torch.cat([migrated, tail], dim=axis)
    return migrated


def _migrate_2d_input_weights(old: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """Migrate input columns; newly appended observations start with zero weight."""
    if old.ndim != 2 or cur.ndim != 2:
        raise ValueError(f"Cannot migrate weight {old.shape} -> {cur.shape}")
    return _migrate_obs_axis(old, cur, axis=1, fill=0.0)


def expand_obs_normalizer_state_dict(
    checkpoint_sd: dict[str, torch.Tensor],
    current_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Migrate running mean/var to the current observation layout.

    Handles both flat ``(N,)`` and row-vector ``(1, N)`` normalizer shapes,
    including removal of legacy Cartesian ball, destination, and redundant
    Cartesian linear/angular command observations.
    """
    expanded: dict[str, torch.Tensor] = {}
    for key, cur in current_sd.items():
        if key not in checkpoint_sd:
            expanded[key] = cur.clone()
            continue
        old = checkpoint_sd[key]
        if old.shape == cur.shape:
            expanded[key] = old
            continue

        fill = 1.0 if "var" in key else 0.0

        # Flat 1-D: remove legacy xyz and/or append new terms.
        if old.ndim == 1 and cur.ndim == 1:
            expanded[key] = _migrate_obs_axis(old, cur, axis=0, fill=fill)
            continue

        # Row-vector 2-D: remove legacy xyz and/or append new terms.
        if old.ndim == 2 and cur.ndim == 2 and old.shape[0] == 1 and cur.shape[0] == 1:
            expanded[key] = _migrate_obs_axis(old, cur, axis=1, fill=fill)
            continue

        raise ValueError(
            f"Incompatible normalizer tensor '{key}': ckpt {tuple(old.shape)} vs env {tuple(cur.shape)}"
        )
    return expanded


def _state_dict_needs_obs_expand(checkpoint_sd: dict[str, torch.Tensor], current_sd: dict[str, torch.Tensor]) -> bool:
    for key, cur in current_sd.items():
        if key not in checkpoint_sd:
            continue
        old = checkpoint_sd[key]
        if old.shape == cur.shape:
            continue
        if old.ndim == 1 and cur.ndim == 1:
            if (old.shape[0], cur.shape[0]) in _LEGACY_OBSERVATION_KEEP_RANGES or cur.shape[0] > old.shape[0]:
                return True
        if old.ndim == 2 and cur.ndim == 2 and old.shape[0] == cur.shape[0]:
            if (old.shape[1], cur.shape[1]) in _LEGACY_OBSERVATION_KEEP_RANGES or cur.shape[1] > old.shape[1]:
                return True
        return False
    return False


def _direct_upper_latent_layout(runner) -> tuple[list[int], int] | None:
    """Return old joint-action rows and new action size for a latent-action task.

    The direct upper-body interface keeps the original lower-body action order,
    then appends PCA latent coordinates.  It is therefore possible to preserve
    the trained lower-body output rows while intentionally reinitializing only
    the new latent rows.
    """
    env = getattr(runner, "env", None)
    base_env = getattr(env, "unwrapped", env)
    try:
        action_term = base_env.action_manager.get_term("joint_pos")
    except (AttributeError, KeyError):
        return None
    if not getattr(action_term, "uses_direct_upper_body_latent", False):
        return None
    lower_ids = getattr(action_term, "_lower_action_ids", None)
    if not isinstance(lower_ids, torch.Tensor):
        return None
    return [int(index) for index in lower_ids.detach().cpu().tolist()], int(action_term.action_dim)


def _migrate_direct_latent_action_output(
    old: torch.Tensor,
    cur: torch.Tensor,
    *,
    key: str,
    lower_action_ids: list[int],
) -> torch.Tensor:
    """Warm-start a 29-D joint-action head into a 21-D latent-action head.

    Lower-body action rows are copied exactly.  The six new upper-body latent
    rows are initialized at zero mean (and conservative exploration variance),
    which decodes to the current reference pose.  A fixed linear conversion of
    old arm targets is deliberately avoided: those targets were absolute and
    reference-clamped, whereas the new coordinates are reference-relative.
    """
    if old.ndim != cur.ndim or old.shape[0] <= max(lower_action_ids, default=-1):
        raise ValueError(f"Cannot migrate direct-latent action tensor {key}: {old.shape} -> {cur.shape}")
    if old.ndim == 2 and old.shape[1] != cur.shape[1]:
        raise ValueError(f"Cannot migrate direct-latent action tensor {key}: {old.shape} -> {cur.shape}")

    migrated = cur.clone()
    lower_count = len(lower_action_ids)
    migrated[:lower_count] = old[lower_action_ids]
    if old.ndim == 1:
        # ``std`` is the only 1-D action parameter.  Start the new latent
        # coordinates with moderate exploration so tanh does not saturate.
        migrated[lower_count:] = 0.5
    else:
        # Actor-output weights and biases for latent rows must decode to the
        # reference pose at the first step of fine-tuning.
        migrated[lower_count:] = 0.0
    return migrated


# STAGE-I TO STAGE-II ACTION MIGRATION --------------------------------------
# A Stage-I checkpoint has no residual-action marker. Initializing the Stage-II
# controller therefore keeps the compatible network state and resets only the
# 14 output rows whose semantics change to bounded reference-relative residuals.
def _is_legacy_joint_action_output(
    key: str,
    old: torch.Tensor,
    cur: torch.Tensor,
    *,
    action_dim: int,
) -> bool:
    """Identify actor-head rows without touching hidden layers or the critic."""
    if old.shape != cur.shape or old.ndim not in (1, 2) or old.shape[0] != action_dim:
        return False
    if key == "std" or key.endswith(".std"):
        return old.ndim == 1
    return key.startswith("actor.") and (
        (old.ndim == 1 and key.endswith(".bias"))
        or (old.ndim == 2 and key.endswith(".weight"))
    )


def _migrate_legacy_upper_body_output_to_residual(
    old: torch.Tensor,
    *,
    key: str,
    upper_action_ids: list[int],
) -> torch.Tensor:
    """Reset only the 14 arm rows incompatible with the bounded residual policy."""
    migrated = old.clone()
    if key == "std" or key.endswith(".std"):
        migrated[upper_action_ids] = 0.5
    elif old.ndim == 1:
        migrated[upper_action_ids] = 0.0
    else:
        migrated[upper_action_ids, :] = 0.0
    return migrated
# END STAGE-I TO STAGE-II ACTION MIGRATION ----------------------------------


def _maybe_expand_normalizer_entry(loaded_dict: dict[str, Any], key: str, current_sd: dict[str, torch.Tensor]) -> bool:
    if key not in loaded_dict:
        return False
    ckpt_sd = loaded_dict[key]
    if not _state_dict_needs_obs_expand(ckpt_sd, current_sd):
        return False
    loaded_dict[key] = expand_obs_normalizer_state_dict(ckpt_sd, current_sd)
    return True


def prepare_loaded_dict_for_obs_expand(
    loaded_dict: dict[str, Any],
    runner,
    *,
    migrate_legacy_upper_body_residual: bool = False,
    migrate_bounded_upper_body_policy: bool = False,
) -> bool:
    """Adapt legacy observation layouts and joint-action checkpoints in place."""
    policy = _get_alg_policy(runner)
    model_key = "model_state_dict"
    if model_key not in loaded_dict:
        return False

    ckpt_sd = loaded_dict[model_key]
    current_sd = policy.state_dict()
    layout = _direct_upper_latent_layout(runner)
    residual_layout = _reference_upper_residual_layout(runner)
    checkpoint_semantics = _checkpoint_upper_body_action_semantics(loaded_dict)
    migrate_residual_output = False
    if migrate_legacy_upper_body_residual and migrate_bounded_upper_body_policy:
        raise ValueError(
            "Choose exactly one upper-body migration flag; they represent different source checkpoint semantics."
        )
    if residual_layout is None:
        if migrate_legacy_upper_body_residual or migrate_bounded_upper_body_policy:
            raise ValueError(
                "An upper-body migration was requested, but this task does not use "
                "reference-relative upper-body residual actions."
            )
    elif checkpoint_semantics == UPPER_BODY_BOUNDED_REFERENCE_RESIDUAL_V2:
        if migrate_legacy_upper_body_residual or migrate_bounded_upper_body_policy:
            raise ValueError(
                "This checkpoint already uses bounded_reference_residual_v2. Remove the migration flag; "
                "normal resume must never reset the arm head."
            )
    elif checkpoint_semantics == UPPER_BODY_REFERENCE_RESIDUAL_V1:
        if not migrate_bounded_upper_body_policy:
            raise RuntimeError(
                "This Stage-2 checkpoint uses the unbounded reference_residual_v1 policy. Load it once with "
                "--migrate_bounded_upper_body_policy to retain the lower body while resetting the 14 saturated "
                "arm output rows. Do not reuse that flag after the converted run saves a checkpoint."
            )
        migrate_residual_output = True
    elif checkpoint_semantics is None:
        if migrate_bounded_upper_body_policy:
            raise ValueError(
                "A checkpoint without an action-semantics marker is pre-residual. Use "
                "--migrate_legacy_upper_body_residual instead of --migrate_bounded_upper_body_policy."
            )
        if not migrate_legacy_upper_body_residual:
            raise RuntimeError(
                "This Stage-2 task expects bounded_reference_residual_v2 arm actions, but the checkpoint has "
                "no residual-action marker. For a pre-residual checkpoint, load it once with "
                "--migrate_legacy_upper_body_residual. Do not use that flag for checkpoints saved "
                "after migration."
            )
        migrate_residual_output = True
    else:
        raise ValueError(
            f"Unsupported upper-body action semantics {checkpoint_semantics!r}; expected "
            f"{UPPER_BODY_BOUNDED_REFERENCE_RESIDUAL_V2!r}."
        )
    old_action_dim = int(ckpt_sd["std"].numel()) if isinstance(ckpt_sd.get("std"), torch.Tensor) else None
    new_action_dim = int(current_sd["std"].numel()) if isinstance(current_sd.get("std"), torch.Tensor) else None

    transformed_sd: dict[str, torch.Tensor] = {}
    notes: list[str] = []
    changed = False
    migrated_residual_keys: list[str] = []
    for key, cur in current_sd.items():
        if key not in ckpt_sd:
            transformed_sd[key] = cur.clone()
            notes.append(f"keep init: {key}")
            changed = True
            continue

        old = ckpt_sd[key]
        is_residual_action_output = (
            migrate_residual_output
            and residual_layout is not None
            and _is_legacy_joint_action_output(
                key,
                old,
                cur,
                action_dim=residual_layout[1],
            )
        )
        if is_residual_action_output:
            upper_action_ids, _ = residual_layout
            transformed_sd[key] = _migrate_legacy_upper_body_output_to_residual(
                old,
                key=key,
                upper_action_ids=upper_action_ids,
            )
            migrated_residual_keys.append(key)
            notes.append(f"{key}: reset 14 incompatible arm rows for bounded residual actions")
            changed = True
            continue

        if old.shape == cur.shape:
            transformed_sd[key] = old
            continue

        is_action_output = (
            layout is not None
            and old_action_dim is not None
            and new_action_dim is not None
            and old.shape[0] == old_action_dim
            and cur.shape[0] == new_action_dim
            and (old.ndim == 1 or (old.ndim == 2 and old.shape[1] == cur.shape[1]))
        )
        if is_action_output:
            lower_action_ids, _ = layout
            transformed_sd[key] = _migrate_direct_latent_action_output(
                old, cur, key=key, lower_action_ids=lower_action_ids
            )
            notes.append(f"{key}: migrated {old_action_dim}-D joint actions -> {new_action_dim}-D latent actions")
            changed = True
            continue

        can_migrate_obs = (
            old.ndim == 2
            and cur.ndim == 2
            and old.shape[0] == cur.shape[0]
            and (
                (old.shape[1], cur.shape[1]) in _LEGACY_OBSERVATION_KEEP_RANGES
                or cur.shape[1] > old.shape[1]
            )
        )
        if can_migrate_obs:
            transformed_sd[key] = _migrate_2d_input_weights(old, cur)
            notes.append(f"{key}: {tuple(old.shape)} -> {tuple(cur.shape)}")
            changed = True
            continue

        raise ValueError(
            f"Incompatible checkpoint tensor '{key}': ckpt {tuple(old.shape)} vs env {tuple(cur.shape)}"
        )

    if migrate_residual_output:
        has_std = any(
            key == "std" or key.endswith(".std") for key in migrated_residual_keys
        )
        has_actor_output = any(
            key.startswith("actor.") and key.endswith(".weight")
            for key in migrated_residual_keys
        )
        if not has_std or not has_actor_output:
            raise RuntimeError(
                "Could not identify both the actor output and exploration std tensors required for "
                "the one-time upper-body residual migration."
            )
        loaded_dict[UPPER_BODY_ACTION_SEMANTICS_KEY] = UPPER_BODY_BOUNDED_REFERENCE_RESIDUAL_V2
        infos = loaded_dict.get("infos")
        stamped_infos = dict(infos) if isinstance(infos, dict) else {}
        stamped_infos[UPPER_BODY_ACTION_SEMANTICS_KEY] = UPPER_BODY_BOUNDED_REFERENCE_RESIDUAL_V2
        loaded_dict["infos"] = stamped_infos

    if not changed:
        return False

    loaded_dict[model_key] = transformed_sd
    loaded_dict.pop("optimizer_state_dict", None)
    loaded_dict.pop("rnd_optimizer_state_dict", None)

    print("[INFO] Warm-start: adapted a legacy checkpoint to the current interface.")
    if migrate_residual_output:
        print(
            "[INFO] One-time bounded-policy migration: kept lower-body/hidden/critic weights, reset only "
            "the 14 incompatible arm output rows, and started a fresh optimizer."
        )
        print(
            "[INFO] Future resumes from checkpoints saved by this run must NOT use "
            "either upper-body migration flag."
        )
    for note in notes:
        if "->" in note:
            print(f"  {note}")

    if getattr(runner, "empirical_normalization", False):
        norm_expanded = False
        if hasattr(runner, "obs_normalizer"):
            if _maybe_expand_normalizer_entry(loaded_dict, "obs_norm_state_dict", runner.obs_normalizer.state_dict()):
                norm_expanded = True
        if hasattr(runner, "privileged_obs_normalizer"):
            if _maybe_expand_normalizer_entry(
                loaded_dict,
                "privileged_obs_norm_state_dict",
                runner.privileged_obs_normalizer.state_dict(),
            ):
                norm_expanded = True
        if norm_expanded:
            print("[INFO] Warm-start: migrated legacy obs normalizer(s) to the current input layout.")

    return True


def load_checkpoint_with_obs_expand(runner, path: str, **load_kwargs) -> Any:
    """Load a checkpoint and migrate legacy observation/action interfaces."""
    map_location = load_kwargs.pop("map_location", getattr(runner, "device", None))
    migrate_legacy_upper_body_residual = bool(
        load_kwargs.pop("migrate_legacy_upper_body_residual", False)
    )
    migrate_bounded_upper_body_policy = bool(
        load_kwargs.pop("migrate_bounded_upper_body_policy", False)
    )
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    expanded = prepare_loaded_dict_for_obs_expand(
        loaded_dict,
        runner,
        migrate_legacy_upper_body_residual=migrate_legacy_upper_body_residual,
        migrate_bounded_upper_body_policy=migrate_bounded_upper_body_policy,
    )

    if not expanded:
        return BaseOnPolicyRunner.load(runner, path, **load_kwargs)

    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(loaded_dict, tmp_path)
        # Fresh optimizer after interface migration; do not restore old Adam state.
        load_kwargs.setdefault("load_optimizer", False)
        return BaseOnPolicyRunner.load(runner, tmp_path, **load_kwargs)
    finally:
        os.remove(tmp_path)
