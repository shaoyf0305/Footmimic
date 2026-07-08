"""Checkpoint helpers for resuming across envs with extra observation terms."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner as BaseOnPolicyRunner


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


def _expand_2d_input_weights(old: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """Append zero columns when input features grew (obs terms appended at end)."""
    if old.shape == cur.shape:
        return old
    if old.ndim != 2 or cur.ndim != 2 or old.shape[0] != cur.shape[0] or cur.shape[1] <= old.shape[1]:
        raise ValueError(f"Cannot expand weight {old.shape} -> {cur.shape}")
    pad = cur.shape[1] - old.shape[1]
    zeros = torch.zeros(old.shape[0], pad, dtype=old.dtype, device=old.device)
    return torch.cat([old, zeros], dim=1)


def _expand_1d_tail(old: torch.Tensor, cur: torch.Tensor, *, fill: float) -> torch.Tensor:
    if old.shape == cur.shape:
        return old
    if old.ndim != 1 or cur.ndim != 1 or cur.shape[0] <= old.shape[0]:
        raise ValueError(f"Cannot expand vector {old.shape} -> {cur.shape}")
    pad = cur.shape[0] - old.shape[0]
    tail = torch.full((pad,), fill, dtype=old.dtype, device=old.device)
    return torch.cat([old, tail], dim=0)


def expand_state_dict_for_obs_growth(
    checkpoint_sd: dict[str, torch.Tensor],
    current_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Pad checkpoint weights along the input-feature axis for new trailing obs dims."""
    expanded: dict[str, torch.Tensor] = {}
    notes: list[str] = []
    for key, cur in current_sd.items():
        if key not in checkpoint_sd:
            expanded[key] = cur.clone()
            notes.append(f"keep init: {key}")
            continue
        old = checkpoint_sd[key]
        if old.shape == cur.shape:
            expanded[key] = old
            continue
        if old.ndim == 2 and cur.ndim == 2:
            expanded[key] = _expand_2d_input_weights(old, cur)
            notes.append(f"{key}: {tuple(old.shape)} -> {tuple(expanded[key].shape)}")
            continue
        raise ValueError(
            f"Incompatible checkpoint tensor '{key}': ckpt {tuple(old.shape)} vs env {tuple(cur.shape)}"
        )
    return expanded, notes


def expand_obs_normalizer_state_dict(
    checkpoint_sd: dict[str, torch.Tensor],
    current_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extend running mean/var for appended observation dimensions.

    Handles both flat ``(N,)`` and row-vector ``(1, N)`` normalizer shapes.
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

        # flat 1-D: (N,) -> (N+k,)
        if old.ndim == 1 and cur.ndim == 1:
            expanded[key] = _expand_1d_tail(old, cur, fill=fill)
            continue

        # row-vector 2-D: (1, N) -> (1, N+k)
        if old.ndim == 2 and cur.ndim == 2 and old.shape[0] == 1 and cur.shape[0] == 1:
            pad = cur.shape[1] - old.shape[1]
            if pad <= 0:
                raise ValueError(
                    f"Incompatible normalizer tensor '{key}': ckpt {tuple(old.shape)} vs env {tuple(cur.shape)}"
                )
            tail = torch.full((1, pad), fill, dtype=old.dtype, device=old.device)
            expanded[key] = torch.cat([old, tail], dim=1)
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
        if old.ndim == 2 and cur.ndim == 2 and old.shape[0] == cur.shape[0] and cur.shape[1] > old.shape[1]:
            return True
        return False
    return False


def _maybe_expand_normalizer_entry(loaded_dict: dict[str, Any], key: str, current_sd: dict[str, torch.Tensor]) -> bool:
    if key not in loaded_dict:
        return False
    ckpt_sd = loaded_dict[key]
    if not _state_dict_needs_obs_expand(ckpt_sd, current_sd):
        return False
    loaded_dict[key] = expand_obs_normalizer_state_dict(ckpt_sd, current_sd)
    return True


def prepare_loaded_dict_for_obs_expand(loaded_dict: dict[str, Any], runner) -> bool:
    """Mutate ``loaded_dict`` in place when resuming a smaller-obs checkpoint. Returns True if expanded."""
    policy = _get_alg_policy(runner)
    model_key = "model_state_dict"
    if model_key not in loaded_dict:
        return False

    ckpt_sd = loaded_dict[model_key]
    current_sd = policy.state_dict()
    if not _state_dict_needs_obs_expand(ckpt_sd, current_sd):
        return False

    expanded_sd, notes = expand_state_dict_for_obs_growth(ckpt_sd, current_sd)
    loaded_dict[model_key] = expanded_sd
    loaded_dict.pop("optimizer_state_dict", None)
    loaded_dict.pop("rnd_optimizer_state_dict", None)

    print("[INFO] Warm-start: expanded checkpoint for new locomotion-cmd obs (zero-padded input weights).")
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
            print("[INFO] Warm-start: expanded obs normalizer(s) for new command inputs.")

    return True


def load_checkpoint_with_obs_expand(runner, path: str, **load_kwargs) -> Any:
    """Load a checkpoint, auto-expanding actor/critic inputs when obs grew (forward -> follow/control)."""
    map_location = load_kwargs.pop("map_location", getattr(runner, "device", None))
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    expanded = prepare_loaded_dict_for_obs_expand(loaded_dict, runner)

    if not expanded:
        return BaseOnPolicyRunner.load(runner, path, **load_kwargs)

    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(loaded_dict, tmp_path)
        # Fresh optimizer after input-dim growth; do not restore old Adam state.
        load_kwargs.setdefault("load_optimizer", False)
        return BaseOnPolicyRunner.load(runner, tmp_path, **load_kwargs)
    finally:
        os.remove(tmp_path)
