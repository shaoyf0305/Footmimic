#!/usr/bin/env python3
"""Export the preserved v4.4 Stage-2 control checkpoint to a small ONNX actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v44_common import (
    ACTION_DIM,
    DEFAULT_CHECKPOINT,
    DEFAULT_JOINT_POS_ISAAC,
    ISAACLAB_JOINT_NAMES,
    OBS_DIM,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export model_93000.pt from the v4.4 control run without starting Isaac Sim."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path (default: <run>/exported/policy_93000_v44_control.onnx).",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        import onnx
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "This exporter needs torch and onnx. Run it in the isaaclab_211 environment."
        ) from exc

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output = (
        args.output.resolve()
        if args.output is not None
        else checkpoint.parent
        / "exported"
        / f"policy_{checkpoint.stem.removeprefix('model_')}_v44_control.onnx"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    norm_state = payload.get("obs_norm_state_dict")
    if norm_state is None:
        raise KeyError("Checkpoint has no obs_norm_state_dict; v4.4 requires empirical normalization.")

    class V44Actor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.LSTM(
                input_size=OBS_DIM,
                hidden_size=RNN_HIDDEN_DIM,
                num_layers=RNN_NUM_LAYERS,
            )
            self.actor = nn.Sequential(
                nn.Linear(RNN_HIDDEN_DIM, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 32),
                nn.ELU(),
                nn.Linear(32, ACTION_DIM),
            )
            self.register_buffer("obs_mean", norm_state["_mean"].detach().clone())
            self.register_buffer("obs_std", norm_state["_std"].detach().clone())

            actor_state = {
                key.removeprefix("actor."): value
                for key, value in state.items()
                if key.startswith("actor.")
            }
            rnn_state = {
                key.removeprefix("memory_a.rnn."): value
                for key, value in state.items()
                if key.startswith("memory_a.rnn.")
            }
            self.actor.load_state_dict(actor_state, strict=True)
            self.rnn.load_state_dict(rnn_state, strict=True)

        def forward(self, obs, h_in, c_in):
            normalized = (obs - self.obs_mean) / (self.obs_std + 1.0e-2)
            recurrent, (h_out, c_out) = self.rnn(
                normalized.unsqueeze(0),
                (h_in, c_in),
            )
            actions = self.actor(recurrent.squeeze(0))
            return actions, h_out, c_out

    actor = V44Actor().eval()
    obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    h_in = torch.zeros(RNN_NUM_LAYERS, 1, RNN_HIDDEN_DIM, dtype=torch.float32)
    c_in = torch.zeros_like(h_in)

    with torch.inference_mode():
        torch.onnx.export(
            actor,
            (obs, h_in, c_in),
            str(output),
            export_params=True,
            opset_version=int(args.opset),
            do_constant_folding=True,
            input_names=["obs", "h_in", "c_in"],
            output_names=["actions", "h_out", "c_out"],
            dynamic_axes=None,
        )

    model = onnx.load(str(output))
    metadata = {
        "deployment_recipe": "Footmimic v4.4 Stage2 control",
        "source_checkpoint": str(checkpoint),
        "checkpoint_iteration": str(payload.get("iter", "")),
        "obs_dim": str(OBS_DIM),
        "action_dim": str(ACTION_DIM),
        "rnn_num_layers": str(RNN_NUM_LAYERS),
        "rnn_hidden_dim": str(RNN_HIDDEN_DIM),
        "normalization_eps": "0.01",
        "joint_names": json.dumps(ISAACLAB_JOINT_NAMES),
        "default_joint_pos": json.dumps(DEFAULT_JOINT_POS_ISAAC.tolist()),
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.checker.check_model(model)
    onnx.save(model, str(output))

    print(f"[OK] Exported: {output}")
    print(f"[OK] inputs: obs=(1,{OBS_DIM}), h/c=({RNN_NUM_LAYERS},1,{RNN_HIDDEN_DIM})")
    print(f"[OK] outputs: actions=(1,{ACTION_DIM}), h_out, c_out")


if __name__ == "__main__":
    main()
