#!/usr/bin/env python3
"""Slice a motion .npz file into approach and strike segments.

Usage
-----
python scripts/slice_motion.py motions/Video/hmr4d_1_unitree_g1_compatible_right.npz

This reads ``kick_frame`` (kick start) from the npz metadata and produces:
  - ``<name>_approach.npz``  →  frames [0, kick_start_frame)
  - ``<name>_strike.npz``    →  frames [kick_start_frame, end]

If ``--kick_start_frame`` is passed on the CLI it overrides the value in the file.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def load_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_npz(path: str, payload: dict[str, np.ndarray]) -> None:
    np.savez(path, **payload)
    print(f"  → saved {path}  ({os.path.getsize(path) / 1024:.1f} KB)")


def slice_trajectories(payload: dict, start: int, end: int) -> dict:
    """Return a new payload with all trajectory arrays sliced to [start:end]."""
    out = {}
    # Keys that are per-frame trajectories (shape[0] == num_frames).
    traj_keys = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                 "body_lin_vel_w", "body_ang_vel_w"}
    num_frames = None
    for k in traj_keys:
        if k in payload:
            if num_frames is None:
                num_frames = payload[k].shape[0]
            out[k] = payload[k][start:end]

    # Copy non-trajectory keys verbatim (fps, kick_leg, etc.).
    for k, v in payload.items():
        if k not in traj_keys:
            out[k] = v

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice motion into approach + strike.")
    parser.add_argument("npz_file", help="Input motion .npz file.")
    parser.add_argument(
        "--kick_start_frame", type=int, default=None,
        help="Override the kick start frame (0-indexed). Reads from npz metadata if omitted.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory. Defaults to same directory as input.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.npz_file):
        sys.exit(f"[ERROR] File not found: {args.npz_file}")

    payload = load_npz(args.npz_file)

    # Determine kick start frame.
    kick_start = args.kick_start_frame
    if kick_start is None:
        if "kick_frame" in payload:
            kick_start = int(np.asarray(payload["kick_frame"]).flat[0])
        else:
            sys.exit("[ERROR] No --kick_start_frame and no 'kick_frame' in npz metadata.")

    # Get total frames from any trajectory key.
    for k in ("joint_pos", "joint_vel", "body_pos_w"):
        if k in payload:
            total_frames = payload[k].shape[0]
            break
    else:
        sys.exit("[ERROR] No trajectory keys (joint_pos, joint_vel, body_pos_w) found in npz.")

    if kick_start <= 0 or kick_start >= total_frames:
        sys.exit(f"[ERROR] kick_start_frame={kick_start} out of range [1, {total_frames - 1}].")

    print(f"[INFO] Total frames: {total_frames},  kick_start: {kick_start}")
    print(f"[INFO] Approach: frames [0, {kick_start}),  Strike: frames [{kick_start}, {total_frames})")

    # Output paths.
    out_dir = args.output_dir or os.path.dirname(args.npz_file)
    basename = os.path.splitext(os.path.basename(args.npz_file))[0]
    approach_path = os.path.join(out_dir, f"{basename}_approach.npz")
    strike_path = os.path.join(out_dir, f"{basename}_strike.npz")

    # Slice and save.
    approach = slice_trajectories(payload, 0, kick_start)
    # Remove kick metadata from approach (no kick in approach).
    approach.pop("kick_frame", None)
    approach.pop("kick_end_frame", None)

    strike = slice_trajectories(payload, kick_start, total_frames)
    # Update kick metadata: in the strike segment, kick happens at frame 0.
    if "kick_frame" in payload:
        strike["kick_frame"] = np.array(0, dtype=np.int32)
    if "kick_end_frame" in payload:
        orig_end = int(np.asarray(payload["kick_end_frame"]).flat[0])
        strike["kick_end_frame"] = np.array(max(0, orig_end - kick_start), dtype=np.int32)

    save_npz(approach_path, approach)
    save_npz(strike_path, strike)
    print("[DONE]")


if __name__ == "__main__":
    main()
