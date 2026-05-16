#!/usr/bin/env python3
"""Batch-write ``kick_frame`` / ``kick_end_frame`` into motion ``.npz`` files.

Reads a label file (same style as ``kick_labels.txt``) and patches matching NPZs
under ``motions/soccerkicks`` (or any directory you pass).

Label file format (one clip per line)::

    # comment
    clip_name:41,           # only kick start → writes kick_frame only
    clip_name:40,55,        # start + end → writes both (commas optional at end)

Numbers are **0-based frame indices** by default. If you labeled from OpenCV
with the on-screen counter starting at 1, pass ``--one-based-input`` so each
value is stored as ``value - 1``.

Matching: ``<clip_name>.npz`` first, then ``<clip_name>_*.npz`` (underscore after
the clip id so ``11_freekick`` does not pick up ``11_freekick1_...``).

python scripts/apply_kick_labels_to_npz.py --dry-run
python scripts/apply_kick_labels_to_npz.py 
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

LINE_RE = re.compile(
    r"^\s*([^:#\s]+)\s*:\s*([\d\s,]+)\s*(?:#.*)?$",
)


def parse_label_line(line: str) -> tuple[str, list[int]] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    m = LINE_RE.match(line)
    if not m:
        print(f"[WARN] skip unparsable line: {line!r}", file=sys.stderr)
        return None
    name, rest = m.group(1), m.group(2)
    parts = [p for p in (x.strip() for x in rest.split(",")) if p]
    if not parts:
        print(f"[WARN] skip line with no integers: {line!r}", file=sys.stderr)
        return None
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        print(f"[WARN] skip non-integer fields: {line!r}", file=sys.stderr)
        return None
    return name, nums


def _collect_npz_candidates(motion_dir: Path, clip: str, recursive: bool) -> list[Path]:
    """Paths whose basename is ``{clip}.npz`` or ``{clip}_*.npz`` (not ``{clip}1_...``)."""
    seen: set[Path] = set()
    out: list[Path] = []
    patterns: list[str]
    if recursive:
        patterns = [f"**/{clip}.npz", f"**/{clip}_*.npz"]
    else:
        patterns = [f"{clip}.npz", f"{clip}_*.npz"]
    for pat in patterns:
        for p in motion_dir.glob(pat):
            if p.is_file() and p.suffix.lower() == ".npz":
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    out.append(p)
    return sorted(out)


def resolve_npz(motion_dir: Path, clip: str, recursive: bool) -> tuple[Path | None, str]:
    """Return (path, err). err empty on success."""
    candidates = _collect_npz_candidates(motion_dir, clip, recursive)
    if len(candidates) == 1:
        return candidates[0], ""
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates[:8])
        more = "" if len(candidates) <= 8 else f", ... (+{len(candidates) - 8} more)"
        return None, f"ambiguous ({len(candidates)} files): {names}{more}"
    return None, "no matching .npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def apply_labels(
    labels_path: Path,
    motion_dir: Path,
    *,
    one_based_input: bool,
    dry_run: bool,
    recursive: bool,
    max_frames: dict[Path, int],
) -> int:
    errors = 0
    for raw in labels_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_label_line(raw)
        if parsed is None:
            continue
        clip, nums = parsed
        delta = 1 if one_based_input else 0
        kick_start = nums[0] - delta
        kick_end: int | None = None
        if len(nums) >= 2:
            kick_end = nums[1] - delta

        npz_path, resolve_err = resolve_npz(motion_dir, clip, recursive)
        if npz_path is None:
            print(f"[ERROR] {clip}: {resolve_err}  (under {motion_dir})", file=sys.stderr)
            errors += 1
            continue

        nmax = max_frames.get(npz_path)
        if nmax is None:
            try:
                payload = load_npz(npz_path)
            except Exception as exc:
                print(f"[ERROR] {npz_path}: load failed: {exc}", file=sys.stderr)
                errors += 1
                continue
            for key in ("joint_pos", "joint_vel", "body_pos_w"):
                if key in payload:
                    nmax = int(payload[key].shape[0])
                    break
            if nmax is None:
                print(f"[ERROR] {npz_path}: cannot infer frame count", file=sys.stderr)
                errors += 1
                continue
            max_frames[npz_path] = nmax

        if kick_start < 0 or kick_start >= nmax:
            print(
                f"[ERROR] {clip}: kick_frame={kick_start} out of range [0, {nmax - 1}]",
                file=sys.stderr,
            )
            errors += 1
            continue
        if kick_end is not None and (kick_end < 0 or kick_end >= nmax):
            print(
                f"[ERROR] {clip}: kick_end_frame={kick_end} out of range [0, {nmax - 1}]",
                file=sys.stderr,
            )
            errors += 1
            continue
        if kick_end is not None and kick_end < kick_start:
            print(
                f"[WARN] {clip}: kick_end ({kick_end}) < kick_start ({kick_start}); still writing.",
                file=sys.stderr,
            )

        if dry_run:
            end_s = f", kick_end_frame={kick_end}" if kick_end is not None else ""
            print(f"[DRY-RUN] {npz_path}  kick_frame={kick_start}{end_s}")
            continue

        payload = load_npz(npz_path)
        payload["kick_frame"] = np.array(kick_start, dtype=np.int32)
        if kick_end is not None:
            payload["kick_end_frame"] = np.array(kick_end, dtype=np.int32)
        save_npz_atomic(npz_path, payload)
        end_s = f", end={kick_end}" if kick_end is not None else ""
        print(f"[OK] {npz_path}  kick_frame={kick_start}{end_s}")

    return errors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).resolve().parent / "kick_labels.txt",
        help="Label file path (default: scripts/kick_labels.txt next to this script).",
    )
    p.add_argument(
        "--motion-dir",
        type=Path,
        default=Path("motions/soccerkicks"),
        help="Directory containing motion .npz files (default: motions/soccerkicks).",
    )
    p.add_argument(
        "--one-based-input",
        action="store_true",
        help="Subtract 1 from each parsed integer before writing (video overlay style).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search motion-dir recursively for <clip>.npz / <clip>_*.npz.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying files.",
    )
    args = p.parse_args()

    if not args.labels.is_file():
        sys.exit(f"[ERROR] label file not found: {args.labels}")
    motion_dir = args.motion_dir.resolve()
    if not motion_dir.is_dir():
        sys.exit(f"[ERROR] motion directory not found: {motion_dir}")

    errs = apply_labels(
        args.labels,
        motion_dir,
        one_based_input=args.one_based_input,
        dry_run=args.dry_run,
        recursive=args.recursive,
        max_frames={},
    )
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
