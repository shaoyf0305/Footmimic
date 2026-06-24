#!/usr/bin/env python3
"""Play an MP4 with ffplay (frame overlay) or slice it by frame indices.

Frames use 0-based, half-open intervals ``[start, end)`` — same convention as
``scripts/slice_motion.py``.

Usage
-----
# Play with absolute frame counter (note frames yourself)
python scripts/slice_video.py play temp/master.mp4

# Slice frames [120,300) and [450,600) — pairs of (start, end)
python scripts/slice_video.py slice temp/master.mp4 --cuts 120 300 450 600

# Explicit segments
python scripts/slice_video.py slice temp/master.mp4 --segments 0:120 120:300
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys


def probe_video(path: str) -> tuple[int, float]:
    """Return (frame_count, fps) via ffprobe."""
    if shutil.which("ffprobe") is None:
        sys.exit("[ERROR] ffprobe not found on PATH.")

    fps_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "csv=p=0",
        path,
    ]
    count_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
        path,
    ]

    try:
        fps_raw = subprocess.check_output(fps_cmd, text=True).strip()
        count_raw = subprocess.check_output(count_cmd, text=True).strip()
    except subprocess.CalledProcessError as exc:
        sys.exit(f"[ERROR] ffprobe failed for {path}: {exc}")

    if "/" in fps_raw:
        num, den = fps_raw.split("/", 1)
        fps = float(num) / float(den)
    else:
        fps = float(fps_raw)

    count = int(count_raw)
    if count <= 0 or fps <= 0:
        sys.exit(f"[ERROR] Invalid video metadata (frames={count}, fps={fps}): {path}")
    return count, fps


def parse_segments(
    total_frames: int,
    *,
    cuts: list[int] | None,
    segments: list[str] | None,
) -> list[tuple[int, int]]:
    if cuts is not None and segments is not None:
        sys.exit("[ERROR] Use only one of --cuts or --segments.")

    if cuts is not None:
        if not cuts:
            sys.exit("[ERROR] --cuts requires frame indices in start/end pairs.")
        if len(cuts) % 2 != 0:
            sys.exit(
                f"[ERROR] --cuts expects an even number of values (start/end pairs); "
                f"got {len(cuts)}."
            )
        out: list[tuple[int, int]] = []
        for i in range(0, len(cuts), 2):
            start, end = cuts[i], cuts[i + 1]
            if start < 0 or end > total_frames or start >= end:
                sys.exit(
                    f"[ERROR] Segment [{start}, {end}) invalid for video with "
                    f"{total_frames} frames."
                )
            out.append((start, end))
        return out

    if segments is not None:
        if not segments:
            sys.exit("[ERROR] --segments requires at least one START:END pair.")
        out: list[tuple[int, int]] = []
        for spec in segments:
            if ":" not in spec:
                sys.exit(f"[ERROR] Bad segment '{spec}'; expected START:END.")
            start_s, end_s = spec.split(":", 1)
            try:
                start, end = int(start_s), int(end_s)
            except ValueError:
                sys.exit(f"[ERROR] Bad segment '{spec}'; START and END must be integers.")
            if start < 0 or end > total_frames or start >= end:
                sys.exit(
                    f"[ERROR] Segment [{start}, {end}) invalid for video with "
                    f"{total_frames} frames."
                )
            out.append((start, end))
        return out

    sys.exit("[ERROR] Provide --cuts or --segments.")


def slice_with_ffmpeg(
    input_path: str,
    output_path: str,
    start: int,
    end: int,
    *,
    reencode: bool,
) -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("[ERROR] ffmpeg not found on PATH.")

    vf = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-vf",
        vf,
        "-an",
    ]
    if reencode:
        cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]
    else:
        cmd += ["-c:v", "copy"]
    cmd.append(output_path)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"[ERROR] ffmpeg failed for [{start}, {end}): {exc}")


def cmd_play(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.input):
        sys.exit(f"[ERROR] File not found: {args.input}")
    if shutil.which("ffplay") is None:
        sys.exit("[ERROR] ffplay not found on PATH.")

    total_frames, fps = probe_video(args.input)
    last = total_frames - 1
    # Use timestamp-based frame index so seek does not reset the counter.
    # %{n} restarts from 0 after ffplay seeks; floor(t*fps) stays absolute.
    fps_lit = re.sub(r"0+$", "", f"{fps:.6f}").rstrip(".")
    drawtext = (
        "drawtext="
        f"text='frame %{{eif\\:floor(t*{fps_lit})\\:d}} / {last}':"
        "x=20:y=30:fontsize=28:fontcolor=lime:borderw=2:bordercolor=black"
    )

    print(f"[INFO] {args.input}")
    print(f"[INFO] frames={total_frames}, fps={fps:.3f} (0-based index)")
    print("[INFO] ffplay: SPACE pause | S step | , / . frame seek | LEFT/RIGHT seek | Q quit")
    print("[INFO] Then slice with: python scripts/slice_video.py slice ... --cuts START END ...")

    subprocess.run(
        [
            "ffplay",
            "-hide_banner",
            "-loglevel",
            "error",
            "-window_title",
            "slice_video",
            "-vf",
            drawtext,
            args.input,
        ],
        check=False,
    )


def cmd_slice(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.input):
        sys.exit(f"[ERROR] File not found: {args.input}")

    total_frames, fps = probe_video(args.input)
    segs = parse_segments(total_frames, cuts=args.cuts, segments=args.segments)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    print(f"[INFO] {args.input}")
    print(f"[INFO] frames={total_frames}, fps={fps:.3f}")
    print(f"[INFO] {len(segs)} segment(s) → {out_dir}")

    for start, end in segs:
        out_path = os.path.join(out_dir, f"{stem}_{start:06d}_{end:06d}.mp4")
        print(f"  [{start:6d}, {end:6d})  ({end - start:5d} frames) → {out_path}")
        slice_with_ffmpeg(
            args.input,
            out_path,
            start,
            end,
            reencode=not args.stream_copy,
        )

    print("[DONE]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play or slice MP4 videos by frame.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_play = sub.add_parser("play", help="Play video with ffplay frame overlay.")
    p_play.add_argument("input", help="Input .mp4 file.")
    p_play.set_defaults(func=cmd_play)

    p_slice = sub.add_parser("slice", help="Cut video into frame-based segments.")
    p_slice.add_argument("input", help="Input .mp4 file.")
    cut = p_slice.add_mutually_exclusive_group(required=True)
    cut.add_argument(
        "--cuts",
        type=int,
        nargs="+",
        metavar="FRAME",
        help="Even count: start/end pairs, e.g. 120 300 450 600 → [120,300) and [450,600).",
    )
    cut.add_argument(
        "--segments",
        nargs="+",
        metavar="START:END",
        help="Explicit half-open segments, e.g. 0:120 120:300.",
    )
    p_slice.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: same as input).",
    )
    p_slice.add_argument(
        "--stream-copy",
        action="store_true",
        help="Use stream copy (fast, but cuts may be inaccurate). Default re-encodes.",
    )
    p_slice.set_defaults(func=cmd_slice)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
