#!/usr/bin/env python3
"""Filter reference motions from an eval_references.py report (no Isaac Sim required)."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter motions using eval_references report.json")
    parser.add_argument("--report", type=str, required=True, help="Path to report.json from eval_references.py")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to symlink passing motions into")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (useful when moving to another machine).",
    )
    parser.add_argument(
        "--include_names",
        type=str,
        default=None,
        help="Comma-separated motion_name substrings; if set, only these are considered.",
    )
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)

    include = None
    if args.include_names:
        include = {s.strip() for s in args.include_names.split(",") if s.strip()}

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    for entry in report.get("motions", []):
        if not entry.get("passed", False):
            continue
        name = entry.get("motion_name", "")
        if include is not None and not any(sub in name for sub in include):
            continue
        src = entry["motion_file"]
        if not os.path.isfile(src):
            print(f"[WARN] Missing source file, skip: {src}", file=sys.stderr)
            continue
        dst = os.path.join(args.output_dir, os.path.basename(src))
        if os.path.lexists(dst):
            os.remove(dst)
        if args.copy:
            import shutil

            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)
        count += 1

    manifest = os.path.join(args.output_dir, "passed_motions.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for fname in sorted(os.listdir(args.output_dir)):
            if fname.endswith(".npz"):
                f.write(os.path.join(args.output_dir, fname) + "\n")

    print(f"[INFO] Wrote {count} motion(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
