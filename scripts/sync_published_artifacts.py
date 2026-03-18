#!/usr/bin/env python3
"""Sync committed reviewer artifacts from canonical track outputs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_ROOT = REPO_ROOT / "results" / "published" / "reviewer_artifacts"
TRACKS = ("upstream_patch", "latest_stack_tuned")
ARTIFACT_FILES = (
    "claim_validation.json",
    "ci_benchmark_report.json",
    "provenance.json",
    "tables.tex",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync results/<track> and artifacts/<track> into results/published/reviewer_artifacts/<track>."
    )
    parser.add_argument(
        "--track",
        choices=(*TRACKS, "all"),
        default="upstream_patch",
        help="Track to sync (default: upstream_patch).",
    )
    return parser.parse_args()


def sync_track(track: str) -> None:
    src_results = REPO_ROOT / "results" / track
    src_artifacts = REPO_ROOT / "artifacts" / track
    dst_dir = PUBLISHED_ROOT / track

    if not src_results.is_dir():
        raise FileNotFoundError(f"Results directory not found: {src_results}")
    if not src_artifacts.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found: {src_artifacts}")

    result_files = sorted(src_results.glob("*.json"))
    if not result_files:
        raise FileNotFoundError(f"No JSON result files found in {src_results}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src_file in result_files:
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied.append(dst_file.relative_to(REPO_ROOT))

    missing_artifacts = []
    for name in ARTIFACT_FILES:
        src_file = src_artifacts / name
        if not src_file.exists():
            missing_artifacts.append(src_file)
            continue
        dst_file = dst_dir / name
        shutil.copy2(src_file, dst_file)
        copied.append(dst_file.relative_to(REPO_ROOT))

    print(f"[{track}] synced {len(copied)} files")
    for path in copied:
        print(f"  - {path}")

    if missing_artifacts:
        print(f"[{track}] warning: skipped missing artifacts", file=sys.stderr)
        for path in missing_artifacts:
            print(f"  - {path}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    tracks = TRACKS if args.track == "all" else (args.track,)

    for track in tracks:
        sync_track(track)


if __name__ == "__main__":
    main()
