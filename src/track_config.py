"""
Shared track, path, and metadata helpers for the reproduction package.

The repo supports two explicit evidence tracks:
  - upstream_patch: the real two-guard C++ change proposed for upstream FA3
  - latest_stack_tuned: the same safety guards plus a tuned low-tile s=3 choice
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRACK_UPSTREAM_PATCH = "upstream_patch"
TRACK_LATEST_STACK_TUNED = "latest_stack_tuned"
TRACK_ALL = "all"
DEFAULT_TRACK = TRACK_UPSTREAM_PATCH

TRACK_CHOICES = (TRACK_UPSTREAM_PATCH, TRACK_LATEST_STACK_TUNED)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
EXPECTED_RESULTS_ROOT = REPO_ROOT / "expected_results"
PYDEPS_ROOT = REPO_ROOT / ".pydeps"


# Route 1: policy_injected — Python injects num_splits + precomputed metadata (same binary)
# Route 2: patched_binary_with_metadata — baseline vs patched binaries, each with precomputed metadata
TRACK_SPECS: dict[str, dict[str, Any]] = {
    TRACK_UPSTREAM_PATCH: {
        "display_name": "Upstream Two-Guard Patch",
        "candidate_policy": "upstream_two_guard",
        "baseline_policy": "baseline_upstream",
        "benchmark_mode": "patched_binary_with_metadata",  # Route 2
        "runtime_profile": "upstream_patch",
        "baseline_runtime_profile": "baseline",
    },
    TRACK_LATEST_STACK_TUNED: {
        "display_name": "Latest-Stack Tuned s=3 Policy",
        "candidate_policy": "latest_stack_tuned_s3",
        "baseline_policy": "baseline_upstream",
        "benchmark_mode": "policy_injected",  # Route 1
        "runtime_profile": "baseline",
        "baseline_runtime_profile": "baseline",
    },
}


def validate_track(track: str) -> str:
    if track not in TRACK_CHOICES:
        raise ValueError(f"Unsupported track '{track}'. Expected one of {TRACK_CHOICES}.")
    return track


def add_track_argument(
    parser: argparse.ArgumentParser,
    *,
    allow_all: bool = False,
    default: str = DEFAULT_TRACK,
) -> None:
    choices = list(TRACK_CHOICES)
    if allow_all:
        choices.append(TRACK_ALL)
    parser.add_argument(
        "--track",
        choices=choices,
        default=default,
        help="Evidence track to run or inspect.",
    )


def add_results_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override the default track-scoped results directory.",
    )


def add_artifacts_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Override the default track-scoped artifacts directory.",
    )


def track_spec(track: str) -> dict[str, Any]:
    return TRACK_SPECS[validate_track(track)]


def results_dir_for_track(track: str, override: Path | None = None) -> Path:
    return override if override is not None else RESULTS_ROOT / validate_track(track)


def artifacts_dir_for_track(track: str, override: Path | None = None) -> Path:
    return override if override is not None else ARTIFACTS_ROOT / validate_track(track)


def claims_file_for_track(track: str) -> Path:
    return EXPECTED_RESULTS_ROOT / validate_track(track) / "claims.json"


def install_root_for_profile(profile: str) -> Path:
    return PYDEPS_ROOT / profile


def runtime_profile_for_track(track: str) -> str:
    return track_spec(track)["runtime_profile"]


def baseline_runtime_profile_for_track(track: str) -> str:
    return track_spec(track)["baseline_runtime_profile"]


def install_root_for_track(track: str) -> Path:
    return install_root_for_profile(runtime_profile_for_track(track))


def baseline_install_root_for_track(track: str) -> Path:
    return install_root_for_profile(baseline_runtime_profile_for_track(track))


def result_path_for_experiment(
    experiment: str,
    *,
    track: str,
    results_dir: Path | None = None,
) -> Path:
    out_dir = results_dir_for_track(track, results_dir)
    return out_dir / f"{experiment}.json"


def ensure_track_dirs(
    *,
    track: str,
    results_dir: Path | None = None,
    artifacts_dir: Path | None = None,
) -> tuple[Path, Path]:
    out_results = results_dir_for_track(track, results_dir)
    out_artifacts = artifacts_dir_for_track(track, artifacts_dir)
    out_results.mkdir(parents=True, exist_ok=True)
    out_artifacts.mkdir(parents=True, exist_ok=True)
    return out_results, out_artifacts


def enrich_result_payload(
    payload: dict[str, Any],
    *,
    experiment: str,
    track: str,
    benchmark_mode: str | None = None,
    candidate_policy: str | None = None,
    baseline_policy: str | None = None,
) -> dict[str, Any]:
    spec = track_spec(track)
    enriched = dict(payload)
    enriched.setdefault("experiment", experiment)
    enriched.setdefault("track", track)
    enriched.setdefault("track_display_name", spec["display_name"])
    enriched.setdefault("benchmark_mode", benchmark_mode or spec["benchmark_mode"])
    enriched.setdefault("baseline_policy", baseline_policy or spec["baseline_policy"])
    enriched.setdefault("candidate_policy", candidate_policy or spec["candidate_policy"])
    enriched.setdefault("runtime_profile", spec["runtime_profile"])
    return enriched


def write_track_json(
    payload: dict[str, Any],
    *,
    experiment: str,
    track: str,
    results_dir: Path | None = None,
    benchmark_mode: str | None = None,
    candidate_policy: str | None = None,
    baseline_policy: str | None = None,
) -> Path:
    out_path = result_path_for_experiment(experiment, track=track, results_dir=results_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = enrich_result_payload(
        payload,
        experiment=experiment,
        track=track,
        benchmark_mode=benchmark_mode,
        candidate_policy=candidate_policy,
        baseline_policy=baseline_policy,
    )
    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2)
    return out_path
