#!/usr/bin/env python3
"""
Unified Reproduction Script — FlashAttention-3 Heuristic Fix

Runs all kernel-level experiments from the evaluation suite in a single entry point.
Excludes end-to-end (vLLM) experiments, which require separate infrastructure.

Experiments (in order):
  1. exp1_correctness       — Numerical correctness and determinism
  2. exp2_profiling         — Mechanism confirmation via profiling
  3. main_results           — Kernel-level latency (main results table)
  4. guard_ablation         — Guard ablation study
  5. boundary_sweep         — MQA crossover sweep around L_K boundary
  6. u_curve_sweep          — Extended split sweep (U-curve)
  7. exp3_safety            — Safety and regression profiling (160 configs)
  8. threshold_sensitivity  — (latest_stack_tuned only)

Usage:
  python3 run_experiments.py [--track TRACK] [--skip-setup] [--quick]
  python3 run_experiments.py --track upstream_patch --quick   # Fast sanity check
  python3 run_experiments.py --track all                      # Both tracks

Prerequisites: Linux, Hopper GPU (H100), CUDA ≥ 12.3, PyTorch ≥ 2.4.0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd: list, check: bool = True) -> int:
    """Run a command and return exit code."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all kernel-level experiments for FA3 heuristic fix reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--track",
        choices=["upstream_patch", "latest_stack_tuned", "all"],
        default="upstream_patch",
        help="Evidence track to run (default: upstream_patch)",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip environment setup (assumes FA3 already built)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer iterations (~5–15 min per track)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
        help="Run only a specific experiment (e.g., main_results)",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    print("=" * 70)
    print("  FlashAttention-3 Heuristic Fix — Unified Reproduction")
    print("=" * 70)

    if not args.skip_setup:
        print("\n[Phase 1/3] Environment Setup (cloning & building FA3)")
        run_cmd(["bash", str(REPO_ROOT / "scripts" / "setup_environment.sh")])
    else:
        print("\n[Phase 1/3] Environment Setup: SKIPPED (--skip-setup)")

    tracks = ["upstream_patch", "latest_stack_tuned"] if args.track == "all" else [args.track]

    for track in tracks:
        print(f"\n[Phase 2/3] Running experiments for track: {track}")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_experiments_inner.py"),
            "--track", track,
        ]
        if args.quick:
            cmd.append("--quick")
        if args.experiment:
            cmd.extend(["--experiment", args.experiment])
        run_cmd(cmd)

        if not args.experiment:
            print(f"\n[Phase 3/3] Validation for track: {track}")
            sys.path.insert(0, str(REPO_ROOT))
            from src.track_config import (
                artifacts_dir_for_track,
                claims_file_for_track,
                results_dir_for_track,
            )
            results_dir = results_dir_for_track(track)
            claims_file = claims_file_for_track(track)
            artifacts_dir = artifacts_dir_for_track(track)
            if claims_file.exists():
                run_cmd([
                    sys.executable, str(REPO_ROOT / "src" / "validate_claims.py"),
                    "--track", track,
                    "--results-dir", str(results_dir),
                    "--claims-file", str(claims_file),
                    "--json-out", str(artifacts_dir / "claim_validation.json"),
                ])
            else:
                print(f"  (No claims file at {claims_file}, skipping validation)")

    print("\n" + "=" * 70)
    print("  Reproduction complete!")
    print("=" * 70)
    for track in tracks:
        results_dir = REPO_ROOT / "results" / track
        artifacts_dir = REPO_ROOT / "artifacts" / track
        print(f"  Results ({track}):   {results_dir}")
        print(f"  Artifacts ({track}):  {artifacts_dir}")


if __name__ == "__main__":
    main()
