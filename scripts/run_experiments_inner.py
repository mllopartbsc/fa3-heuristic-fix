#!/usr/bin/env python3
"""
Inner runner for kernel-level experiments.

Called by run_experiments.py with --track. Sets up PYTHONPATH and runs
each experiment module in sequence. Excludes e2e_decode_simulation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# All kernel-level experiments (excluding e2e_decode_simulation)
EXPERIMENTS = [
    ("exp1_correctness", "exp1_correctness.py"),
    ("exp2_profiling", "exp2_mechanism_profiling.py"),
    ("main_results", "main_results.py"),
    ("guard_ablation", "guard_ablation.py"),
    ("boundary_sweep", "boundary_sweep.py"),
    ("u_curve_sweep", "u_curve_sweep.py"),
    ("exp3_safety", "exp3_safety_verification.py"),
]

# Only for latest_stack_tuned
EXPERIMENTS_LATEST_STACK = [
    ("threshold_sensitivity", "threshold_sensitivity.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["upstream_patch", "latest_stack_tuned"], required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--experiment", type=str, default="")
    args = parser.parse_args()

    track = args.track
    profile_root = REPO_ROOT / ".pydeps" / ("upstream_patch" if track == "upstream_patch" else "baseline")
    results_dir = REPO_ROOT / "results" / track
    artifacts_dir = REPO_ROOT / "artifacts" / track
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{profile_root}:{REPO_ROOT}" + (f":{existing}" if existing else "")

    all_experiments = list(EXPERIMENTS)
    if track == "latest_stack_tuned":
        all_experiments.extend(EXPERIMENTS_LATEST_STACK)

    ran_any = False
    for name, script in all_experiments:
        if args.experiment and args.experiment != name:
            continue

        script_path = EXPERIMENTS_DIR / script
        if not script_path.exists():
            print(f"[WARN] Script not found: {script_path}")
            continue

        print("\n" + "─" * 70)
        print(f"  Running: {name}")
        print("─" * 70)

        cmd = [sys.executable, str(script_path), "--track", track, "--results-dir", str(results_dir)]
        if args.quick:
            cmd.append("--quick")

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"\n[ERROR] {name} failed with exit code {result.returncode}")
            sys.exit(result.returncode)
        ran_any = True

    if args.experiment and not ran_any:
        valid = ", ".join(n for n, _ in all_experiments)
        print(f"\n[ERROR] No experiment matched '--experiment {args.experiment}'")
        print(f"Valid names: {valid}")
        sys.exit(1)

    if not args.experiment:
        print("\n" + "─" * 70)
        print("  Generating tables")
        print("─" * 70)
        gen_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_tables.py"),
            "--track", track,
            "--results-dir", str(results_dir),
            "--output-tex", str(artifacts_dir / "tables.tex"),
        ]
        subprocess.run(gen_cmd, env=env)


if __name__ == "__main__":
    main()
