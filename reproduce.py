#!/usr/bin/env python3
"""
Reproduce FA3 Heuristic Fix Results
═══════════════════════════════════
Main entry point for the reproduction package.
Runs setup, experiments, and validation in sequence.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_cmd(cmd, check=True):
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="FA3 Heuristic Fix — Full Reproduction")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip cloning and building FlashAttention-3 (assumes already built)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer iterations for a fast sanity check (~5 mins)")
    parser.add_argument("--experiment", type=str, default="",
                        help="Run only a specific experiment (e.g., 'main_results')")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.absolute()
    os.chdir(repo_root)

    print("══════════════════════════════════════════════════════════════════")
    print("  FlashAttention-3 Heuristic Fix — Full Reproduction             ")
    print("══════════════════════════════════════════════════════════════════")

    # 1. Setup Environment
    if not args.skip_setup:
        print("\n[Phase 1/4] Environment Setup (Cloning & Building FA3)")
        run_cmd(["bash", "scripts/setup_environment.sh"])
    else:
        print("\n[Phase 1/4] Environment Setup: SKIPPED (--skip-setup)")

    # 2. Run Experiments
    print("\n[Phase 2/4] Running Experiments")
    cmd = ["bash", "scripts/run_all.sh"]
    if args.quick:
        cmd.append("--quick")
    if args.experiment:
        cmd.extend(["--experiment", args.experiment])
    run_cmd(cmd)

    # 3. Validate Claims
    if not args.experiment:
        print("\n[Phase 3/4] Validating Results against Expected Claims")
        run_cmd(["python3", "src/validate_claims.py"])
    else:
        print("\n[Phase 3/4] Validating Results: SKIPPED (single experiment mode)")

    # 4. Generate Tables (if full run)
    if not args.experiment:
        print("\n[Phase 4/4] Generating Summary Tables")
        run_cmd(["python3", "scripts/generate_tables.py"], check=False)
    else:
        print("\n[Phase 4/4] Summary Tables: SKIPPED (single experiment mode)")

    print("\n══════════════════════════════════════════════════════════════════")
    print("  Reproduction Workflow Complete!                                 ")
    print("══════════════════════════════════════════════════════════════════")
    print("\nResults are available in the 'results/' directory.")


if __name__ == "__main__":
    main()
