#!/bin/bash
# ============================================================================
# Run End-to-End Decode Benchmark (via parent reproduction package)
#
# This script invokes the reproduction package's vLLM-based E2E benchmark.
# Run from fa3-heuristic-fix repo root.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPRO_PACKAGE="$(cd "$REPO_ROOT/.." && pwd)"

if [[ ! -f "$REPRO_PACKAGE/submit_vllm_e2e_latest_patch.sh" ]]; then
    echo "ERROR: submit_vllm_e2e_latest_patch.sh not found in reproduction package."
    echo "  Expected: $REPRO_PACKAGE/submit_vllm_e2e_latest_patch.sh"
    echo "  Run this script from within the fa3-heuristic-fix repo (inside reproduction_package)."
    exit 1
fi

echo "Submitting E2E decode job from reproduction package..."
cd "$REPRO_PACKAGE"
sbatch submit_vllm_e2e_latest_patch.sh
