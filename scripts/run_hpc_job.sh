#!/bin/bash
# ============================================================================
# One-command HPC reproduction: prepare (on login node) + submit
#
# Requires CONTAINER_IMG to be set for your cluster.
#
# Usage:
#   export CONTAINER_IMG=/path/to/vllm_openai.sif
#   bash scripts/run_hpc_job.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${CONTAINER_IMG:-}" ]]; then
    echo "ERROR: CONTAINER_IMG is not set."
    echo "  export CONTAINER_IMG=/path/to/vllm_openai.sif"
    echo "  bash scripts/run_hpc_job.sh"
    exit 1
fi

echo "Step 1: Preparing FlashAttention clone (needs network)..."
bash scripts/prepare_flash_attention.sh

echo
echo "Step 2: Submitting Slurm job..."
sbatch scripts/submit_slurm.sh