#!/bin/bash
# ============================================================================
# Prepare FlashAttention clone (run on LOGIN NODE - has network access)
#
# Compute nodes on many HPC systems cannot reach GitHub. Run this once before
# submitting the Slurm job to clone flash-attention into the repo.
#
# Usage: bash scripts/prepare_flash_attention.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FA3_DIR="$REPO_ROOT/flash-attention"
FA3_COMMIT="fbf24f67"

echo "Preparing FlashAttention clone (requires network)..."
echo "  Target: $FA3_DIR"
echo "  Commit: $FA3_COMMIT"
echo

if [[ -d "$FA3_DIR/.git" ]]; then
    echo "Clone exists. Ensuring correct commit and submodules..."
    cd "$FA3_DIR"
    git fetch --all --tags 2>/dev/null || true
    git checkout "$FA3_COMMIT" 2>/dev/null || {
        echo "ERROR: Could not checkout $FA3_COMMIT. Try: git fetch origin && git checkout $FA3_COMMIT"
        exit 1
    }
    echo "  Initializing submodules (CUTLASS, etc.)..."
    git submodule update --init --recursive
    echo "  Ready."
else
    echo "Cloning FlashAttention-3..."
    git clone --recurse-submodules https://github.com/Dao-AILab/flash-attention.git "$FA3_DIR"
    cd "$FA3_DIR"
    git checkout "$FA3_COMMIT"
    echo "  Initializing submodules (in case --recurse missed any)..."
    git submodule update --init --recursive
    echo "  Done."
fi

echo
echo "Next: Edit scripts/submit_slurm.sh (YOUR_ACCOUNT, YOUR_QOS), then:"
echo "  export CONTAINER_IMG=/path/to/container.sif && sbatch scripts/submit_slurm.sh"
