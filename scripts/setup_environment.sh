#!/bin/bash
# ============================================================================
# Environment Setup: FlashAttention-3 Heuristic Fix Reproduction
#
# This script clones FlashAttention-3, applies the two-line patch, and
# builds both baseline and patched versions for benchmarking.
#
# Prerequisites:
#   - NVIDIA H100 GPU (or Hopper-class)
#   - CUDA >= 12.3
#   - PyTorch >= 2.4.0
#   - Python >= 3.9
#
# Usage:
#   bash scripts/setup_environment.sh [--baseline-only] [--patched-only]
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
PYTHON_TARGET="$REPO_ROOT/.pydeps"

echo "============================================"
echo "  FlashAttention-3 Heuristic Fix — Setup"
echo "============================================"
echo

mkdir -p "$RESULTS_DIR"
mkdir -p "$PYTHON_TARGET"

# Install into a repo-local site-packages directory so containerized runs do
# not depend on write access to system Python paths.
export PYTHONPATH="$PYTHON_TARGET${PYTHONPATH:+:$PYTHONPATH}"

# Cap Torch/Ninja parallelism to avoid overcommitting memory during FA3 builds.
export MAX_JOBS="${MAX_JOBS:-20}"
# Only build SM90 (H100/Hopper) kernels — SM80 is not needed for this reproduction.
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_FORCE_BUILD=TRUE
echo "  MAX_JOBS: $MAX_JOBS"
echo "  DISABLE_SM80: $FLASH_ATTENTION_DISABLE_SM80"
echo "  PYTHON_TARGET: $PYTHON_TARGET"
echo

# ── Check prerequisites ──────────────────────────────────────────────────
echo "[1/5] Checking prerequisites..."

if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Please ensure CUDA >= 12.3 is installed."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "  CUDA version: $CUDA_VERSION"

python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}')" || {
    echo "ERROR: PyTorch not found. Please install PyTorch >= 2.4.0."
    exit 1
}

python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
props = torch.cuda.get_device_properties(0)
print(f'  GPU: {props.name} ({props.multi_processor_count} SMs)')
if props.multi_processor_count < 100:
    print('  WARNING: This GPU has fewer SMs than H100 (132). Results will differ.')
"

echo

# ── Clone FlashAttention-3 ───────────────────────────────────────────────
FA3_DIR="$REPO_ROOT/flash-attention"
FA3_COMMIT="fbf24f67"  # Baseline version

echo "[2/5] Setting up FlashAttention-3..."

if [ -d "$FA3_DIR" ]; then
    echo "  flash-attention/ already exists. Skipping clone."
else
    echo "  Cloning FlashAttention-3..."
    git clone https://github.com/Dao-AILab/flash-attention.git "$FA3_DIR"
fi

cd "$FA3_DIR"

# Checkout the exact commit
# echo "  Checking out baseline commit $FA3_COMMIT..."
# git fetch origin
# git checkout "$FA3_COMMIT" 2>/dev/null || {
#     echo "  WARNING: Could not checkout exact commit $FA3_COMMIT."
#     echo "  Using current HEAD instead. Results may differ slightly."
# }

echo

# ── Build baseline ───────────────────────────────────────────────────────
if [[ "${1:-}" != "--patched-only" ]]; then
    echo "[3/5] Building BASELINE FlashAttention-3..."
    echo "  This may take 10-30 minutes depending on your system."

    # Reset any patches
    git checkout -- hopper/heuristics.h 2>/dev/null || true

    # Build
    cd "$FA3_DIR/hopper"
    rm -rf "$PYTHON_TARGET"/flash_attn*
    rm -rf "$PYTHON_TARGET"/flash_attn_3*
    python3 -m pip install \
        --no-build-isolation \
        --no-deps \
        --upgrade \
        --target "$PYTHON_TARGET" \
        . 2>&1 | tee "$RESULTS_DIR/setup_baseline_build.log"

    echo "  Baseline build complete."
    echo
fi

# ── Apply patch and build patched version ────────────────────────────────
if [[ "${1:-}" != "--baseline-only" ]]; then
    echo "[4/5] Applying tile-aware fix and building PATCHED version..."

    cd "$FA3_DIR"

    # Apply the patch
    # The patch modifies a single guard in hopper/heuristics.h
    HEURISTICS_FILE="hopper/heuristics.h"

    # Check if the baseline guard is still present (not yet patched)
    if grep -q 'if (num_n_blocks <= 4) { return 1; }' "$HEURISTICS_FILE"; then
        echo "  Found premature guard. Applying two-line fix..."

        # The actual patch: replace the single guard with two tile-aware guards.
        # Use the same regex as the Dockerfile for consistency.
        python3 -c "
import re, sys

path = '$HEURISTICS_FILE'
with open(path, 'r') as f:
    src = f.read()

if 'num_n_blocks <= 4' not in src:
    print('ERROR: Could not find baseline guard in heuristics.h', file=sys.stderr)
    sys.exit(1)

old = r'if \(num_n_blocks <= 4\) \{ return 1; \}'
new = (
    '// Guard 1: L_K <= 384 (nblk <= 3) -- combine cost always too high\\n'
    '    if (num_n_blocks <= 3) { return 1; }\\n'
    '\\n'
    '    // Guard 2: L_K = 448-512 (nblk = 4) with enough tiles\\n'
    '    // total_mblocks == batch_size * num_heads (function parameter)\\n'
    '    if (num_n_blocks <= 4 && total_mblocks >= 4) { return 1; }'
)

patched = re.sub(old, new, src, count=1)
if patched == src:
    print('ERROR: Regex replacement failed — check heuristics.h format', file=sys.stderr)
    sys.exit(1)

with open(path, 'w') as f:
    f.write(patched)

print('Patch applied: heuristics.h')
print('  - Guard 1: if (num_n_blocks <= 3) { return 1; }')
print('  - Guard 2: if (num_n_blocks <= 4 && total_mblocks >= 4) { return 1; }')
"
    elif grep -q 'num_n_blocks <= 3' "$HEURISTICS_FILE"; then
        echo "  Patch already applied (found Guard 1 signature). Skipping."
    else
        echo "  WARNING: Neither the baseline guard nor the patch was found."
        echo "  Check $HEURISTICS_FILE manually."
    fi

    # Build patched version
    cd "$FA3_DIR/hopper"
    rm -rf "$PYTHON_TARGET"/flash_attn*
    rm -rf "$PYTHON_TARGET"/flash_attn_3*
    python3 -m pip install \
        --no-build-isolation \
        --no-deps \
        --upgrade \
        --target "$PYTHON_TARGET" \
        . 2>&1 | tee "$RESULTS_DIR/setup_patched_build.log"

    echo "  Patched build complete."
    echo
fi

# ── Verify installation ──────────────────────────────────────────────────
echo "[5/5] Verifying installation..."

python3 -c "
import flash_attn_interface
print('  flash_attn_interface imported successfully.')
print(f'  Module location: {flash_attn_interface.__file__}')
" || {
    echo "  ERROR: Could not import flash_attn_interface."
    echo "  Please check the build output above for errors."
    exit 1
}

echo
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo
echo "Repo-local Python packages:"
echo "  $PYTHON_TARGET"
echo
echo "To run all experiments:"
echo "  cd $REPO_ROOT"
echo "  bash scripts/run_all.sh"
echo
echo "To run a single experiment:"
echo "  cd $REPO_ROOT"
echo "  python3 experiments/main_results.py"
