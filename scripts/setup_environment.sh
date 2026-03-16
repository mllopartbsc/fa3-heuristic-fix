#!/bin/bash
# ============================================================================
# Environment Setup: FlashAttention-3 Heuristic Fix Reproduction
#
# Builds two explicit binary profiles (Hopper-only, SM90):
#   - baseline:       upstream FA3 at the pinned commit
#   - upstream_patch: same commit + heuristics.h patch only (what FA3 merges)
#
# All builds are Hopper-only. Route 2 uses baseline vs patched with precomputed
# metadata. Route 1 (latest_stack_tuned) reuses baseline and injects from Python.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT
RESULTS_DIR="$REPO_ROOT/results/setup"
PYDEPS_ROOT="$REPO_ROOT/.pydeps"
BASELINE_ROOT="$PYDEPS_ROOT/baseline"
UPSTREAM_PATCH_ROOT="$PYDEPS_ROOT/upstream_patch"
FA3_DIR="$REPO_ROOT/flash-attention"
PATCH_FILE="$REPO_ROOT/patch/heuristics.patch"
FA3_COMMIT="fbf24f67"

BUILD_BASELINE=true
BUILD_PATCHED=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline-only) BUILD_PATCHED=false; shift ;;
        --patched-only) BUILD_BASELINE=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"
mkdir -p "$PYDEPS_ROOT"

export MAX_JOBS="${MAX_JOBS:-20}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
# Hopper-only minimal build (faster): SM90, hdim128, no backward/FP8/other head dims
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE

echo "============================================"
echo "  FlashAttention-3 Heuristic Fix — Setup"
echo "============================================"
echo
echo "  Pinned commit:     $FA3_COMMIT"
echo "  Build:             Hopper-only, hdim128, no backward/FP8 (faster)"
echo "  Baseline root:     $BASELINE_ROOT"
echo "  Upstream root:     $UPSTREAM_PATCH_ROOT"
echo "  Patch artifact:    $PATCH_FILE"
echo

echo "[1/6] Checking prerequisites..."
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Please ensure CUDA >= 12.3 is installed."
    exit 1
fi
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

echo "[2/6] Preparing FlashAttention clone..."
if [[ -d "$FA3_DIR/.git" ]]; then
    echo "  Reusing existing flash-attention clone."
else
    echo "  Cloning FlashAttention-3..."
    git clone https://github.com/Dao-AILab/flash-attention.git "$FA3_DIR"
fi
echo

ensure_clean_checkout() {
    cd "$FA3_DIR"
    git fetch --all --tags >/dev/null 2>&1 || true
    # Force checkout to discard any leftover patch from previous runs
    git checkout -f "$FA3_COMMIT" >/dev/null 2>&1 || {
        echo "ERROR: Could not checkout pinned commit $FA3_COMMIT."
        exit 1
    }
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "ERROR: $FA3_DIR has local modifications after checkout."
        git status --short
        exit 1
    fi
}

install_profile() {
    local target_root="$1"
    local build_label="$2"
    rm -rf "$target_root"
    mkdir -p "$target_root"
    python3 "$REPO_ROOT/scripts/apply_hopper_only_setup.py" "$FA3_DIR/hopper/setup.py" || true
    cd "$FA3_DIR/hopper"
    # Force full recompile: remove stale build artifacts (pip/ninja cache can
    # return in ~30s otherwise, producing wrong or inconsistent binaries)
    rm -rf build/ dist/ *.egg-info
    # Progress: run pip and a background timer that prints elapsed time every 30s
    (
        BUILD_START=$(date +%s)
        while true; do
            sleep 30
            ELAPSED=$(($(date +%s) - BUILD_START))
            echo "  [Build progress] $((ELAPSED / 60))m $((ELAPSED % 60))s elapsed..."
        done
    ) &
    PROGRESS_PID=$!
    python3 -m pip install \
        --no-build-isolation \
        --no-deps \
        --upgrade \
        --target "$target_root" \
        . 2>&1 | tee "$RESULTS_DIR/${build_label}_build.log"
    PIP_EXIT=$?
    kill $PROGRESS_PID 2>/dev/null || true
    wait $PROGRESS_PID 2>/dev/null || true
    [ $PIP_EXIT -eq 0 ] || exit $PIP_EXIT
}

verify_profile() {
    local target_root="$1"
    local label="$2"
    local pycode="import flash_attn_interface; print('  [${label}] flash_attn_interface:', flash_attn_interface.__file__)"
    PYTHONPATH="$target_root${PYTHONPATH:+:$PYTHONPATH}" python3 -c "$pycode" || {
        echo "ERROR: Could not import flash_attn_interface from $target_root"
        exit 1
    }
}

if [[ "$BUILD_BASELINE" == "true" ]]; then
    echo "[3/6] Building baseline profile..."
    ensure_clean_checkout
    install_profile "$BASELINE_ROOT" "baseline"
    verify_profile "$BASELINE_ROOT" "baseline"
    echo
fi

if [[ "$BUILD_PATCHED" == "true" ]]; then
    echo "[4/6] Applying upstream patch and building patched profile..."
    ensure_clean_checkout
    if git apply --check "$PATCH_FILE" 2>/dev/null; then
        git apply "$PATCH_FILE"
    else
        echo "  (git apply failed, using direct patch script)"
        python3 "$REPO_ROOT/scripts/apply_heuristics_patch.py" || {
            echo "ERROR: Failed to apply heuristics patch."
            exit 1
        }
    fi
    install_profile "$UPSTREAM_PATCH_ROOT" "upstream_patch"
    verify_profile "$UPSTREAM_PATCH_ROOT" "upstream_patch"
    echo
fi

echo "[5/6] Writing build manifest..."
python3 - <<'PY'
import json
import os
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
manifest = {
    "flash_attention_commit": "fbf24f67",
    "profiles": {
        "baseline": str(repo_root / ".pydeps" / "baseline"),
        "upstream_patch": str(repo_root / ".pydeps" / "upstream_patch"),
    },
    "latest_stack_tuned_runtime_profile": "baseline",
    "patch_artifact": str(repo_root / "patch" / "heuristics.patch"),
}
out = repo_root / "artifacts" / "build_manifest.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
print(f"  Build manifest: {out}")
PY
echo

echo "[6/6] Setup complete."
echo
echo "To run experiments:"
echo "  python3 run_experiments.py --track upstream_patch --quick"
echo "  python3 run_experiments.py --track latest_stack_tuned --quick"
