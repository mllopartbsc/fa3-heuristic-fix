#!/bin/bash
# ============================================================================
# U-Curve Sweep Job — Kernel-level split sweep (1–64) + optional figure regeneration
#
# Runs the u_curve_sweep experiment with kernel-level timing and precomputed
# scheduler metadata (same as main_results). Optionally regenerates the paper
# figure if PREPRINT_DIR is set and points to pre_prints/fa3_patch.
#
# Prerequisites: Run prepare_flash_attention.sh once on login node.
#
# Usage:
#   1. Edit partition, account, and qos below for your cluster.
#   2. export CONTAINER_IMG=/path/to/your/container.sif
#   3. From repo root: cd /path/to/fa3-heuristic-fix && sbatch scripts/submit_u_curve_sweep.sh
#
# Optional figure regeneration (when paper layout exists):
#   export PREPRINT_DIR=/path/to/repo/pre_prints/fa3_patch
#
# Results:
#   - results/upstream_patch/u_curve_sweep.json
#   - (if PREPRINT_DIR set) pre_prints/fa3_patch/u_curve_kernel_hkv1.png, .pdf
# ============================================================================

#SBATCH --job-name=fa3-u-curve
#SBATCH --output=results/u_curve_sweep_%j.out
#SBATCH --error=results/u_curve_sweep_%j.err
#SBATCH --partition=acc
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
# ^ Edit partition, account, and qos for your cluster before submitting
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:30:00

set -euo pipefail

# Derive paths from script location (works when repo is at any path)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FA3_FIX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${FA3_FIX_DIR}/results"

CONTAINER_IMG="${CONTAINER_IMG:-}"
if [[ -z "$CONTAINER_IMG" ]] || [[ ! -f "$CONTAINER_IMG" ]]; then
    echo "ERROR: CONTAINER_IMG must point to an existing container image."
    echo "  export CONTAINER_IMG=/path/to/vllm_openai.sif"
    echo "  sbatch scripts/submit_u_curve_sweep.sh"
    exit 1
fi

FA3_CLONE="${FA3_FIX_DIR}/flash-attention"
if [[ ! -d "$FA3_CLONE/.git" ]]; then
    echo "ERROR: FlashAttention clone not found at $FA3_CLONE"
    echo "Run on LOGIN node first: bash scripts/prepare_flash_attention.sh"
    exit 1
fi

module load singularity/4.1.5 2>/dev/null || module load apptainer/1 2>/dev/null
CNT_BIN="$(command -v singularity || command -v apptainer)"
if [[ ! -x "$CNT_BIN" ]]; then
    echo "ERROR: singularity/apptainer not found. Try: module load singularity/4.1.5"
    exit 1
fi

mkdir -p "${RESULTS_DIR}/upstream_patch"
export SINGULARITY_TMPDIR="${FA3_FIX_DIR}/tmp_singularity"
mkdir -p "${SINGULARITY_TMPDIR}"

# Determine mount: standalone (fa3-heuristic-fix only) or full repo (sweep + figure)
if [[ -n "${PREPRINT_DIR:-}" ]] && [[ -f "${PREPRINT_DIR}/plot_u_curve_from_repro.py" ]]; then
    REPO_ROOT="$(cd "$(dirname "$PREPRINT_DIR")/../.." && pwd)"
    MOUNT_SRC="$REPO_ROOT"
    FA3_IN_REPO="/repo/reproduction_package/fa3-heuristic-fix"
    RUN_FIGURE=1
else
    MOUNT_SRC="$FA3_FIX_DIR"
    FA3_IN_REPO="/repo"
    RUN_FIGURE=0
fi

INNER="${RESULTS_DIR}/run_u_curve_${SLURM_JOB_ID}.sh"
cat > "${INNER}" << INNER_EOF
#!/bin/bash
set -euo pipefail

cd ${FA3_IN_REPO}
echo "=== Phase 1: Build baseline + patched FA3 ==="
bash scripts/setup_environment.sh

echo ""
echo "=== Phase 2: U-curve sweep (s=1..64, kernel-level + metadata) ==="
python3 run_experiments.py --skip-setup --track upstream_patch --experiment u_curve_sweep

if [[ "${RUN_FIGURE}" == "1" ]]; then
    echo ""
    echo "=== Phase 3: Regenerate figure ==="
    cd /repo/pre_prints/fa3_patch
    python3 plot_u_curve_from_repro.py
fi

echo ""
echo "Done. Results: ${FA3_IN_REPO}/results/upstream_patch/u_curve_sweep.json"
INNER_EOF

chmod +x "${INNER}"

echo "Launching container (mount $MOUNT_SRC -> /repo)..."
"${CNT_BIN}" exec --nv --cleanenv \
    -B "${MOUNT_SRC}:/repo" \
    "${CONTAINER_IMG}" \
    bash "${INNER}"

rm -f "${INNER}"
echo "Results: ${RESULTS_DIR}/upstream_patch/u_curve_sweep.json"
