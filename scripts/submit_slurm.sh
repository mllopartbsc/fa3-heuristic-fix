#!/bin/bash
# ============================================================================
# Slurm Job — Full Reproduction (HPC with Hopper GPUs)
#
# Runs the complete fa3-heuristic-fix reproduction suite inside a container
# (PyTorch, CUDA, nvcc). Builds are Hopper-only (SM90). Requires H100 or compatible.
#
# IMPORTANT: Compute nodes often cannot reach GitHub. Run this ONCE on the login node first:
#   bash scripts/prepare_flash_attention.sh
#
# Usage:
#   1. Run prepare_flash_attention.sh once on the login node (has network).
#   2. Edit YOUR_ACCOUNT, YOUR_QOS, and partition below for your cluster.
#   3. From repo root: export CONTAINER_IMG=/path/to/vllm_openai.sif && sbatch scripts/submit_slurm.sh
#
# After the job completes:
#   Results: results/upstream_patch/ and results/latest_stack_tuned/
# ============================================================================

#SBATCH --job-name=fa3-heuristic-fix
#SBATCH --output=results/slurm_full_run_%j.out
#SBATCH --error=results/slurm_full_run_%j.err
#SBATCH --partition=acc
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
# ^ Edit partition, account, and qos for your cluster before submitting
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00

set -euo pipefail

# Derive repo root from script location (works when repo is at any path)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"

# Container image (required; set before sbatch for your cluster)
CONTAINER_IMG="${CONTAINER_IMG:-}"
if [[ -z "$CONTAINER_IMG" ]] || [[ ! -f "$CONTAINER_IMG" ]]; then
    echo "ERROR: CONTAINER_IMG must point to an existing container image."
    echo "  export CONTAINER_IMG=/path/to/vllm_openai.sif"
    echo "  sbatch scripts/submit_slurm.sh"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/upstream_patch"
mkdir -p "${RESULTS_DIR}/latest_stack_tuned"
mkdir -p "${RESULTS_DIR}/published"

module load singularity/4.1.5 2>/dev/null || module load apptainer/1 2>/dev/null
CNT_BIN="$(command -v singularity || command -v apptainer)"
if [[ ! -x "$CNT_BIN" ]]; then
    echo "ERROR: singularity/apptainer not found. Try: module load singularity/4.1.5"
    exit 1
fi

if [[ ! -f "$CONTAINER_IMG" ]]; then
    echo "ERROR: Container not found: $CONTAINER_IMG"
    exit 1
fi

FA3_CLONE="${REPO_ROOT}/flash-attention"
if [[ ! -d "$FA3_CLONE/.git" ]]; then
    echo "ERROR: FlashAttention clone not found at $FA3_CLONE"
    echo "Compute nodes cannot reach GitHub. Run this on the LOGIN node first:"
    echo "  bash scripts/prepare_flash_attention.sh"
    exit 1
fi

export SINGULARITY_TMPDIR="${REPO_ROOT}/tmp_singularity"
export SINGULARITY_CACHEDIR="${SINGULARITY_TMPDIR}"
export APPTAINER_TMPDIR="${SINGULARITY_TMPDIR}"
export APPTAINER_CACHEDIR="${SINGULARITY_TMPDIR}"
mkdir -p "${SINGULARITY_TMPDIR}"

INNER_SCRIPT="${RESULTS_DIR}/run_inner_${SLURM_JOB_ID}.sh"

cat > "${INNER_SCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail

cd /repo

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  FlashAttention-3 Heuristic Fix — Full Reproduction             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_JOB_NODELIST}"
echo "Repo:         /repo"
echo "[Container] Python:  \$(python3 --version)"
echo "[Container] PyTorch: \$(python3 -c 'import torch; print(torch.__version__)')"
echo "[Container] CUDA:    \$(python3 -c 'import torch; print(torch.version.cuda)')"
echo "[Container] nvcc:    \$(command -v nvcc || echo 'not found')"
date
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 1: Building baseline and patched FA3 profiles"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash scripts/setup_environment.sh
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 2: Running all kernel experiments (both tracks)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 run_experiments.py --skip-setup --track all --quick
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 3: CI reports + regression tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for TRACK in upstream_patch latest_stack_tuned; do
    if [[ -f "results/\$TRACK/main_results.json" ]]; then
        mkdir -p "artifacts/\$TRACK"
        python3 scripts/benchmark_ci_report.py --track "\$TRACK" --output "artifacts/\$TRACK/ci_benchmark_report.json"
    fi
done
python3 -m pytest tests/test_dispatch_rule.py -v
echo

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Full Reproduction Complete                                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
date
INNER_EOF

chmod +x "${INNER_SCRIPT}"

# Mount REPO_ROOT to /repo (so the repo is at /repo inside container)
CONTAINER_INNER="/repo/results/run_inner_${SLURM_JOB_ID}.sh"

echo "Launching container job..."
"${CNT_BIN}" exec --nv --cleanenv \
    -B "${REPO_ROOT}:/repo" \
    "${CONTAINER_IMG}" \
    bash "${CONTAINER_INNER}"

echo
echo "Results: ${RESULTS_DIR}/upstream_patch/ and ${RESULTS_DIR}/latest_stack_tuned/"