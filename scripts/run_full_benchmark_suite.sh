#!/bin/bash
# ============================================================================
# Full Benchmark Suite: kernel-level + full vLLM E2E
#
# Runs the complete fa3-heuristic-fix benchmark suite:
#   1. Kernel job (1 GPU): kernel-level experiments, regression tests
#   2. Full E2E job (4 GPUs): real vLLM decode on GQA/MQA models
#
# Run from the fa3-heuristic-fix repo root.
# Requires: CONTAINER_IMG (kernel job), VLLM_CONTAINER (E2E job). Edit both scripts' #SBATCH for your cluster.
#
# Usage:
#   bash scripts/run_full_benchmark_suite.sh           # submit both jobs
#   bash scripts/run_full_benchmark_suite.sh --kernel-only
#   bash scripts/run_full_benchmark_suite.sh --e2e-only
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_KERNEL=true
RUN_E2E=true

for arg in "$@"; do
    case "$arg" in
        --kernel-only) RUN_E2E=false ;;
        --e2e-only)   RUN_KERNEL=false ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  fa3-heuristic-fix — Full Benchmark Suite                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

echo "Step 1: Preparing FlashAttention clone (needs network)..."
bash scripts/prepare_flash_attention.sh
echo

if $RUN_KERNEL; then
    echo "Step 2a: Submitting kernel-level job (1 GPU)..."
    if [[ -z "${CONTAINER_IMG:-}" ]]; then
        echo "ERROR: CONTAINER_IMG required. export CONTAINER_IMG=/path/to/container.sif"
        exit 1
    fi
    KERNEL_JOB=$(sbatch scripts/submit_slurm.sh | awk '{print $4}')
    echo "  Kernel job ID: $KERNEL_JOB"
    echo
fi

if $RUN_E2E; then
    echo "Step 2b: Submitting full vLLM E2E job (4 GPUs)..."
    if [[ -z "${VLLM_CONTAINER:-}" ]]; then
        echo "ERROR: VLLM_CONTAINER required for E2E. export VLLM_CONTAINER=/path/to/vllm_openai.sif"
        exit 1
    fi
    E2E_JOB=$(sbatch scripts/submit_e2e_full.sh | awk '{print $4}')
    echo "  E2E job ID: $E2E_JOB"
    echo
fi

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Jobs submitted                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
$RUN_KERNEL && echo "  Kernel:  results/slurm_full_run_*.out, results/upstream_patch/, results/latest_stack_tuned/"
$RUN_E2E   && echo "  Full E2E:      results/e2e_full_*.out, results/e2e_full_*/"
echo
echo "Monitor: squeue"
