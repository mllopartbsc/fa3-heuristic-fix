#!/bin/bash
# ============================================================================
# Run All Experiments: FlashAttention-3 Heuristic Fix Reproduction
#
# Runs all experiments from the paper sequentially and saves results to
# the results/ directory as JSON files.
#
# Usage:
#   bash scripts/run_all.sh                    # Run all experiments
#   bash scripts/run_all.sh --quick            # Quick mode (fewer iterations, ~5-10 min)
#   bash scripts/run_all.sh --experiment main_results  # Run one experiment
#
# For SLURM clusters:
#   sbatch scripts/run_all.sh
#
# Tip: prefer python3 reproduce.py for a fully automated end-to-end run
#      (includes build, validation, and LaTeX table generation).
#
# Expected runtime on H100 SXM5:
#   - Quick mode: ~5-10 minutes
#   - Full mode:  ~45-60 minutes
# ============================================================================

#SBATCH --job-name=fa3-repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/run_all_%j.log
#SBATCH --error=results/run_all_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPRO_PYDEPS="$REPO_ROOT/.pydeps"
cd "$REPO_ROOT"

export PYTHONPATH="$REPRO_PYDEPS${PYTHONPATH:+:$PYTHONPATH}"

# Create results directory
mkdir -p results

# Parse arguments
QUICK=false
SINGLE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=true; shift ;;
        --experiment) SINGLE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  FlashAttention-3 Heuristic Fix — Reproduction Suite            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

if $QUICK; then
    echo "*** QUICK MODE: reduced iterations (~5-10 min, less statistically precise) ***"
    echo
fi

# Print environment info
python3 -c "
import torch
props = torch.cuda.get_device_properties(0)
print(f'Device:     {props.name} ({props.multi_processor_count} SMs)')
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA:       {torch.version.cuda}')
try:
    import flash_attn_interface
    print(f'FA3:        {flash_attn_interface.__file__}')
except ImportError:
    print('FA3:        NOT FOUND — run scripts/setup_environment.sh or python3 reproduce.py first')
    exit(1)
"
echo

START_TIME=$(date +%s)

# ── Build the extra args passed to every experiment ─────────────────────────
# All experiment scripts accept --quick (reduces iterations for a fast sanity check)
QUICK_FLAG=""
if $QUICK; then
    QUICK_FLAG="--quick"
fi

run_experiment() {
    local name="$1"
    local script="$2"
    local extra_args="${3:-}"

    if [[ -n "$SINGLE" && "$SINGLE" != "$name" ]]; then
        return 0
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local exp_start=$(date +%s)
    # All experiments receive QUICK_FLAG so they can reduce iterations
    python3 "$script" $QUICK_FLAG $extra_args 2>&1 | tee "results/${name}.log"
    local exp_end=$(date +%s)
    local elapsed=$((exp_end - exp_start))

    echo "  Completed in ${elapsed}s"
    echo
}

# ── Paper Table/Figure mapping ───────────────────────────────────────────────
# Experiment 1 (Section 4.1)  → exp1_correctness.json
# Experiment 2 (Section 4.2)  → exp2_profiling.json
# Table 5 (Main Results)      → main_results.json
# Table 8 (Guard Ablation)    → guard_ablation.json
# Table 9 (Boundary Sweep)    → boundary_sweep.json
# Figure 2b (U-curve)         → u_curve_sweep.json
# Table 6 (E2E Decode)        → e2e_decode_simulation.json
# Table 7 (Safety Contract)   → exp3_safety_verification.json
# Appendix Table 10           → threshold_sensitivity.json

# Run experiments in order of paper appearance
run_experiment "exp1_correctness"       "experiments/exp1_correctness.py"
run_experiment "exp2_profiling"         "experiments/exp2_mechanism_profiling.py"
run_experiment "main_results"           "experiments/main_results.py"
run_experiment "guard_ablation"         "experiments/guard_ablation.py"
run_experiment "boundary_sweep"         "experiments/boundary_sweep.py"
run_experiment "u_curve"                "experiments/u_curve_sweep.py"
run_experiment "e2e_simulation"         "experiments/e2e_decode_simulation.py"
run_experiment "exp3_safety"            "experiments/exp3_safety_verification.py"
run_experiment "threshold_sensitivity"  "experiments/threshold_sensitivity.py"

# ── Generate LaTeX tables ────────────────────────────────────────────────────
if [[ -z "$SINGLE" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Generating LaTeX tables..."
    python3 scripts/generate_tables.py \
        --results-dir results/ \
        --output-tex results/tables.tex 2>&1 | tee results/generate_tables.log
    echo
fi

# ── Summary ──────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  All experiments complete!                                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo
echo "Total runtime: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s)"
echo
echo "Results saved to:"
ls -la results/*.json 2>/dev/null || echo "  (no JSON results found)"
echo
echo "LaTeX tables: results/tables.tex"
echo
echo "To validate headline claims:"
echo "  python3 src/validate_claims.py"
echo
echo "For full automated reproduction (build + run + validate):"
echo "  python3 reproduce.py"
