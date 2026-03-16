#!/bin/bash
# ============================================================================
# Run All Experiments: FlashAttention-3 Heuristic Fix Reproduction
#
# Track-aware execution entry point. Results are written to results/<track>/ and
# reviewer-visible generated tables are written to artifacts/<track>/.
# ============================================================================

#SBATCH --job-name=fa3-repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# ^ Edit partition; add --account and --qos if your cluster requires them
#SBATCH --time=02:00:00
#SBATCH --output=results/run_all_%j.log
#SBATCH --error=results/run_all_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

QUICK=false
SINGLE=""
TRACK="upstream_patch"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=true; shift ;;
        --track) TRACK="$2"; shift 2 ;;
        --experiment) SINGLE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$TRACK" in
    upstream_patch)
        PROFILE_ROOT="$REPO_ROOT/.pydeps/upstream_patch"
        ;;
    latest_stack_tuned)
        PROFILE_ROOT="$REPO_ROOT/.pydeps/baseline"
        ;;
    *)
        echo "Unsupported track: $TRACK"
        exit 1
        ;;
esac

RESULTS_DIR="$REPO_ROOT/results/$TRACK"
ARTIFACT_DIR="$REPO_ROOT/artifacts/$TRACK"
mkdir -p "$RESULTS_DIR" "$ARTIFACT_DIR"

export PYTHONPATH="$PROFILE_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROUTE=$([ "$TRACK" = "upstream_patch" ] && echo "Route 2: heuristics.h patch + precomputed metadata (upstream merge)" || echo "Route 1: policy injection + precomputed metadata")
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  FlashAttention-3 Heuristic Fix — Reproduction Suite            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "Track: $TRACK ($ROUTE)"
echo "Runtime profile: $PROFILE_ROOT"
echo

if $QUICK; then
    echo "*** QUICK MODE: reduced iterations (~5-10 min, less statistically precise) ***"
    echo
fi

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
    print('FA3:        NOT FOUND — run scripts/setup_environment.sh or python3 run_experiments.py first')
    exit(1)
"
echo

START_TIME=$(date +%s)
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
    python3 "$script" --track "$TRACK" --results-dir "$RESULTS_DIR" $QUICK_FLAG $extra_args \
        2>&1 | tee "$RESULTS_DIR/${name}.log"
    local exp_end=$(date +%s)
    local elapsed=$((exp_end - exp_start))
    echo "  Completed in ${elapsed}s"
    echo
}

run_experiment "exp1_correctness"      "experiments/exp1_correctness.py"
run_experiment "exp2_profiling"        "experiments/exp2_mechanism_profiling.py"
run_experiment "main_results"          "experiments/main_results.py"
run_experiment "guard_ablation"        "experiments/guard_ablation.py"
run_experiment "boundary_sweep"        "experiments/boundary_sweep.py"
run_experiment "u_curve_sweep"         "experiments/u_curve_sweep.py"
run_experiment "exp3_safety"           "experiments/exp3_safety_verification.py"
if [[ "$TRACK" == "latest_stack_tuned" ]]; then
    run_experiment "threshold_sensitivity" "experiments/threshold_sensitivity.py"
fi

if [[ -z "$SINGLE" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Generating reviewer tables..."
    python3 scripts/generate_tables.py \
        --track "$TRACK" \
        --results-dir "$RESULTS_DIR" \
        --output-tex "$ARTIFACT_DIR/tables.tex" 2>&1 | tee "$RESULTS_DIR/generate_tables.log"
    echo
fi

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  All experiments complete!                                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo
echo "Total runtime: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s)"
echo
echo "Results saved to: $RESULTS_DIR"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (no JSON results found)"
echo
echo "Reviewer artifacts: $ARTIFACT_DIR"
echo "To validate headline claims:"
echo "  python3 src/validate_claims.py --track $TRACK --results-dir $RESULTS_DIR"
echo
echo "For full automated reproduction:"
echo "  python3 run_experiments.py --track $TRACK"
