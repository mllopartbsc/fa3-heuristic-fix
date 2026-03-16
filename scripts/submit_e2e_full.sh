#!/bin/bash
# ============================================================================
# Full vLLM E2E Decode Benchmark — fa3-heuristic-fix
#
# Runs real vLLM-based decode benchmarks on
# representative GQA/MQA models. Requires 4× Hopper GPUs.
#
# Uses the parent reproduction package for benchmarks/, models/, tools/.
# Uses $REPRO/flash-attention-orig (unpatched) for baseline; applies heuristics.patch for patched.
# Do NOT use fa3-heuristic-fix/flash-attention — it is already patched (would compare patched vs patched).
#
# Usage:
#   export VLLM_CONTAINER=/path/to/vllm_openai.sif
#   sbatch scripts/submit_e2e_full.sh
#
# Run only the decode scenario where the fix applies (~3% expected, ~25 min):
#   MODELS=mock_attn_heavy_mqa sbatch scripts/submit_e2e_full.sh
#
# Results: results/e2e_full_<job_id>/
#   - FA3 proxy traces: fa3_proxy_trace_${variant}_${model}.pid*.json
#   - Summaries: fa3_proxy_summary_${variant}_${model}.json (target_decode_calls, effective_num_splits)
#   - If target_decode_calls>0 and baseline effective_num_splits=1 vs heuristic=3, the fix is verified.
#
# If trace proxy bind fails (wrong vLLM path), set VLLM_FAI_PATH to your container's
# vllm/vllm_flash_attn/flash_attn_interface.py path.
#
# Use latest vLLM: set VLLM_UPGRADE=1. First run on login node (has internet):
#   bash scripts/preinstall_vllm.sh
# Then: VLLM_UPGRADE=1 sbatch scripts/submit_e2e_full.sh
# Default VLLM_UPGRADE=0: use container's vLLM (0.11) and bind-mount the proxy.
# ============================================================================

#SBATCH --job-name=fa3-e2e-full
#SBATCH --output=results/e2e_full_%j.out
#SBATCH --error=results/e2e_full_%j.err
#SBATCH --partition=acc
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
# ^ Edit for your cluster

set -euo pipefail

# Requires: VLLM_CONTAINER (container image path), full reproduction package layout (REPRO with benchmarks/, models/).
# Run from repo root so output paths resolve.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FA3_FIX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPRO="${REPRO:-$(cd "$FA3_FIX_DIR/.." && pwd)}"
IMG="${VLLM_CONTAINER:-}"
if [[ -z "$IMG" ]] || [[ ! -f "$IMG" ]]; then
  echo "ERROR: VLLM_CONTAINER must point to an existing container image."
  exit 1
fi

cd "$FA3_FIX_DIR"
mkdir -p "${FA3_FIX_DIR}/results"

# Use reproduction package's flash-attention-orig for TRUE baseline (unpatched heuristics).
# fa3-heuristic-fix/flash-attention is already patched — using it would compare patched vs patched.
FA3_SRC="${FA3_SRC:-$REPRO/flash-attention-orig}"
PATCH_FILE="${FA3_FIX_DIR}/patch/heuristics.patch"
TRACE_PROXY="${REPRO}/shim/flash_attn_interface_trace_proxy.py"
VLLM_FAI_PATH="${VLLM_FAI_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/flash_attn_interface.py}"
VLLM_UPGRADE="${VLLM_UPGRADE:-0}"
VLLM_INSTALL_DIR="${VLLM_INSTALL_DIR:-$FA3_FIX_DIR/vllm_latest}"
BUILD_DIR="${FA3_FIX_DIR}/builds/e2e_full_${SLURM_JOB_ID}"
RUN_DIR="${FA3_FIX_DIR}/results/e2e_full_${SLURM_JOB_ID}"

# Space-separated list of models. Use MODELS="mock_attn_heavy_mqa" to run only the
# decode scenario where the fix applies (~3% expected). Default runs both (MQA + attn-heavy).
MODELS="${MODELS:-mock_mqa_llama mock_attn_heavy_mqa}"
NUM_REQUESTS="${NUM_REQUESTS:-300}"
INPUT_LEN="${INPUT_LEN:-384}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
TP_SIZE="${TP_SIZE:-4}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# Check FlashAttention source exists (default: $REPRO/flash-attention-orig)
if [[ ! -d "$FA3_SRC/hopper" ]]; then
    echo "ERROR: FlashAttention source not found at $FA3_SRC"
    echo "Expected: $REPRO/flash-attention-orig (unpatched baseline for true A/B comparison)"
    exit 1
fi

# Check parent has benchmarks and models
if [[ ! -f "$REPRO/benchmarks/benchmark_production.py" ]]; then
    echo "ERROR: Parent reproduction package benchmarks not found at $REPRO/benchmarks/"
    echo "This script requires the full reproduction package layout."
    exit 1
fi
for m in $MODELS; do
    if [[ ! -d "$REPRO/models/$m" ]]; then
        echo "ERROR: Model not found: $REPRO/models/$m"
        echo "Available: $(ls -1 "$REPRO/models/" 2>/dev/null || echo 'none')"
        exit 1
    fi
done

if [[ ! -f "$TRACE_PROXY" ]]; then
    echo "ERROR: Trace proxy not found: $TRACE_PROXY"
    exit 1
fi

module load singularity/4.1.5 2>/dev/null || module load apptainer/1 2>/dev/null
CNT_BIN="$(command -v singularity || command -v apptainer)"
if [[ ! -x "$CNT_BIN" ]]; then
    echo "ERROR: singularity/apptainer not found"
    exit 1
fi

mkdir -p "$BUILD_DIR" "$RUN_DIR"

# When VLLM_UPGRADE=1, we upgrade vLLM and copy the proxy (no bind). Otherwise bind the proxy.
BIND_ARGS="--bind $REPRO:$REPRO"
if [[ "${VLLM_UPGRADE:-1}" != "1" ]]; then
  BIND_ARGS="$BIND_ARGS --bind $TRACE_PROXY:$VLLM_FAI_PATH"
fi

"$CNT_BIN" exec --nv --cleanenv \
  $BIND_ARGS \
  "$IMG" \
  bash -lc "
    set -euo pipefail
    cd \"$REPRO\"

    VLLM_UPGRADE=\"${VLLM_UPGRADE}\"
    TRACE_PROXY=\"$TRACE_PROXY\"

    if [[ \"\$VLLM_UPGRADE\" == \"1\" ]]; then
      echo '==== Use latest vLLM (from preinstall) for meaningful proof ===='
      export PYTHONPATH=\"$VLLM_INSTALL_DIR:\${PYTHONPATH:-}\"
      VLLM_FAI_PATH=\$(python3 -c 'import vllm.vllm_flash_attn.flash_attn_interface as m; print(m.__file__)') || {
        echo 'ERROR: vLLM not found. Run on login node first: bash scripts/preinstall_vllm.sh'
        echo '  (VLLM_INSTALL_DIR=$VLLM_INSTALL_DIR)'
        exit 1
      }
      echo \"vLLM version: \$(python3 -c 'import vllm; print(vllm.__version__)')\"
      echo \"Installing trace proxy at: \$VLLM_FAI_PATH\"
      cp \"\$TRACE_PROXY\" \"\$VLLM_FAI_PATH\"
    fi

    export CC=gcc
    export CXX=g++
    export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
    export FLASH_ATTENTION_DISABLE_FP8=TRUE
    export FLASH_ATTENTION_DISABLE_SM8x=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    export FLASH_ATTENTION_DISABLE_SM80=TRUE
    export MAX_JOBS=\${MAX_JOBS:-12}

    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export VLLM_FLASH_ATTN_VERSION=3

    echo '==== Build baseline wheel from fa3-heuristic-fix FA3 source ===='
    rm -rf \"$BUILD_DIR/baseline_src\"
    cp -r \"$FA3_SRC\" \"$BUILD_DIR/baseline_src\"
    python3 \"$FA3_FIX_DIR/scripts/apply_hopper_only_setup.py\" \"$BUILD_DIR/baseline_src/hopper/setup.py\" || true
    python3 \"$FA3_FIX_DIR/scripts/apply_hopper_scheduler_metadata.py\" \"$BUILD_DIR/baseline_src/hopper/flash_attn_interface.py\" || true
    python3 \"$FA3_FIX_DIR/scripts/apply_batch_size_mqa_fix.py\" \"$BUILD_DIR/baseline_src/hopper\" || true
    cd \"$BUILD_DIR/baseline_src/hopper\"
    python3 setup.py bdist_wheel
    BASELINE_WHEEL=\$(echo \"$BUILD_DIR\"/baseline_src/hopper/dist/*.whl)
    echo \"Baseline wheel: \$BASELINE_WHEEL\"

    echo '==== Build patched wheel using fa3-heuristic-fix patch ===='
    rm -rf \"$BUILD_DIR/patched_src\"
    cp -r \"$FA3_SRC\" \"$BUILD_DIR/patched_src\"
    if git -C \"$BUILD_DIR/patched_src\" apply --check \"$PATCH_FILE\" 2>/dev/null; then
      git -C \"$BUILD_DIR/patched_src\" apply \"$PATCH_FILE\"
    else
      echo '  (git apply failed, using apply_heuristics_patch.py)'
      python3 \"$FA3_FIX_DIR/scripts/apply_heuristics_patch.py\" \"$BUILD_DIR/patched_src/hopper/heuristics.h\" || {
        echo 'ERROR: Failed to apply heuristics patch.'
        exit 1
      }
    fi
    python3 \"$FA3_FIX_DIR/scripts/apply_hopper_only_setup.py\" \"$BUILD_DIR/patched_src/hopper/setup.py\" || true
    python3 \"$FA3_FIX_DIR/scripts/apply_hopper_scheduler_metadata.py\" \"$BUILD_DIR/patched_src/hopper/flash_attn_interface.py\" || true
    python3 \"$FA3_FIX_DIR/scripts/apply_batch_size_mqa_fix.py\" \"$BUILD_DIR/patched_src/hopper\" || true
    cd \"$BUILD_DIR/patched_src/hopper\"
    python3 setup.py bdist_wheel
    PATCHED_WHEEL=\$(echo \"$BUILD_DIR\"/patched_src/hopper/dist/*.whl)
    echo \"Patched wheel: \$PATCHED_WHEEL\"

    run_variant () {
      local wheel=\"\$1\"
      local variant=\"\$2\"
      local model=\"\$3\"

      echo \"==== Running \$variant on \$model ====\"
      pip uninstall -y flash_attn_3 flash_attn >/dev/null 2>&1 || true
      pip install --force-reinstall --no-deps \"\$wheel\"
      cd \"$REPRO\"
      python3 tools/print_fa3_runtime_fingerprint.py \
        --variant \"\$variant\" \
        --wheel \"\$wheel\"

      unset OE_BASELINE_MODE FLASH_ATTENTION_FA3_INTERFACE_PATH REPO_DIR PYTHONPATH
      export OE_FA3_TRACE_PATH=\"$RUN_DIR/fa3_proxy_trace_\${variant}_\${model}.json\"
      export OE_FA3_TRACE_STDOUT=\"${OE_FA3_TRACE_STDOUT:-0}\"

      export MODEL_PATH=\"$REPRO/models/\$model\"
      export FA3_VARIANT=\"\$variant\"
      export NUM_REQUESTS=\"$NUM_REQUESTS\"
      export INPUT_LEN=\"$INPUT_LEN\"
      export OUTPUT_LEN=\"$OUTPUT_LEN\"
      export TP_SIZE=\"$TP_SIZE\"
      export ENFORCE_EAGER=\"$ENFORCE_EAGER\"

      cd \"$REPRO\"
      python3 benchmarks/benchmark_production.py
      cp \"results/production_\${variant}_\${model}.json\" \"$RUN_DIR/\"

      python3 \"$REPRO/tools/summarize_vllm_fa3_proxy_traces.py\" \
        --glob \"$RUN_DIR/fa3_proxy_trace_\${variant}_\${model}.pid*.json\" \
        --out \"$RUN_DIR/fa3_proxy_summary_\${variant}_\${model}.json\" || true
    }

    PROD_FILES=\"\"
    for m in $MODELS; do
      run_variant \"\$BASELINE_WHEEL\" baseline \"\$m\"
      run_variant \"\$PATCHED_WHEEL\" heuristic \"\$m\"
      PROD_FILES=\"\$PROD_FILES$RUN_DIR/production_baseline_\$m.json:$RUN_DIR/production_heuristic_\$m.json:\"
    done
    PROD_FILES=\"\${PROD_FILES%:}\"

    export RESULTS_DIR=\"$RUN_DIR\"
    export PRODUCTION_RESULT_FILES=\"\$PROD_FILES\"
    cd \"$REPRO\"
    python3 benchmarks/analyze_production.py
    cp \"$RUN_DIR/production_summary.json\" \"$FA3_FIX_DIR/results/e2e_full_summary_${SLURM_JOB_ID}.json\"

    echo ''
    echo '==== Decode-scenario verification (target: L_K 385-512, tiles<4) ===='
    for m in $MODELS; do
      for v in baseline heuristic; do
        f=\"$RUN_DIR/fa3_proxy_summary_\${v}_\${m}.json\"
        if [[ -f \"\$f\" ]]; then
          target=\$(python3 -c \"import json; d=json.load(open('\$f')); print(d.get('target_decode_calls',0))\" 2>/dev/null || echo '?')
          splits=\$(python3 -c \"import json; d=json.load(open('\$f')); print(d.get('target_decode_effective_num_splits',{}))\" 2>/dev/null || echo '{}')
          echo \"  \$v \$m: target_decode_calls=\$target  effective_num_splits=\$splits\"
        fi
      done
    done
    echo ''

    echo \"Run artifacts stored in: $RUN_DIR\"
  "

echo "Done. Results: $RUN_DIR"
