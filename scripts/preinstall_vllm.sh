#!/bin/bash
# ============================================================================
# Pre-install latest vLLM (run on login node with internet)
#
# Installs to VLLM_INSTALL_DIR (default: project's vllm_latest/) to avoid home
# quota. Compute nodes have no network; run this once, then VLLM_UPGRADE=1.
#
# Usage: bash scripts/preinstall_vllm.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FA3_FIX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_INSTALL_DIR="${VLLM_INSTALL_DIR:-$FA3_FIX_DIR/vllm_latest}"
IMG="${VLLM_CONTAINER:-}"
if [[ -z "$IMG" ]] || [[ ! -f "$IMG" ]]; then
  echo "ERROR: VLLM_CONTAINER must point to an existing container image."
  echo "  export VLLM_CONTAINER=/path/to/vllm_openai.sif"
  exit 1
fi

module load singularity/4.1.5 2>/dev/null || module load apptainer/1 2>/dev/null || true
CNT_BIN="$(command -v singularity 2>/dev/null || command -v apptainer 2>/dev/null)"
if [[ ! -x "$CNT_BIN" ]]; then
  echo "ERROR: singularity/apptainer not found"
  exit 1
fi

# Use a writable temp dir for Singularity. On some clusters you may need to set
# SINGULARITY_TMPDIR to a location with sufficient space (e.g. project scratch).
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${TMPDIR:-$HOME}/singularity_tmp}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SINGULARITY_TMPDIR}"
export SINGULARITY_LOCALCACHEDIR="${SINGULARITY_LOCALCACHEDIR:-$SINGULARITY_TMPDIR}"
mkdir -p "$SINGULARITY_TMPDIR"

mkdir -p "$VLLM_INSTALL_DIR"
echo "Installing latest vLLM to $VLLM_INSTALL_DIR (requires internet)..."
if ! "$CNT_BIN" exec --nv --bind "$VLLM_INSTALL_DIR:$VLLM_INSTALL_DIR" "$IMG" \
  pip install --target "$VLLM_INSTALL_DIR" vllm --upgrade; then
  echo ""
  echo "If you see 'failed to resolve session directory': set SINGULARITY_TMPDIR to a writable path."
  echo "If you see 'Disk quota exceeded': set VLLM_INSTALL_DIR to a dir with more space."
  exit 1
fi

echo ""
echo "vLLM version:"
"$CNT_BIN" exec --nv --bind "$VLLM_INSTALL_DIR:$VLLM_INSTALL_DIR" "$IMG" \
  python3 -c "import sys; sys.path.insert(0, '$VLLM_INSTALL_DIR'); import vllm; print(vllm.__version__)"

echo ""
echo "Done. Now run: VLLM_UPGRADE=1 sbatch scripts/submit_e2e_full.sh"
echo "  (or set VLLM_INSTALL_DIR=$VLLM_INSTALL_DIR if you use a custom path)"
