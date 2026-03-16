# End-to-End Decode Latency on Representative GQA/MQA Models

This document describes how to run and interpret end-to-end decode latency benchmarks for representative GQA/MQA models.

**Build scope:** All FA3 builds in this repo are Hopper-only (SM90). A Hopper GPU (e.g. H100) is required.

## Overview

The fa3-heuristic-fix repository provides two levels of decode benchmarks:

1. **Kernel-level** (in-repo): `experiments/e2e_decode_simulation.py` produces theoretical TPOT estimates from the single-layer kernel delta. These are serving estimates, not measured end-to-end results.

2. **Full E2E** (reproduction package): The parent reproduction package runs real vLLM-based decode benchmarks with baseline vs patched FA3 wheels on representative models.

## Running Full E2E Decode Benchmarks

The full E2E benchmarks are **included in this repository**. From the fa3-heuristic-fix repo root:

```bash
# One-time: prepare FlashAttention clone (login node)
bash scripts/prepare_flash_attention.sh

# Edit scripts/submit_e2e_full.sh (YOUR_ACCOUNT, YOUR_QOS), then:
export VLLM_CONTAINER=/path/to/vllm_openai.sif
sbatch scripts/submit_e2e_full.sh
```

Or run the complete benchmark suite (kernel-level + full E2E):

```bash
export CONTAINER_IMG=/path/to/vllm_openai.sif
export VLLM_CONTAINER=/path/to/vllm_openai.sif
bash scripts/run_full_benchmark_suite.sh
```

The full E2E job expects a parent directory (by default `fa3-heuristic-fix/..`) containing `benchmarks/`, `models/`, and `tools/`. Set `REPRO` if your layout differs.

### Job Configuration

The script builds baseline and patched FA3 wheels, then runs vLLM with:

- **Models**: MQA and GQA-style workloads (configurable via `MODELS`)
- **Metrics**: Per-request TPOT (time per output token), statistical comparison
- **Output**: `results/e2e_full_<job_id>/`

### Requirements

- 4× GPU (Hopper; builds are Hopper-only), Slurm partition with `acc` or equivalent
- Singularity/Apptainer with the vLLM container
- FlashAttention clone at `fa3-heuristic-fix/flash-attention` (run `prepare_flash_attention.sh` first)
- Parent reproduction package with `benchmarks/`, `models/`, `tools/`

## Kernel-Level E2E Simulation (In-Repo)

For a quick sanity check without full vLLM:

```bash
cd fa3-heuristic-fix
python3 experiments/e2e_decode_simulation.py --track upstream_patch --quick
python3 experiments/e2e_decode_simulation.py --track latest_stack_tuned --quick
```

This measures the single-layer kernel delta and extrapolates to a theoretical TPOT. Results are written to `results/<track>/e2e_decode_simulation.json`.

## Representative Configurations

| Model Type | H_KV | Regime | Expected Impact |
|------------|------|--------|-----------------|
| MQA (Llama-3 70B decode) | 1 | Win | Patch improves TPOT |
| GQA-2 (e.g. 2 KV heads) | 2 | Win | Patch improves TPOT |
| GQA-8 | 8 | Safe | Neutral (Guard 2) |

The full E2E job uses models that exercise these regimes to validate the kernel-level findings at the serving layer.
