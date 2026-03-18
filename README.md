# FlashAttention-3 Heuristic Fix — Reproduction Package

This repository provides a **reproduction package** for the FlashAttention-3 sequence-aware split heuristic. It includes the C++ patch, benchmarking harnesses, and a unified script to run all kernel-level experiments.

## Prerequisites

- Linux
- Hopper GPU (H100 or compatible)
- CUDA ≥ 12.3
- PyTorch ≥ 2.4.0

## Quick Start

```bash
git clone https://github.com/mllopartbsc/fa3-heuristic-fix.git
cd fa3-heuristic-fix

# Full reproduction (setup + all experiments)
python3 run_experiments.py --track upstream_patch --quick

# Skip setup if FA3 is already built
python3 run_experiments.py --skip-setup --track upstream_patch --quick

# Run both tracks
python3 run_experiments.py --track all --quick
```

## Unified Reproduction Script

`run_experiments.py` is the main entry point. It runs all kernel-level experiments in sequence:

| Experiment | Description |
|------------|-------------|
| exp1_correctness | Numerical correctness and determinism |
| exp2_profiling | Mechanism confirmation via profiling |
| main_results | Kernel-level latency (main results table) |
| guard_ablation | Guard ablation study |
| boundary_sweep | MQA crossover sweep around L_K boundary |
| u_curve_sweep | Extended split sweep (U-curve) |
| exp3_safety | Safety and regression profiling (160 configs) |
| threshold_sensitivity | (latest_stack_tuned track only) |

**Options:**

- `--track` — `upstream_patch`, `latest_stack_tuned`, or `all`
- `--skip-setup` — Skip cloning and building FA3 (assumes already built)
- `--quick` — Fewer iterations (~5–15 min per track)
- `--experiment NAME` — Run only a specific experiment

## Two Benchmark Routes

| Route | Track | Description | Expected speedup |
|-------|-------|-------------|------------------|
| **Route 1** | `latest_stack_tuned` | Policy injection + precomputed metadata (same binary) | ~1.18–1.25× |
| **Route 2** | `upstream_patch` | heuristics.h patch only (the upstream merge path) | ~1.20–1.24× |

Route 2 is what FA3 maintainers would merge: only the `heuristics.h` change. Precomputed metadata enabled. It is the canonical evidence track for the paper and FA3 pull request.

## Output Layout

- `results/<track>/` — JSON results
- `artifacts/<track>/` — Generated tables, validation reports
- `results/published/reviewer_artifacts/` — Committed reviewer artifacts; `upstream_patch/` is the paper/PR evidence bundle

## Refresh Reviewer Artifacts

After regenerating canonical results, refresh the committed reviewer bundle with:

```bash
python3 scripts/sync_published_artifacts.py --track upstream_patch
```

## Running on HPC (Slurm)

See [REPRODUCTION.md](REPRODUCTION.md) for step-by-step instructions. Quick reference: [RERUN_INSTRUCTIONS.md](RERUN_INSTRUCTIONS.md).

Edit `scripts/submit_slurm.sh` (YOUR_ACCOUNT, YOUR_QOS) once, then:

```bash
cd fa3-heuristic-fix
export CONTAINER_IMG=/path/to/vllm_openai.sif
bash scripts/run_hpc_job.sh
```

Or manually:

```bash
cd fa3-heuristic-fix
# 1. On login node (has network): clone FlashAttention
bash scripts/prepare_flash_attention.sh

# 2. Edit #SBATCH account/qos in scripts/submit_slurm.sh, then submit
export CONTAINER_IMG=/path/to/vllm_openai.sif
sbatch scripts/submit_slurm.sh
```

## Repository Structure

```
run_experiments.py           # Main entry point (unified reproduction)
reproduce.py                 # Legacy entry (delegates to run_experiments.py)
patch/heuristics.patch       # C++ heuristics patch
scripts/
  setup_environment.sh       # Build baseline and patched FA3
  run_experiments_inner.py   # Inner runner (experiments loop)
  prepare_flash_attention.sh # Clone FA3 (for HPC without GitHub on compute nodes)
  submit_slurm.sh            # Slurm job script (edit account/qos for your cluster)
  run_hpc_job.sh             # One-command prepare + submit
  sync_published_artifacts.py # Refresh committed reviewer bundles from canonical outputs
  run_full_benchmark_suite.sh # Kernel + E2E jobs (see docs/BENCHMARKING.md)
  run_all.sh                 # Alternative: direct sbatch (no container)
experiments/                 # Individual experiment scripts
src/                         # Benchmark utilities, heuristics reference
docs/                        # Methodology, E2E decode, technical details
```

## Regression Tests

```bash
python3 -m pytest tests/test_dispatch_rule.py -v
```

## Notes

- All builds are Hopper-only (SM90).
- End-to-end (vLLM) experiments require separate infrastructure. See [docs/E2E_DECODE.md](docs/E2E_DECODE.md) and [docs/BENCHMARKING.md](docs/BENCHMARKING.md).
- See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for both routes explained.