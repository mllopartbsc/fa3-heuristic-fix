# Step-by-Step Reproduction Guide

This guide explains how to reproduce the kernel-level experiments on any HPC with Slurm and Hopper GPUs (H100 or compatible).

## Prerequisites

- Access to an HPC with Hopper GPUs and Slurm
- A container image with PyTorch, CUDA, and nvcc (e.g. `vllm_openai.sif`)
- Your cluster account and partition configured in the Slurm script

## Option A: One-Command Run (HPC)

Edit `scripts/submit_slurm.sh` (YOUR_ACCOUNT, YOUR_QOS) once before first run. Then, from the repository root on the **login node**:

```bash
cd fa3-heuristic-fix
export CONTAINER_IMG=/path/to/your/vllm_openai.sif
bash scripts/run_hpc_job.sh
```

This will:

1. Clone FlashAttention into `flash-attention/` (requires network; run on login node)
2. Submit the Slurm job via `sbatch scripts/submit_slurm.sh`

## Option B: Manual Steps

### Step 1: Prepare FlashAttention (login node)

Compute nodes often cannot reach GitHub. Run this **once** on the login node:

```bash
cd fa3-heuristic-fix
bash scripts/prepare_flash_attention.sh
```

This clones `flash-attention` at commit `fbf24f67` into `flash-attention/`.

### Step 2: Submit the job

Edit `scripts/submit_slurm.sh` and set `YOUR_ACCOUNT` and `YOUR_QOS` for your cluster, then:

```bash
# From repo root
export CONTAINER_IMG=/path/to/vllm_openai.sif
sbatch scripts/submit_slurm.sh
```

### Step 3: Monitor the job

```bash
squeue -u $USER
# Or: squeue -j <job_id>
```

### Step 4: Check results

After the job completes:

- **Logs:** `results/slurm_full_run_<job_id>.out` and `.err`
- **Results:** `results/upstream_patch/` and `results/latest_stack_tuned/`
- **Artifacts:** `artifacts/upstream_patch/` and `artifacts/latest_stack_tuned/`

## Option C: Local / Interactive Run

If you have direct access to an H100 (e.g., interactive session):

```bash
cd fa3-heuristic-fix

# Full run (setup + experiments)
python3 run_experiments.py --track upstream_patch

# Quick sanity check (~5–15 min)
python3 run_experiments.py --track upstream_patch --quick

# Both tracks
python3 run_experiments.py --track all --quick

# Skip setup (FA3 already built)
python3 run_experiments.py --skip-setup --track all --quick
```

## Slurm Job Configuration

Edit the `#SBATCH` directives at the top of `scripts/submit_slurm.sh` for your cluster:

| Parameter | Placeholder | Description |
|-----------|-------------|--------------|
| Partition | `acc` | Your cluster's partition |
| Account | `YOUR_ACCOUNT` | Your project account |
| QOS | `YOUR_QOS` | Queue of service |
| GPUs | 1 | Hopper GPU |
| CPUs | 20 | CPUs per task |
| Time | 2 hours | Job time limit |
| Output | `results/slurm_full_run_%j.out` | Relative to cwd when sbatch runs |

Run `sbatch` from the repo root so output paths resolve correctly.

## What the Job Runs

1. **Phase 1:** Build baseline and patched FA3 profiles (Hopper-only)
2. **Phase 2:** Run all kernel experiments for both tracks (`--quick` mode)
3. **Phase 3:** Generate CI reports and run regression tests

Experiments: exp1_correctness, exp2_profiling, main_results, guard_ablation, boundary_sweep, u_curve_sweep, exp3_safety, threshold_sensitivity (latest_stack_tuned only).

## Troubleshooting

### "FlashAttention clone not found"

Run `bash scripts/prepare_flash_attention.sh` on the **login node** (has network). Compute nodes often cannot reach GitHub.

### "Container not found" / CONTAINER_IMG

Set `CONTAINER_IMG` to your container image path before `sbatch`:

```bash
export CONTAINER_IMG=/path/to/vllm_openai.sif
sbatch scripts/submit_slurm.sh
```

### "nvcc not found" or build failures

The container must provide CUDA and nvcc. Verify with:

```bash
singularity exec --nv your_container.sif nvcc --version
```

### Job timeout

For full (non-quick) runs, increase `#SBATCH --time` in `submit_slurm.sh`. Quick mode typically finishes in ~30–60 minutes.