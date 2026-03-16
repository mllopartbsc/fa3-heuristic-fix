# Rerun Instructions

For full step-by-step instructions, see [REPRODUCTION.md](REPRODUCTION.md).

## Quick Reference

### Full reproduction (kernel experiments only)

```bash
cd fa3-heuristic-fix
python3 run_experiments.py --track all --quick
```

### HPC (one command from login node)

Edit `scripts/submit_slurm.sh` once, then:

```bash
cd fa3-heuristic-fix
export CONTAINER_IMG=/path/to/vllm_openai.sif
bash scripts/run_hpc_job.sh
```

### HPC (manual)

```bash
cd fa3-heuristic-fix
bash scripts/prepare_flash_attention.sh   # once, on login node
# Edit scripts/submit_slurm.sh (YOUR_ACCOUNT, YOUR_QOS), then:
export CONTAINER_IMG=/path/to/vllm_openai.sif
sbatch scripts/submit_slurm.sh
```

### Output locations

| Content | Path |
|---------|------|
| Results (upstream_patch) | `results/upstream_patch/` |
| Results (latest_stack_tuned) | `results/latest_stack_tuned/` |
| Artifacts | `artifacts/<track>/` |
| Slurm logs | `results/slurm_full_run_<job_id>.out` |