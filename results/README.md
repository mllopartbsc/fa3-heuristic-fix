# Results Directory

Benchmark outputs. By default gitignored (scratch space).

**Published artifacts** for reviewers without H100: [`results/published/reviewer_artifacts/`](published/reviewer_artifacts/).

## Run Full Benchmark

1. Run `bash scripts/prepare_flash_attention.sh` once (on login node).
2. Edit `scripts/submit_slurm.sh` (YOUR_ACCOUNT, YOUR_QOS).
3. From repo root:

```bash
export CONTAINER_IMG=/path/to/vllm_openai.sif
sbatch scripts/submit_slurm.sh
```

Output layout:

```
results/
  upstream_patch/       # main_results.json, exp3_safety_verification.json, etc.
  latest_stack_tuned/
  slurm_full_run_<job_id>.out
```

## Publish Results

```bash
cp results/upstream_patch/*.json results/published/reviewer_artifacts/upstream_patch/
cp results/latest_stack_tuned/*.json results/published/reviewer_artifacts/latest_stack_tuned/
cp artifacts/upstream_patch/*.json artifacts/upstream_patch/*.tex results/published/reviewer_artifacts/upstream_patch/
cp artifacts/latest_stack_tuned/*.json artifacts/latest_stack_tuned/*.tex results/published/reviewer_artifacts/latest_stack_tuned/
git add -f results/published/
git commit -m "Add benchmark results"
```