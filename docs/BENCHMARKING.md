# Benchmarking Guide

## Two Routes

- **Route 1** (`latest_stack_tuned`): Policy injection with precomputed metadata. Same binary.
- **Route 2** (`upstream_patch`): Baseline vs patched binary, both with precomputed metadata. Shows the patch merged upstream with metadata on.

## Quick Run

```bash
bash scripts/setup_environment.sh
python3 run_experiments.py --track upstream_patch --quick   # Route 2
python3 run_experiments.py --track latest_stack_tuned --quick  # Route 1
```

## Full Suite

```bash
export CONTAINER_IMG=/path/to/vllm_openai.sif
export VLLM_CONTAINER=/path/to/vllm_openai.sif   # for E2E job
bash scripts/run_full_benchmark_suite.sh
```

Or `--kernel-only` / `--e2e-only` to run just one part.

## Expected Results

| Track | Route | Win regime speedup |
|-------|-------|--------------------|
| upstream_patch | 2 | ~1.20–1.24× |
| latest_stack_tuned | 1 | ~1.18–1.25× |

## Validate Claims

```bash
python3 src/validate_claims.py --track upstream_patch
```

## Reviewer Artifacts

`results/published/reviewer_artifacts/` — Committed results for reviewers without H100. `upstream_patch/` is the canonical paper/PR evidence bundle.
