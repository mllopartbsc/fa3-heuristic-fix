# Published Reviewer Artifacts

Benchmark results for reviewers without H100 access.

## Contents

- **upstream_patch/**: Canonical reviewer bundle for the upstream two-guard patch track used in the paper and FA3 pull request

## Measurement Protocol

- **upstream_patch** (Route 2): Heuristics.h patch only + precomputed metadata. What FA3 merges. Expect ~1.20–1.24× in win regime.
- **latest_stack_tuned** (Route 1): Policy injection with precomputed metadata. Kept as an optional local benchmark route rather than a committed reviewer bundle.

## Regenerate

```bash
bash scripts/prepare_flash_attention.sh   # once, on login node
# Edit scripts/submit_slurm.sh (YOUR_ACCOUNT, YOUR_QOS), then:
export CONTAINER_IMG=/path/to/vllm_openai.sif
sbatch scripts/submit_slurm.sh
```

Then sync the committed reviewer artifacts:

```bash
python3 scripts/sync_published_artifacts.py --track upstream_patch
# Use --track all only if you intentionally want to publish both routes.
```
