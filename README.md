# FlashAttention-3 Tile-Aware Split Heuristic Fix

This repository contains the reproduction package for the **Tile-Aware Split Heuristic Fix** for FlashAttention-3 on modern NVIDIA Hoppers (H100/H800).

The proposed two-line patch optimally adapts the work-splitting heuristic in low-tile decode scenarios—specifically when `L_K` is between 448 and 512, and `H_KV` is 1 or 2 (e.g., Llama-3 70B MQA/GQA regimes). It yields up to a **1.2x kernel-level speedup** in this specific boundary while maintaining a strictly safe **0 regressions** profile across a 160-configuration tested matrix.

## 💡 Motivation & Industry Impact

Modern Large Language Models (LLMs) like Llama-3 and Claude rely on **Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)** to reduce memory usage during decoding. However, for short context lengths ($L_K \le 512$), this drastically reduces the attention workload per generation step. 

The baseline FlashAttention-3 heuristic miscalculates this specific, highly-common MQA/GQA decoding shape, assigning as few as **8 blocks of work** to an H100 GPU that possesses **132 Streaming Multiprocessors (SMs)**. This causes a severe occupancy collapse, leaving the vast majority of the GPU idle. 

By replacing the naive block with a mathematically rigorous, tile-aware heuristic, this patch correctly identifies the starvation condition and splits the workload across the sequence dimension. The result is a **20% to 24% kernel-level speedup** on Hopper architecture, which translates directly to massive CapEx/OpEx savings and reduced latency (TTFT/TPOT) for at-scale LLM inference deployments.

---

## 🚀 Quick Start: Automated Reproduction

We provide a single, fully-automated entry point script that clones FlashAttention-3, applies the patch, builds both baseline and patched versions, runs all benchmarking experiments, generates LaTeX tables, and validates the results against the expected claims.

### Option 1: Native Environment (Requires CUDA & PyTorch)

**Prerequisites:** Ubuntu/Linux, H100 GPU, CUDA $\ge$ 12.3, PyTorch $\ge$ 2.4.0.

```bash
git clone <this-repository>
cd fa3-heuristic-fix

# Run the full automated suite (~5-10 minutes in quick mode)
python3 reproduce.py --quick

# Or run the full statistically-precise suite (~45-60 minutes)
# python3 reproduce.py
```

### Option 2: Docker Container (Fully Isolated)

If you prefer an isolated environment with exact dependency versions:

```bash
cd fa3-heuristic-fix
docker build -t fa3-repro .
docker run --gpus all -it fa3-repro
```

### Option 3: HPC / SLURM Apptainer (Singularity)

For multi-tenant HPC clusters that don't support Docker:

```bash
cd fa3-heuristic-fix
apptainer build fa3-repro.sif Apptainer.def
apptainer run --nv fa3-repro.sif --quick
```

---

## 📁 Repository Structure

The reproduction package is strictly curated to include only the essential files:

```
├── patch/
│   └── heuristics.patch                # The proposed two-line C++ fix
├── reproduce.py                        # Automated entry point orchestrator
├── README.md                           # This document
├── requirements.txt                    # Minimal dependencies
├── Dockerfile / Apptainer.def          # Isolated container builds
├── configs/
│   └── experiment_params.yaml          # Benchmarking parameter definitions
├── expected_results/
│   └── claims.json                     # Tolerance limits for CI validation
├── docs/
│   ├── PR_EVIDENCE.md                  # Concise reviewer evidence packet
│   └── LATEST_STACK_FINDINGS.md        # Software stack updates analysis
├── scripts/
│   ├── setup_environment.sh            # FA3 clone, patch, and dual-build
│   ├── run_all.sh                      # Execute all experiment scripts
│   └── generate_tables.py              # Export JSON results to LaTeX
├── src/
│   ├── heuristics_reference.py         # Python implementations of policies
│   ├── bench_utils.py                  # CUDA-Graph timing utilities
│   └── validate_claims.py              # Automated assertion validation
└── experiments/
    ├── exp1_correctness.py             # 1,000-trial FP64 equivalence test
    ├── exp2_mechanism_profiling.py     # SM active-warp mechanism profiling
    ├── exp3_safety_verification.py     # 160-cfg zero-regression sweep
    ├── main_results.py                 # Kernel latency A/B benchmark (Table 5)
    ├── guard_ablation.py               # Proof of two-guard design (Table 8)
    ├── boundary_sweep.py               # MQA L_K crossover behavior (Table 9)
    ├── u_curve_sweep.py                # Service latency vs splits (Fig 2b)
    ├── e2e_decode_simulation.py        # E2E step & TPOT estimates (Table 6)
    └── threshold_sensitivity.py        # Robustness of thresholds (Table 10)
```

---

## 🔍 Manual Execution

If you prefer to run individual components:

### 1. Build the Environments

```bash
# This will clone https://github.com/Dao-AILab/flash-attention
# and install dual versions into a local .pydeps/ directory.
bash scripts/setup_environment.sh
```

### 2. Run Experiments

You can run individual experiment scripts from the `experiments/` directory:

```bash
# Run the main headline kernel-latency benchmarks (Table 5)
python3 experiments/main_results.py

# Run the 160-configuration safety matrix sweep (Table 7)
python3 experiments/exp3_safety_verification.py

# Run the numerical correctness FP64-equivalence trials
python3 experiments/exp1_correctness.py
```

*Note: All python scripts accept a `--quick` flag to reduce iterations for a faster but less statistically precise test run.*

### 3. Verify Results

Experimental results are saved as `.json` files in the `results/` directory. You can automatically validate the generated JSON files against the expected tolerances defined in the publication:

```bash
python3 src/validate_claims.py
```

---

## 🛠 Benchmarking Methodology

All performance measurements in this reproduction package adhere to strict methodology to ensure reliability:

*   **CUDA Graphs:** Python dispatch overhead ($\sim$30-55 µs) is explicitly eliminated using CUDA-graph capture and replay to reveal pure kernel time.
*   **A/B Interleaving:** For any configuration where the policy alters work-split behavior, execution alternates back and forth between baseline and fix graphs to eliminate unobserved thermal/JIT bias.
*   **Significance Windows:** 10,000 sampling iterations are gathered, returning medians, $P_{05}$, and $P_{95}$ bounds to guarantee statistical significance.


