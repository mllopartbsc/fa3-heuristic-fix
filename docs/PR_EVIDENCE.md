# PR Evidence Packet

## Purpose

This document provides the reviewer-ready technical evidence for the FlashAttention-3 heuristic fix. It summarizes the scope, evidence, and reproducibility path for the Pull Request.

---

## Recommended PR Scope

This PR introduces a **kernel-scoped improvement** for low-tile FA3 decode on the modern software stack:

1. **Safety First:** Zero regressions across a 160-configuration tested matrix.
2. **Targeted Win:** Improves service latency in the specific low-tile decode regime (L_K = 448-512, H_KV = 1 or 2).
3. **End-to-End Context:** While the kernel-level speedup is significant (up to 1.2x), end-to-end serving gains in frameworks like vLLM will be modest and bound by Amdahl's Law (expected 0-3% TPOT reduction depending on the model pipeline).

This PR does **not** claim large generic end-to-end serving throughput gains across arbitrary workloads.

---

## Strongest Claims Supported by Evidence

1. **Kernel Win:** In the target low-tile decode kernel regime (e.g., L_K=512, H_KV=1), the tuned split policy produces a real kernel win on the modern stack.
2. **Updated Tuning:** The modern-stack optimum for the low-tile `nblk=4` case is `3` splits rather than `4` (which was optimal on older stacks).
3. **Integration Proof:** Worker-side vLLM runs show that the FA3 varlen decode path is reached, traced with concrete shape metadata, and successfully imports the patched backend.
4. **End-to-End Reality:** Small or absent end-to-end serving gains do not invalidate the kernel result; they are consistent with Amdahl limits and with regime mismatch in broader serving workloads.

---

## Concrete Evidence

### 1. Kernel-Level Benchmark Evidence

The reproduction package (`experiments/main_results.py`) demonstrates the main result on H100 SXM5:

- **MQA (H_KV=1, L_K=512):** `13.48 µs -> 11.15 µs` (1.208x speedup)
- **GQA-2 (H_KV=2, L_K=512):** `13.30 µs -> 10.83 µs` (1.227x speedup)

### 2. Safety Verification

A comprehensive 160-configuration matrix sweep (`experiments/exp3_safety_verification.py`) confirms:
- **0 Regressions** (no slowdowns < 0.97x)
- **3 Wins** (speedups > 1.03x in target regimes)
- **157 Neutral / Unchanged** setups

### 3. End-to-End Interpretation

Using the measured kernel delta:
- ~2.33 µs saved per layer in the best case (H_KV=1, L_K=512).
- On an 80-layer model, this is ~0.186 ms/token.
- On a ~28.7 ms/token whole-stack step (vLLM), the theoretical ceiling is ~0.65% TPOT improvement.
- This confirms that the kernel win is mathematically consistent with modest serving-level improvements.

---

## Minimal Reproduction Commands

The full suite of experiments verifying these claims can be run using the automated entry point in this repository:

```bash
# Clone this reproduction package
git clone <repository-url>
cd fa3-heuristic-fix

# Run the full suite (builds FA3, runs benchmarks, generates validation report)
python3 reproduce.py --quick
```

For detailed setup instructions and granular execution, see the `README.md`.
