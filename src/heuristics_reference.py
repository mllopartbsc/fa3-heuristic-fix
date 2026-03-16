"""
Reference Python implementations of the split-selection policies discussed in
the reproduction package.

These functions mirror the decision logic used in the reproduction package so that
benchmarking code can reason about:

  - baseline_upstream:      upstream FA3 heuristic with the premature guard
  - upstream_two_guard:     the real upstreamable two-guard patch
  - latest_stack_tuned_s3:  same guards plus an explicit low-tile s=3 choice
  - no_shortcut:            baseline with the guard entirely removed
  - relaxed:                guard relaxed to num_n_blocks <= 2
"""

import math


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# =============================================================================
#  Policy A: Baseline (upstream FA3 heuristic)
# =============================================================================
def baseline_num_splits(
    *,
    b: int,
    hkv: int,
    lq: int,
    lk: int,
    d: int,
    num_sms: int,
    is_causal_or_local: bool = True,
    max_splits: int = 128,
    block_m: int = 128,
    block_n: int = 128,
) -> int:
    """
    Reference implementation of the baseline heuristic from FA3.
    This mirrors num_splits_heuristic() in heuristics.h for decode-like paths.
    """
    num_n_blocks = _ceil_div(lk, block_n)
    num_m_blocks = _ceil_div(lq, block_m)
    total_mblocks = b * hkv * num_m_blocks

    # Saturation check
    if total_mblocks >= 0.8 * num_sms:
        size_l2 = 50 * 1024 * 1024
        size_one_kv_head = lk * (d + d) * 2
        if (size_one_kv_head > size_l2
                and num_m_blocks >= num_sms * 2
                and not is_causal_or_local):
            return min(_ceil_div(size_one_kv_head, size_l2), max_splits)
        return 1

    # >>> THE PREMATURE GUARD (root cause of the bug) <<<
    if num_n_blocks <= 4:
        return 1

    # SM-coverage efficiency loop (never reached for L_K <= 512)
    max_splits = min(max_splits, num_sms, num_n_blocks)
    max_eff = 0.0
    effs = []
    for s in range(1, max_splits + 1):
        n_waves = (total_mblocks * s) / num_sms
        eff = n_waves / math.ceil(n_waves)
        if eff > max_eff:
            max_eff = eff
        effs.append(eff)
    for s in range(1, max_splits + 1):
        if effs[s - 1] >= 0.85 * max_eff:
            return s
    return 1


# =============================================================================
#  Upstream Patch Track: two-guard fall-through patch
# =============================================================================
def upstream_two_guard_num_splits(
    *,
    b: int,
    hkv: int,
    lq: int,
    lk: int,
    d: int,
    num_sms: int,
    is_causal_or_local: bool = True,
    max_splits: int = 128,
    block_m: int = 128,
    block_n: int = 128,
    min_tiles_for_shortcut: int = 4,
) -> int:
    """
    The upstreamable two-guard patch:
      1. if (num_n_blocks <= 3) return 1;          // Guard 1
      2. if (num_n_blocks <= 4 && tiles >= 4) return 1;  // Guard 2
      3. if (num_n_blocks == 4 && tiles < 4) return 3;   // Low-tile boundary override
      4. Fall through to existing efficiency loop.
    """
    num_n_blocks = _ceil_div(lk, block_n)
    num_m_blocks = _ceil_div(lq, block_m)
    total_mblocks = b * hkv * num_m_blocks

    # Saturation check (unchanged from baseline)
    if total_mblocks >= 0.8 * num_sms:
        size_l2 = 50 * 1024 * 1024
        size_one_kv_head = lk * (d + d) * 2
        if (size_one_kv_head > size_l2
                and num_m_blocks >= num_sms * 2
                and not is_causal_or_local):
            return min(_ceil_div(size_one_kv_head, size_l2), max_splits)
        return 1

    # Guard 1: L_K <= 384 (nblk <= 3)
    if num_n_blocks <= 3:
        return 1

    # Guard 2: L_K = 448-512 (nblk = 4) with enough tiles
    if num_n_blocks <= 4 and total_mblocks >= min_tiles_for_shortcut:
        return 1

    # Low-tile boundary case: explicit override for L_K=512 when tiles < 4
    if num_n_blocks == 4 and total_mblocks < min_tiles_for_shortcut:
        return 3

    # Existing efficiency loop (unchanged)
    max_splits = min(max_splits, num_sms, num_n_blocks)
    max_eff = 0.0
    effs = []
    for s in range(1, max_splits + 1):
        n_waves = (total_mblocks * s) / num_sms
        eff = n_waves / math.ceil(n_waves)
        if eff > max_eff:
            max_eff = eff
        effs.append(eff)
    for s in range(1, max_splits + 1):
        if effs[s - 1] >= 0.85 * max_eff:
            return s
    return 1


# =============================================================================
#  Latest-stack tuned policy
# =============================================================================
def latest_stack_tuned_num_splits(
    *,
    b: int,
    hkv: int,
    lq: int,
    lk: int,
    d: int,
    num_sms: int,
    is_causal_or_local: bool = True,
    max_splits: int = 128,
    block_m: int = 128,
    block_n: int = 128,
    min_tiles_for_shortcut: int = 4,
    tuned_low_tile_splits: int = 3,
) -> int:
    """
    Tuned variant for the latest software stack.

    The safety logic remains the same as the upstream two-guard patch:
      - Guard 1 keeps L_K <= 384 at 1 split.
      - Guard 2 keeps nblk=4 with enough tiles at 1 split.

    For the low-tile nblk=4 win regime, the latest stack benchmarks show that
    3 splits outperforms the original 4-way choice for H_KV in {1, 2}.
    """
    num_n_blocks = _ceil_div(lk, block_n)
    num_m_blocks = _ceil_div(lq, block_m)
    total_mblocks = b * hkv * num_m_blocks

    if total_mblocks >= 0.8 * num_sms:
        size_l2 = 50 * 1024 * 1024
        size_one_kv_head = lk * (d + d) * 2
        if (size_one_kv_head > size_l2
                and num_m_blocks >= num_sms * 2
                and not is_causal_or_local):
            return min(_ceil_div(size_one_kv_head, size_l2), max_splits)
        return 1

    if num_n_blocks <= 3:
        return 1

    if num_n_blocks <= 4 and total_mblocks >= min_tiles_for_shortcut:
        return 1

    if num_n_blocks == 4 and total_mblocks < min_tiles_for_shortcut:
        return min(tuned_low_tile_splits, max_splits, num_sms, num_n_blocks)

    max_splits = min(max_splits, num_sms, num_n_blocks)
    max_eff = 0.0
    effs = []
    for s in range(1, max_splits + 1):
        n_waves = (total_mblocks * s) / num_sms
        eff = n_waves / math.ceil(n_waves)
        if eff > max_eff:
            max_eff = eff
        effs.append(eff)
    for s in range(1, max_splits + 1):
        if effs[s - 1] >= 0.85 * max_eff:
            return s
    return 1


# =============================================================================
#  Ablation: No Shortcut (guard entirely removed)
# =============================================================================
def no_shortcut_num_splits(
    *,
    b: int,
    hkv: int,
    lq: int,
    lk: int,
    d: int,
    num_sms: int,
    is_causal_or_local: bool = True,
    max_splits: int = 128,
    block_m: int = 128,
    block_n: int = 128,
) -> int:
    """
    Baseline with the num_n_blocks<=4 shortcut entirely removed.
    Used in guard ablation to expose the cost of removing the shortcut entirely.
    """
    num_n_blocks = _ceil_div(lk, block_n)
    num_m_blocks = _ceil_div(lq, block_m)
    total_mblocks = b * hkv * num_m_blocks

    if total_mblocks >= 0.8 * num_sms:
        size_l2 = 50 * 1024 * 1024
        size_one_kv_head = lk * (d + d) * 2
        if (size_one_kv_head > size_l2
                and num_m_blocks >= num_sms * 2
                and not is_causal_or_local):
            return min(_ceil_div(size_one_kv_head, size_l2), max_splits)
        return 1

    # GUARD REMOVED — efficiency loop always runs
    max_splits = min(max_splits, num_sms, num_n_blocks)
    max_eff = 0.0
    effs = []
    for s in range(1, max_splits + 1):
        n_waves = (total_mblocks * s) / num_sms
        eff = n_waves / math.ceil(n_waves)
        if eff > max_eff:
            max_eff = eff
        effs.append(eff)
    for s in range(1, max_splits + 1):
        if effs[s - 1] >= 0.85 * max_eff:
            return s
    return 1


# =============================================================================
#  Ablation: Relaxed (guard threshold lowered to nBlk <= 2)
# =============================================================================
def relaxed_num_splits(
    *,
    b: int,
    hkv: int,
    lq: int,
    lk: int,
    d: int,
    num_sms: int,
    is_causal_or_local: bool = True,
    max_splits: int = 128,
    block_m: int = 128,
    block_n: int = 128,
) -> int:
    """
    Relaxed fix: guard lowered to num_n_blocks <= 2 (L_K <= 256).
    Used in guard ablation to show that lowering the threshold misses the
    low-tile L_K=512 win region.
    """
    num_n_blocks = _ceil_div(lk, block_n)
    num_m_blocks = _ceil_div(lq, block_m)
    total_mblocks = b * hkv * num_m_blocks

    if total_mblocks >= 0.8 * num_sms:
        size_l2 = 50 * 1024 * 1024
        size_one_kv_head = lk * (d + d) * 2
        if (size_one_kv_head > size_l2
                and num_m_blocks >= num_sms * 2
                and not is_causal_or_local):
            return min(_ceil_div(size_one_kv_head, size_l2), max_splits)
        return 1

    if num_n_blocks <= 2:  # relaxed from <= 4 to <= 2
        return 1

    max_splits = min(max_splits, num_sms, num_n_blocks)
    max_eff = 0.0
    effs = []
    for s in range(1, max_splits + 1):
        n_waves = (total_mblocks * s) / num_sms
        eff = n_waves / math.ceil(n_waves)
        if eff > max_eff:
            max_eff = eff
        effs.append(eff)
    for s in range(1, max_splits + 1):
        if effs[s - 1] >= 0.85 * max_eff:
            return s
    return 1


def candidate_policy_name(track: str) -> str:
    if track == "upstream_patch":
        return "upstream_two_guard"
    if track == "latest_stack_tuned":
        return "latest_stack_tuned_s3"
    raise ValueError(f"Unsupported track '{track}'")


def candidate_num_splits_for_track(track: str, **kwargs) -> int:
    if track == "upstream_patch":
        return upstream_two_guard_num_splits(**kwargs)
    if track == "latest_stack_tuned":
        return latest_stack_tuned_num_splits(**kwargs)
    raise ValueError(f"Unsupported track '{track}'")


# Backward-compatible aliases used by older docs / scripts.
tile_aware_num_splits = upstream_two_guard_num_splits
