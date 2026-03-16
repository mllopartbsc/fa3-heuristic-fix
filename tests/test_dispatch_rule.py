"""
Regression tests for the dispatch rule (heuristics policy).

Verifies that baseline_num_splits, upstream_two_guard_num_splits, and
latest_stack_tuned_num_splits return the expected values for specific
(b, hkv, lq, lk, d, num_sms) inputs.
"""

import pytest

from src.heuristics_reference import (
    baseline_num_splits,
    upstream_two_guard_num_splits,
    latest_stack_tuned_num_splits,
    no_shortcut_num_splits,
    relaxed_num_splits,
    candidate_num_splits_for_track,
)


# H100-like SM count
NUM_SMS = 132


def test_baseline_premature_guard_lk512_hkv1():
    """Baseline returns 1 for L_K=512, H_KV=1 (premature guard)."""
    s = baseline_num_splits(b=1, hkv=1, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s == 1


def test_baseline_premature_guard_lk512_hkv2():
    """Baseline returns 1 for L_K=512, H_KV=2 (premature guard)."""
    s = baseline_num_splits(b=1, hkv=2, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s == 1


def test_baseline_safe_lk384():
    """Baseline returns 1 for L_K=384 (nblk=3, guard triggers)."""
    s = baseline_num_splits(b=1, hkv=1, lq=1, lk=384, d=128, num_sms=NUM_SMS)
    assert s == 1


def test_upstream_two_guard_win_regime_lk512_hkv1():
    """Upstream patch returns 3 for L_K=512, H_KV=1 (explicit low-tile override)."""
    s = upstream_two_guard_num_splits(b=1, hkv=1, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s == 3


def test_upstream_two_guard_win_regime_lk512_hkv2():
    """Upstream patch returns 3 for L_K=512, H_KV=2 (explicit low-tile override)."""
    s = upstream_two_guard_num_splits(b=1, hkv=2, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s == 3


def test_upstream_two_guard_safe_lk384():
    """Upstream patch returns 1 for L_K=384 (Guard 1)."""
    s = upstream_two_guard_num_splits(b=1, hkv=1, lq=1, lk=384, d=128, num_sms=NUM_SMS)
    assert s == 1


def test_upstream_two_guard_safe_lk512_hkv8():
    """Upstream patch returns 1 for L_K=512, H_KV=8 (Guard 2: tiles=8 >= 4)."""
    s = upstream_two_guard_num_splits(b=1, hkv=8, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s == 1


def test_latest_stack_tuned_explicit_s3():
    """Latest-stack tuned returns 3 for low-tile nblk=4 regime."""
    s = latest_stack_tuned_num_splits(b=1, hkv=1, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s == 3


def test_latest_stack_tuned_explicit_s3_hkv2():
    """Latest-stack tuned returns 3 for H_KV=2, L_K=512."""
    s = latest_stack_tuned_num_splits(b=1, hkv=2, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s == 3


def test_latest_stack_tuned_safe_lk384():
    """Latest-stack tuned returns 1 for L_K=384 (Guard 1)."""
    s = latest_stack_tuned_num_splits(b=1, hkv=1, lq=1, lk=384, d=128, num_sms=NUM_SMS)
    assert s == 1


def test_latest_stack_tuned_safe_lk512_hkv8():
    """Latest-stack tuned returns 1 for L_K=512, H_KV=8 (Guard 2)."""
    s = latest_stack_tuned_num_splits(b=1, hkv=8, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s == 1


def test_no_shortcut_efficiency_loop():
    """No-shortcut policy lets efficiency loop run for L_K=512, H_KV=1."""
    s = no_shortcut_num_splits(b=1, hkv=1, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s >= 2
    assert s <= 4


def test_relaxed_misses_win_regime():
    """Relaxed (nblk<=2) returns 1 for L_K=256; for L_K=512 loop runs."""
    # Relaxed: nblk<=2 returns 1. For L_K=256, nblk=2, guard triggers.
    s_256 = relaxed_num_splits(b=1, hkv=1, lq=1, lk=256, d=128, num_sms=NUM_SMS)
    assert s_256 == 1
    # For L_K=512, nblk=4, relaxed guard does NOT trigger; loop runs.
    s_512 = relaxed_num_splits(b=1, hkv=1, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s_512 >= 2


def test_candidate_for_track_upstream():
    """candidate_num_splits_for_track returns upstream policy for upstream_patch."""
    s = candidate_num_splits_for_track(
        "upstream_patch", b=1, hkv=1, lq=1, lk=512, d=128, num_sms=NUM_SMS
    )
    assert s == upstream_two_guard_num_splits(
        b=1, hkv=1, lq=1, lk=512, d=128, num_sms=NUM_SMS
    )


def test_candidate_for_track_latest_stack():
    """candidate_num_splits_for_track returns 3 for latest_stack_tuned."""
    s = candidate_num_splits_for_track(
        "latest_stack_tuned", b=1, hkv=1, lq=1, lk=512, d=128, num_sms=NUM_SMS
    )
    assert s == 3


def test_baseline_vs_upstream_differ_at_boundary():
    """Baseline and upstream differ at L_K=512, H_KV=1."""
    s_base = baseline_num_splits(b=1, hkv=1, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    s_up = upstream_two_guard_num_splits(b=1, hkv=1, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s_base == 1
    assert s_up != 1


def test_nblk_4_tiles_3():
    """nblk=4, tiles=3: upstream patch explicit return 3 (tiles < 4)."""
    s = upstream_two_guard_num_splits(b=1, hkv=3, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s == 3


def test_nblk_4_tiles_4():
    """nblk=4, tiles=4: Guard 2 triggers, return 1."""
    s = upstream_two_guard_num_splits(b=1, hkv=4, lq=1, lk=512, d=128, num_sms=NUM_SMS)
    assert s == 1
