#!/usr/bin/env python3
"""
Add scheduler_metadata support to hopper flash_attn_varlen_func for precomputed metadata path.

vLLM v1 passes scheduler_metadata; the hopper interface doesn't expose it by default.
This patch adds it so the heuristic fix delivers full ~20% kernel gain with metadata.
"""
import sys
from pathlib import Path


def apply(interface_path: Path) -> bool:
    path = Path(interface_path)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return False

    text = path.read_text(encoding="utf-8")

    # 1. Add scheduler_metadata to flash_attn_varlen_func signature
    old_sig = """def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):"""
    new_sig = """def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
    scheduler_metadata=None,
):"""

    if new_sig in text:
        print(f"  Hopper scheduler_metadata patch already applied: {path}")
        return True
    if old_sig not in text:
        print(f"ERROR: Could not find flash_attn_varlen_func signature in {path}", file=sys.stderr)
        return False

    text = text.replace(old_sig, new_sig, 1)

    # 2. Add scheduler_metadata to FlashAttnVarlenFunc.forward and pass to _flash_attn_forward
    old_forward = """        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_varlen_forward(
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,   # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )"""
    new_forward = """        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
        scheduler_metadata=None,
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_varlen_forward(
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,   # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )"""

    if new_forward in text:
        pass  # Already applied
    elif old_forward in text:
        text = text.replace(old_forward, new_forward, 1)
    else:
        print(f"ERROR: Could not find FlashAttnVarlenFunc.forward block in {path}", file=sys.stderr)
        return False

    # 3. Add scheduler_metadata to public flash_attn_varlen_func and pass to FlashAttnVarlenFunc.apply
    old_apply = """    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
    )"""
    new_apply = """    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
        scheduler_metadata,
    )"""

    if new_apply in text:
        pass
    elif old_apply in text:
        text = text.replace(old_apply, new_apply, 1)
    else:
        print(f"ERROR: Could not find FlashAttnVarlenFunc.apply call in {path}", file=sys.stderr)
        return False

    path.write_text(text, encoding="utf-8")
    print(f"  Applied Hopper scheduler_metadata patch: {path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 apply_hopper_scheduler_metadata.py <path-to-hopper/flash_attn_interface.py>", file=sys.stderr)
        sys.exit(1)
    ok = apply(Path(sys.argv[1]))
    sys.exit(0 if ok else 1)
