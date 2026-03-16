#!/usr/bin/env python3
"""
Relax batch_size == batch_size_k check for MQA/GQA with paged KV.

vLLM with MQA can pass tensors where batch_size != batch_size_k due to
different Q vs K/V batch layouts. This patch allows the mismatch when using
paged KV with MQA/GQA (num_heads > num_heads_k).
"""
import sys
from pathlib import Path


def apply_one(path: Path, check_macro: str) -> bool:
    """Apply fix to a single file. check_macro is TORCH_CHECK or STD_TORCH_CHECK."""
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return False

    text = path.read_text(encoding="utf-8")

    old_block = f"""    if (!kv_batch_idx_.has_value()) {{
        {check_macro}(batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
    }}"""

    new_block = f"""    if (!kv_batch_idx_.has_value()) {{
        bool const is_mqa_gqa = (num_heads > num_heads_k);
        bool const allow_mismatch = paged_KV && is_mqa_gqa;
        {check_macro}(allow_mismatch || batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
    }}"""

    if new_block in text:
        print(f"  Batch-size MQA fix already applied: {path}")
        return True
    if old_block not in text:
        return False

    text = text.replace(old_block, new_block, 1)
    path.write_text(text, encoding="utf-8")
    print(f"  Applied batch-size MQA fix: {path}")
    return True


def apply(hopper_path: Path) -> bool:
    """Apply fix to flash_api.cpp and flash_api_stable.cpp in hopper dir."""
    hopper = Path(hopper_path)
    if hopper.is_file():
        # Single file: detect macro from content
        check = "STD_TORCH_CHECK" if "STD_TORCH_CHECK" in hopper.read_text() else "TORCH_CHECK"
        return apply_one(hopper, check)

    api_cpp = hopper / "flash_api.cpp"
    api_stable_cpp = hopper / "flash_api_stable.cpp"
    ok_api = apply_one(api_cpp, "TORCH_CHECK") if api_cpp.exists() else True
    ok_stable = apply_one(api_stable_cpp, "STD_TORCH_CHECK") if api_stable_cpp.exists() else True
    return ok_api and ok_stable


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 apply_batch_size_mqa_fix.py <path-to-hopper>", file=sys.stderr)
        sys.exit(1)
    ok = apply(Path(sys.argv[1]))
    sys.exit(0 if ok else 1)
