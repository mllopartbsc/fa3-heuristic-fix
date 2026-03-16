#!/usr/bin/env python3
"""
Apply Hopper-only build fix to flash-attention hopper/setup.py.

When FLASH_ATTENTION_DISABLE_SM80=TRUE, shared CUDA files (flash_fwd_combine.cu,
flash_prepare_scheduler.cu) should use sm_90a only, not sm_80+sm_90a.
Without this patch, they use cuda_compile_sm80_sm90 which builds for both arches.

Usage: python3 apply_hopper_only_setup.py <path-to-hopper-setup.py>
"""
import sys
from pathlib import Path


def apply(hopper_setup_path: Path) -> bool:
    path = Path(hopper_setup_path)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return False

    text = path.read_text(encoding="utf-8")
    old = """            elif source_file.endswith('_sm100.cu'):
                rule = 'cuda_compile_sm100'
            else:
                rule = 'cuda_compile_sm80_sm90'"""
    new = """            elif source_file.endswith('_sm100.cu'):
                rule = 'cuda_compile_sm100'
            else:
                # Hopper-only: use sm_90a only for shared files when DISABLE_SM8x
                rule = 'cuda_compile' if DISABLE_SM8x else 'cuda_compile_sm80_sm90'"""

    if new in text:
        print(f"  Hopper-only setup patch already applied: {path}")
        return True
    if old not in text:
        print(f"ERROR: Could not find target block in {path}", file=sys.stderr)
        return False

    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  Applied Hopper-only setup patch: {path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 apply_hopper_only_setup.py <path-to-hopper/setup.py>", file=sys.stderr)
        sys.exit(1)
    ok = apply(Path(sys.argv[1]))
    sys.exit(0 if ok else 1)
