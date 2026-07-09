"""Pre-run dependency checks for the RPU compile tests.

The compile tests drive the real RPU toolchain (clang + assembler). Rather
than fail with a cryptic assertion when the toolchain is not configured, these
helpers validate the required environment up front and fail with a clear,
actionable message naming the env var that is missing or the path that does
not exist.

Required environment for the compile tests:

  RPU_LLVM_ROOT   install prefix of the RPU LLVM toolchain — must contain
                  ``bin/clang`` (release layout) or ``build/bin/clang``
                  (source-build layout). The assembler ships with the same
                  toolchain as ``bin/rpuasm``.
  RPU_ASM_PATH    optional — overrides the assembler path when it does not
                  sit at ``<RPU_LLVM_ROOT>/bin/rpuasm``.

The on-board launch_kernel test additionally needs RPU_LK_RUNNER; it performs
its own check (see test/board/lk_board_smoke.py).
"""

import os
from pathlib import Path

import pytest


def require_rpu_toolchain():
    """Validate the RPU compile toolchain; return the RPU_LLVM_ROOT Path.

    Fails the calling test with a clear message (no traceback) when a
    dependency is unset or its path is missing.
    """
    root = os.getenv("RPU_LLVM_ROOT")
    if not root:
        pytest.fail(
            "RPU_LLVM_ROOT is not set. Export the RPU LLVM install prefix "
            "(the directory containing bin/clang):\n"
            "  export RPU_LLVM_ROOT=/path/to/hxcc-llvm",
            pytrace=False,
        )
    llvm_root = Path(root)
    if not ((llvm_root / "bin" / "clang").exists() or (llvm_root / "build" / "bin" / "clang").exists()):
        pytest.fail(
            f"RPU clang not found under RPU_LLVM_ROOT={root} "
            "(looked for bin/clang and build/bin/clang).",
            pytrace=False,
        )

    asm = os.getenv("RPU_ASM_PATH")
    if asm:
        asm_path = Path(asm)
    else:
        # The assembler ships inside the toolchain as <root>/bin/rpuasm
        # (or build/bin/rpuasm for a source build).
        asm_path = llvm_root / "bin" / "rpuasm"
        if not asm_path.exists() and (llvm_root / "build" / "bin" / "rpuasm").exists():
            asm_path = llvm_root / "build" / "bin" / "rpuasm"
    if not asm_path.exists():
        pytest.fail(
            f"RPU assembler not found at {asm_path}. It ships with the LLVM "
            "toolchain as bin/rpuasm; set RPU_ASM_PATH to override its path.",
            pytrace=False,
        )

    return llvm_root


def make_ast_source(fn, signature, constants=None):
    """Build an ``ASTSource`` from positional (integer-indexed) dicts.

    These RPU compile tests describe a kernel's signature and constexpr
    specializations positionally (``{0: "*fp16", ...}`` / ``{3: n}``). The
    modern ``ASTSource`` API takes a string-keyed ``signature`` and a
    ``constexprs`` dict keyed by argument name or single-element index tuple,
    so this helper performs that conversion in one place for the test suite.
    """
    from triton.compiler import ASTSource

    sig = {(fn.arg_names[k] if isinstance(k, int) else k): v for k, v in signature.items()}
    constexprs = None
    if constants is not None:
        constexprs = {((k, ) if isinstance(k, int) else k): v for k, v in constants.items()}
    return ASTSource(fn=fn, signature=sig, constexprs=constexprs)
