# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FlagMega: an editable, staged model compiler for FlagTree."""

from triton.flagmega._version import __version__
from triton.flagmega.artifacts import load_artifact, write_artifact
from triton.flagmega.compiler import CompileResult, Compiler
from triton.flagmega.diagnostics import DumpFlags
from triton.flagmega.ir import IRBuilder, IRModule, emit_module, load_module, verify_module
from triton.flagmega.options import CompileOptions, RunOptions
from triton.flagmega.runtime import load

__all__ = [
    "CompileOptions",
    "CompileResult",
    "Compiler",
    "DumpFlags",
    "IRBuilder",
    "IRModule",
    "RunOptions",
    "__version__",
    "emit_module",
    "load",
    "load_artifact",
    "load_module",
    "verify_module",
    "write_artifact",
]
