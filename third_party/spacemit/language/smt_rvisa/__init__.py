# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""smt_rvisa — Python eDSL for writing raw Linalg/memref/vector MLIR kernels.

Public API:
    smt_rvisa   : decorator factory to mark a function as a raw MLIR kernel
    raw_kernel  : convenience alias for smt_rvisa(name="linalg")
    mem, index  : parameter annotations (sr.mem(f16) / sr.mem(f32, out=True)
                  / sr.index)
    call        : inside @triton.jit, emit tle.dsl_region (C++ DSLRegionOpPattern
                  lowers it to spine_ext.raw_region)
    proton_mark : profiling marker (rdtime + proton_record)
    vload, vstore, vmacc, vreduce_sum, ... : vector-level built-in operators
"""

from .types import mem, index
from .runtime import smt_rvisa, SmtLinalgJITFunction
from .call_registry import call
from .builtins import proton_mark
from .builtins import vconfig, vzero, vload, vmacc, vreduce_sum, vreduce_max, vreduce_min, vreduce_mul, vstore, alloc, pack, vpack, vmadot, vshape, vbroadcast, spread
from .builtins import vmin, vmax, sqrt, rsqrt, vexp, vlog, sload, sstore, viota, abs, cast, select  # §6.4 elementwise + transcendental
from .builtins import call_intrinsic, llvm_poison, llvm_const, llvm_base_ptr, llvm_gep, llvm_size  # LLVM-dialect llvm-direct
from .builtins import f16, f32, bf16
from .builtins import mma_cube
from .builtins import range as range  # noqa: A001 (shadows builtin intentionally)

# raw_kernel: bare decorator alias for @smt_rvisa(name="linalg") to match the
# feishu 3.3 surface (`@sr.raw_kernel`).
raw_kernel = smt_rvisa(name="linalg")

__all__ = [
    "smt_rvisa",
    "raw_kernel",
    "SmtLinalgJITFunction",
    "mem",
    "index",
    "call",
    "proton_mark",
    "vconfig",
    "vzero",
    "vload",
    "vmacc",
    "vreduce_sum",
    "vreduce_max",
    "vreduce_min",
    "vreduce_mul",
    "vstore",
    "alloc",
    "pack",
    "vpack",
    "vmadot",
    "vshape",
    "vbroadcast",
    "spread",
    "vmin",
    "vmax",
    "sqrt",
    "rsqrt",
    "vexp",
    "vlog",
    "sload",
    "sstore",
    "viota",
    "abs",
    "cast",
    "select",
    "call_intrinsic",
    "llvm_poison",
    "llvm_const",
    "llvm_base_ptr",
    "llvm_gep",
    "llvm_size",
    "f16",
    "f32",
    "bf16",
    "mma_cube",
    "range",
]
