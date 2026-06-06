#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Demo: declare scratch/extra buffers on a custom op via `extra_buffers` (dtype, size)
# and read back the sizes from lowered HIVM MLIR (`extra_buffers_sizes` attribute).

from __future__ import annotations

import os
import re
import subprocess

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, ttir_to_linalg

# Scratch buffers requested by the custom kernel (element type + length in elements).
SCRATCH_SPEC = [
    (tl.float32, 1024),
    (tl.bfloat16, 512),
    (tl.int32, 256),
]


@al.register_custom_op
class demo_extra_buffer_op:
    """Custom op that advertises extra device buffers for the NPU compiler / runtime."""

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "demo_extra_buffer_op_impl"
    bitcode = os.path.abspath(__file__)

    def __init__(self, x, out=None):
        self.indexing_map = [al.affine_map.get_identity(1)]
        self.extra_buffers = list(SCRATCH_SPEC)


@triton.jit
def kernel_extra_buf(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(out_ptr + i, mask=i < n)
    r = al.custom("demo_extra_buffer_op", x, out=y)
    tl.store(out_ptr + i, r, mask=i < n)


def compile_to_linalg_mlir(kernel, signature: dict, constants: dict) -> str | None:
    src = ASTSource(kernel, signature, constants)
    ctx = ir.context()
    ir.load_dialects(ctx)
    ascend_ir.load_dialects(ctx)
    options = NPUOptions()
    try:
        ttir = ast_to_ttir(kernel, src, ctx, options, {}, {})
        meta = {**options.__dict__}
        return str(ttir_to_linalg(ttir, meta, options, named_ops=True))
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        return None


def extract_extra_buffer_sizes_from_mlir(mlir: str) -> list[int]:
    """
    Parse `extra_buffers_sizes` from HIVM custom op text.
    """
    # Parse [1024, 512, 256, ...]
    m = re.search(r"extra_buffers_sizes\s*=\s*\[([^\]]+)\]", mlir)
    if m:
        raw = m.group(1).replace(" ", "")
        return [int(x) for x in raw.split(",") if x]

    return []


def main() -> None:
    expected_sizes = [size for _, size in SCRATCH_SPEC]
    print("Declared extra_buffers (dtype, element_count):")
    for dt, sz in SCRATCH_SPEC:
        print(f"  {dt} -> {sz} elements")

    mlir = compile_to_linalg_mlir(
        kernel_extra_buf,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 128},
    )
    if not mlir:
        print("Compilation failed.")
        return

    parsed = extract_extra_buffer_sizes_from_mlir(mlir)
    print("\nParsed extra_buffers_sizes from MLIR:", parsed)
    if parsed == expected_sizes:
        print("OK: MLIR sizes match the Python extra_buffers specification.")
    elif parsed:
        print("Note: parsed sizes differ from spec; inspect MLIR spelling below.")
    else:
        print("Could not parse extra_buffers_sizes automatically; "
              "search the dump for 'extra_buffers_sizes'.")

    print("\n--- MLIR excerpt (lines containing hivm.hir.custom) ---")
    for line in mlir.splitlines():
        if "hivm.hir.custom" in line and "demo_extra_buffer_op" in line:
            print(line)


if __name__ == "__main__":
    main()
