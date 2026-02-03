#!/usr/bin/env python3
import os

import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel to MLIR."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {}, {})
    return str(module)


# ============== Kernel definitions ==============


@triton.jit
def kernel_debug_barrier(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test debug barrier."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


# ============== Pytest tests ==============


def test_debug_barrier():
    """Test debug barrier generates."""
    mlir = compile_kernel(
        kernel_debug_barrier,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert "annotation.mark" in mlir and "VV_ALL" in mlir


# ============== Main for manual testing ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Test: debug barrier")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_debug_barrier,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)
