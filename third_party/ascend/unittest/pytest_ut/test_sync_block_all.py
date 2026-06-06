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
def test_sync_block_all():
    al.sync_block_all("all_cube", 8)
    al.sync_block_all("all_vector", 9)
    al.sync_block_all("all", 10)
    al.sync_block_all("all_sub_vector", 11)


# ============== Main for manual testing ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: test_sync_block_all")
    print("=" * 60)
    mlir = compile_kernel(test_sync_block_all, {}, {})
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)
