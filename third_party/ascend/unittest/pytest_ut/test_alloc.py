# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os

import pytest
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton._C.libtriton import ir, buffer_ir
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
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {"create_address_space": al.semantic.create_address_space},
                         {})
    return str(module)


# ============== Kernel definitions ==============


@triton.jit
def allocate_local_buffer(XBLOCK: tl.constexpr):
    # this statement has no effect, just to test the builder
    bl.alloc(tl.float32, [XBLOCK])
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L1)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0A)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0B)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0C)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB, is_mem_unique=True)


@triton.jit
def allocate_to_smem_buffer(x_ptr, XBLOCK: tl.constexpr):
    x_l1_keep = bl.alloc(tl.float32, [XBLOCK], al.ascend_address_space.L1)
    for i in range(XBLOCK):
        offsets = tl.arange(0, XBLOCK)
        x = tl.load(x_ptr + offsets, mask=offsets < XBLOCK)
        bl.to_buffer(tensor=x, bind_buffer=x_l1_keep)
        tl.store(x_ptr + offsets, x, mask=offsets < XBLOCK)


def test_allocate_local_buffer():
    """Test allocating local buffers in different address spaces."""
    mlir = compile_kernel(allocate_to_smem_buffer, {"x_ptr": "*fp32"}, {"XBLOCK": 256})
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


# ============== Main for manual testing ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Nested Scopes")
    print("=" * 60)
    mlir = compile_kernel(allocate_local_buffer, {}, {"XBLOCK": 256})
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)
