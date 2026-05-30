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
def test_subview_kernel1(XBLOCK: tl.constexpr):
    # 1. Allocate a local buffer
    src_buffer = bl.alloc(tl.float32, [XBLOCK, XBLOCK])

    result_buffer = bl.subview(src_buffer, offsets=[1, 1], sizes=[XBLOCK - 2, XBLOCK - 2], strides=[1, 1])


@triton.jit
def test_subview_kernel2(XBLOCK: tl.constexpr, offsets: tl.constexpr, sizes: tl.constexpr, strides: tl.constexpr):
    # this statement has no effect, just to test the builder
    bl.alloc(tl.float32, [XBLOCK]).subview([offsets], [sizes], [strides])


# ============== Main for manual testing ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: test_subview_function")
    print("=" * 60)
    mlir = compile_kernel(test_subview_kernel1, {}, {"XBLOCK": 8})
    print(f"Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)

    print("\n" + "=" * 60)
    print("Test 2: test_subview_constructor")
    print("=" * 60)
    mlir = compile_kernel(test_subview_kernel2, {}, {"XBLOCK": 32, "offsets": 1, "sizes": 30, "strides": 1})
    print(f"Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)
