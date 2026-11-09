# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
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
    arch = "Ascend910_95"


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


# Test cases for core.py - buffer_type class
@triton.jit
def kernel_buffer_type_operations(M: tl.constexpr, N: tl.constexpr):
    # Test buffer allocation (which creates buffer_type instances)
    buf_ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    buf_l1 = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L1)


# Test cases for core.py - buffer class
@triton.jit
def kernel_buffer_class_operations(M: tl.constexpr, N: tl.constexpr):
    # Allocate a buffer
    buf = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)

    # Test __str__ method implicitly
    # Test subview method
    sub = buf.subview([0, 0], [M // 2, N // 2], [1, 1])

    # Test to_tensor method
    tensor = buf.to_tensor(writable=True)


# Test cases for core.py - alloc function
@triton.jit
def kernel_alloc_operations(M: tl.constexpr, N: tl.constexpr):
    # Test alloc with different address spaces
    buf_ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    buf_l1 = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L1)
    buf_l0a = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L0A)
    buf_l0b = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L0B)
    buf_l0c = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L0C)

    # Test alloc with mem_unique flag
    buf_unique = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB, is_mem_unique=True)

    # Test alloc with different data types
    buf_float16 = bl.alloc(tl.float16, [M, N], al.ascend_address_space.UB)
    buf_int32 = bl.alloc(tl.int32, [M, N], al.ascend_address_space.UB)
    buf_int16 = bl.alloc(tl.int16, [M, N], al.ascend_address_space.UB)


# Test cases for core.py - to_buffer function
@triton.jit
def kernel_to_buffer_operations(M: tl.constexpr, N: tl.constexpr):
    # Create tensors of different data types
    x_float32 = tl.full([M, N], 0.0, dtype=tl.float32)
    x_float16 = tl.full([M, N], 0.0, dtype=tl.float16)
    x_int32 = tl.full([M, N], 0, dtype=tl.int32)

    # Test to_buffer with different address spaces
    buf_ub = bl.to_buffer(x_float32, al.ascend_address_space.UB)
    buf_l1 = bl.to_buffer(x_float32, al.ascend_address_space.L1)

    # Test to_buffer with bind_buffer
    buf_bind = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    buf_bound = bl.to_buffer(x_float32, al.ascend_address_space.UB, bind_buffer=buf_bind)


# Test cases for core.py - to_tensor function
@triton.jit
def kernel_to_tensor_operations(M: tl.constexpr, N: tl.constexpr):
    # Allocate buffers of different data types
    buf_float32 = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    buf_float16 = bl.alloc(tl.float16, [M, N], al.ascend_address_space.UB)

    # Test to_tensor with default parameters
    tensor_float32 = bl.to_tensor(buf_float32, writable=True)

    # Test to_tensor with writable=False
    tensor_readonly = bl.to_tensor(buf_float32, writable=False)

    # Test to_tensor with target shape (must be different from source shape)
    tensor_reshaped = bl.to_tensor(buf_float32, writable=True, target_shape=[M * 2, N * 2])


# Test cases for core.py - subview function
@triton.jit
def kernel_subview_operations(M: tl.constexpr, N: tl.constexpr):
    # Allocate a buffer
    buf = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)

    # Test subview function with different parameters
    sub1 = bl.subview(buf, [0, 0], [M // 2, N // 2], [1, 1])
    sub2 = bl.subview(buf, [M // 2, 0], [M // 2, N], [1, 1])
    sub3 = bl.subview(buf, [0, N // 2], [M, N // 2], [1, 1])

    # Test subview with different offset types
    # Integer offsets
    sub_int = bl.subview(buf, [0, 0], [M // 2, N // 2], [1, 1])


# Test function definitions
def test_buffer_type_operations():
    print("=" * 60)
    print("Test 1: Buffer Type Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_buffer_type_operations,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_buffer_class_operations():
    print("=" * 60)
    print("Test 2: Buffer Class Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_buffer_class_operations,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_alloc_operations():
    print("=" * 60)
    print("Test 3: Alloc Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_alloc_operations,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_to_buffer_operations():
    print("=" * 60)
    print("Test 4: to_buffer Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_to_buffer_operations,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_to_tensor_operations():
    print("=" * 60)
    print("Test 5: to_tensor Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_to_tensor_operations,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_subview_operations():
    print("=" * 60)
    print("Test 6: Subview Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_subview_operations,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


# ============== Main for manual testing ==============
if __name__ == "__main__":
    test_buffer_type_operations()
    test_buffer_class_operations()
    test_alloc_operations()
    test_to_buffer_operations()
    test_to_tensor_operations()
    test_subview_operations()
