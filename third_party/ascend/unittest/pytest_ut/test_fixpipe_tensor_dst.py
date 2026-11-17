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
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend import _apply_ascend_patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
DEFAULT_A5_ARCH = "Ascend910_9589"
HACC_TARGET_ATTR = f'hacc.target = #hacc.target<"{DEFAULT_A5_ARCH}">'

_apply_ascend_patch()


def make_options(arch):

    class Options:
        num_warps = 4
        num_stages = 3
        num_ctas = 1
        cluster_dims = (1, 1, 1)
        enable_fp_fusion = True
        debug = False
        sanitize_overflow = True

    Options.arch = arch
    return Options()


def compile_kernel(kernel, signature, constants, *, arch=None):
    """Compile to TTIR mlir; mirrors triton.compile frontend + ascend patch."""
    arch = arch or DEFAULT_A5_ARCH
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, make_options(arch), {}, {})
    return str(module)


@triton.jit
def fixpipe_without_dst(
    A_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    row_matmul = tl.program_id(0)

    offs_i = tl.arange(0, tl.constexpr(M))[:, None]
    offs_k = tl.arange(0, K)

    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)

    result = al.fixpipe(a_vals, dual_dst_mode=al.FixpipeDualDstMode.NO_DUAL)
    tl.store(a_ptrs, result)


@triton.jit
def fixpipe_with_buffer_dst(
    A_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    row_matmul = tl.program_id(0)

    offs_i = tl.arange(0, tl.constexpr(M))[:, None]
    offs_k = tl.arange(0, K)

    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)

    ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    al.fixpipe(a_vals, ub, dual_dst_mode=al.FixpipeDualDstMode.NO_DUAL)


@pytest.mark.parametrize("M, K, N", [(16, 16, 16)])
def test_fixpipe_without_dst(M, K, N):
    mlir = compile_kernel(
        fixpipe_without_dst,
        {"A_ptr": "*fp32"},
        {"M": M, "K": K, "N": N},
    )
    assert len(mlir) > 0
    assert "hivm.hir.fixpipe" in mlir
    assert HACC_TARGET_ATTR in mlir
    assert HACC_TARGET_ATTR in mlir


@triton.jit
def fixpipe_row_split(
    A_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    row_matmul = tl.program_id(0)

    offs_i = tl.arange(0, tl.constexpr(M))[:, None]
    offs_k = tl.arange(0, K)

    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)

    result = al.fixpipe(a_vals, dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)


@triton.jit
def fixpipe_column_split(
    A_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    row_matmul = tl.program_id(0)

    offs_i = tl.arange(0, tl.constexpr(M))[:, None]
    offs_k = tl.arange(0, K)

    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)

    result = al.fixpipe(a_vals, dual_dst_mode=al.FixpipeDualDstMode.COLUMN_SPLIT)


@pytest.mark.parametrize("M, K, N", [(32, 16, 16)])
def test_fixpipe_row_split(M, K, N):
    mlir = compile_kernel(
        fixpipe_row_split,
        {"A_ptr": "*fp32"},
        {"M": M, "K": K, "N": N},
    )
    assert len(mlir) > 0
    assert "dual_dst_mode = <ROW_SPLIT>" in mlir
    assert HACC_TARGET_ATTR in mlir


@pytest.mark.parametrize("M, K, N", [(16, 16, 32)])
def test_fixpipe_column_split(M, K, N):
    mlir = compile_kernel(
        fixpipe_column_split,
        {"A_ptr": "*fp32"},
        {"M": M, "K": K, "N": N},
    )
    assert len(mlir) > 0
    assert "dual_dst_mode = <COLUMN_SPLIT>" in mlir
    assert HACC_TARGET_ATTR in mlir


@pytest.mark.parametrize("M, K, N", [(16, 16, 16)])
def test_fixpipe_with_buffer_dst(M, K, N):
    mlir = compile_kernel(
        fixpipe_with_buffer_dst,
        {"A_ptr": "*fp32"},
        {"M": M, "K": K, "N": N},
    )
    assert len(mlir) > 0
    assert "hivm.hir.fixpipe" in mlir
    assert HACC_TARGET_ATTR in mlir
