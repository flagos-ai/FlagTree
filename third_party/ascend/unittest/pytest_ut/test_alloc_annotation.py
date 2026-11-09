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
"""Tests for the annotation bl.alloc puts on a buffer."""
import os

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir, buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.utils import is_compile_on_910_95

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    sanitize_overflow = True


def compile_kernel(kernel, signature, constants):
    """Compile a kernel to TTIR."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {}, {})
    return str(module)


def mark_attrs(mlir):
    """Attribute dict of every annotation.mark in mlir."""
    return [
        line.split("{", 1)[1].split("}", 1)[0]
        for line in mlir.splitlines()
        if "annotation.mark" in line and "{" in line
    ]


@triton.jit
def alloc_unused_buffer(x_ptr, y_ptr, XBLOCK: tl.constexpr):
    offsets = tl.arange(0, XBLOCK)
    x = tl.load(x_ptr + offsets)
    bl.alloc(tl.float32, [XBLOCK], al.ascend_address_space.UB)
    tl.store(y_ptr + offsets, x * 2.0)


@triton.jit
def alloc_unique_unused_buffer(x_ptr, y_ptr, XBLOCK: tl.constexpr):
    offsets = tl.arange(0, XBLOCK)
    x = tl.load(x_ptr + offsets)
    bl.alloc(tl.float32, [XBLOCK], al.ascend_address_space.UB, is_mem_unique=True)
    tl.store(y_ptr + offsets, x + 1.0)


@triton.jit
def alloc_ub_as_fixpipe_dst(a_ptr, b_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n)
    acc = tl.dot(a, b)
    ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    al.fixpipe(acc, ub, al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.NO_DUAL)
    tl.store(y_ptr + offs_m * N + offs_n, bl.to_tensor(ub))


@triton.jit
def alloc_unique_ub_as_fixpipe_dst(a_ptr, b_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n)
    acc = tl.dot(a, b)
    ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB, is_mem_unique=True)
    al.fixpipe(acc, ub, al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.NO_DUAL)
    tl.store(y_ptr + offs_m * N + offs_n, bl.to_tensor(ub))


def _run_fixpipe_kernel(kernel, m, n, k):
    """Run kernel on NPU, return (result, expected)."""
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")
    y = torch.zeros(m, n, dtype=torch.float32, device="npu")

    kernel[(1, )](a, b, y, M=m, N=n, K=k)
    torch.npu.synchronize()
    return y.cpu(), a.cpu().float() @ b.cpu().float()


def _check_fixpipe_result(kernel, what):
    got, expected = _run_fixpipe_kernel(kernel, 64, 64, 64)
    max_diff = float((got - expected).abs().max())
    assert max_diff < 0.5, f"{what} mismatch: max_abs_diff={max_diff:.3e} {got[0, :4]} != {expected[0, :4]}"


@pytest.mark.skipif(
    not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch),
    reason="only support in Ascend 950",
)
def test_alloc_as_fixpipe_dst_on_npu():
    _check_fixpipe_result(alloc_ub_as_fixpipe_dst, "fixpipe into bl.alloc buffer")


@pytest.mark.skipif(
    not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch),
    reason="only support in Ascend 950",
)
def test_unique_alloc_as_fixpipe_dst_on_npu():
    _check_fixpipe_result(alloc_unique_ub_as_fixpipe_dst, "fixpipe into mem_unique buffer")


def test_alloc_does_not_annotate_memory_effects():
    signature = {"x_ptr": "*fp32", "y_ptr": "*fp32"}
    consts = {"XBLOCK": 1024}

    mlir = compile_kernel(alloc_unused_buffer, signature, consts)
    assert not any("effects" in a for a in mark_attrs(mlir)), \
        "bl.alloc must not annotate write/read effects:\n" + mlir

    mlir = compile_kernel(alloc_unique_unused_buffer, signature, consts)
    attrs = mark_attrs(mlir)
    assert any("mem_unique" in a for a in attrs), "is_mem_unique must stay annotated:\n" + mlir
    assert not any("effects" in a for a in attrs), \
        "bl.alloc must not annotate write/read effects:\n" + mlir
