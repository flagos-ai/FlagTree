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

import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir, buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    sanitize_overflow = True


def compile_kernel(kernel, signature, constants):
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(
        kernel,
        src,
        context,
        Options(),
        {"create_address_space": al.semantic.create_address_space},
        {},
    )
    return str(module)


@triton.jit
def kernel_device_assert_with_mask(x_ptr, XBLOCK: tl.constexpr):
    offsets = tl.arange(0, XBLOCK)
    x = tl.load(x_ptr + offsets)
    cond = x > 0
    mask = offsets < XBLOCK
    tl.device_assert(cond, msg="x must be positive", mask=mask)


def test_device_assert_with_mask():
    mlir = compile_kernel(kernel_device_assert_with_mask, {"x_ptr": "*fp32"}, {"XBLOCK": 16})
    assert "device_assert" in mlir or "hivm.hir.device_assert" in mlir, "device_assert with mask not found in MLIR"
