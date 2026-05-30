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
def helper_use_q(q, XBLOCK: tl.constexpr):
    tmp = q + 1.0
    tl.device_assert(tmp.shape[0] == XBLOCK if isinstance(tmp.shape[0], tl.constexpr) else True)


@triton.jit
def helper_use_buffer(q_l1, XBLOCK: tl.constexpr):
    q = bl.to_tensor(q_l1)
    helper_use_q(q, XBLOCK)


@triton.jit
def kernel_buffer_arg(XBLOCK: tl.constexpr):
    q_l1 = bl.alloc(tl.float32, [XBLOCK], al.ascend_address_space.L1)
    helper_use_buffer(q_l1, XBLOCK)


@triton.jit
def kernel_scope_for_if_call_buffer(x_ptr, XBLOCK: tl.constexpr):
    q_l1 = bl.alloc(tl.float32, [XBLOCK], al.ascend_address_space.L1)
    offsets = tl.arange(0, XBLOCK)
    with al.scope(core_mode="vector"):
        for skv_loop_idx in range(2):
            if skv_loop_idx == 0:
                q = tl.load(x_ptr + offsets, mask=offsets < XBLOCK)
                bl.to_buffer(tensor=q, bind_buffer=q_l1)
            else:
                q = bl.to_tensor(q_l1)
            helper_use_buffer(q_l1, XBLOCK)


def test_buffer_argument_to_jit_function():
    mlir = compile_kernel(kernel_buffer_arg, {}, {"XBLOCK": 16})
    assert len(mlir) > 0


def test_buffer_helper_call_inside_scope_loop_if():
    mlir = compile_kernel(kernel_scope_for_if_call_buffer, {"x_ptr": "*fp32"}, {"XBLOCK": 16})
    assert "scf.for" in mlir
    assert "scope.scope" in mlir
