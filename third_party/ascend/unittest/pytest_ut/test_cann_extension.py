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
import pytest
import triton
import torch
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import index_put, gather_out_to_ub, scatter_ub_to_out
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir, buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.compiler.errors import MLIRCompilationError
from triton.backends.ascend import _apply_ascend_patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

_apply_ascend_patch()


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    arch = "Ascend910_9589"


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


# Test cases for core.py
@triton.jit
def kernel_core_functions(M: tl.constexpr, N: tl.constexpr):
    # Test buffer allocation with different address spaces
    buf_ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    buf_l1 = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L1)
    buf_l0a = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L0A)
    buf_l0b = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L0B)
    buf_l0c = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L0C)

    # Test int64
    int64_val = al.int64(1234567890123456789)

    # Test sub_vec_id and sub_vec_num
    vec_id = al.sub_vec_id()
    vec_num = al.sub_vec_num()

    # Test debug_barrier
    al.debug_barrier(al.SYNC_IN_VF.VV_ALL)

    # Test fixpipe (simple usage)
    data = tl.full([M, N], 0.0, dtype=tl.float32)
    result_fixpipe = al.fixpipe(data, buf_ub)


# Test cases for math_ops.py
@triton.jit
def kernel_math_ops(M: tl.constexpr, N: tl.constexpr):
    # Test atan2
    x = tl.full([M, N], 1.0, dtype=tl.float32)
    y = tl.full([M, N], 1.0, dtype=tl.float32)
    result_atan2 = al.atan2(y, x)

    # Test isfinited
    result_isfinited = al.isfinited(x)

    # Test finitef
    result_finitef = al.finitef(x)


# Test cases for vec_ops.py
@triton.jit
def kernel_vec_ops(M: tl.constexpr, N: tl.constexpr):
    # Test insert_slice
    data = tl.full([M, N], 0.0, dtype=tl.float32)
    slice_data = tl.full([M // 2, N // 2], 1.0, dtype=tl.float32)
    result_insert = al.insert_slice(data, slice_data, [0, 0], [M // 2, N // 2], [1, 1])

    # Test extract_slice
    result_extract = al.extract_slice(data, [0, 0], [M // 2, N // 2], [1, 1])

    # Test get_element
    result_get = al.get_element(data, [0, 0])

    # Test cast
    data_int = tl.full([M, N], 1, dtype=tl.int32)
    result_cast = al.cast(data_int, tl.float32)


# Test cases for aux_ops.py
@triton.jit
def kernel_aux_ops(M: tl.constexpr, N: tl.constexpr):
    # Test parallel
    data = tl.full([M, N], 0.0, dtype=tl.float32)
    for _ in al.parallel(M):
        pass

    # Test compile_hint
    al.compile_hint(data, "core_mode")

    # Test multibuffer
    al.multibuffer(data, 2)


# Test cases for scope.py
@triton.jit
def kernel_scope(M: tl.constexpr, N: tl.constexpr):
    # Test scope with core_mode
    data = tl.full([M, N], 0.0, dtype=tl.float32)
    with al.scope(core_mode="vector"):
        result = data + 1.0


# Test cases for sync operations
@triton.jit
def kernel_sync_operations(M: tl.constexpr, N: tl.constexpr):
    # Test sync_block_all
    al.sync_block_all("all", 0)

    # Test sync_block_set and sync_block_wait
    al.sync_block_set("cube", "vector", 0)
    al.sync_block_wait("cube", "vector", 0)


# Test cases for custom_op.py
@triton.jit
def kernel_custom_op(M: tl.constexpr, N: tl.constexpr):
    # Test custom op functionality (placeholder)
    data = tl.full([M, N], 0.0, dtype=tl.float32)
    # Actual custom op usage would depend on specific ops


@triton.jit
def kernel_index_put_simple(value_ptr, index_ptr, dst_ptr):
    # index tile shape: [2]
    index_local = tl.arange(0, 2)
    x1_local = tl.arange(0, 2)[None, :]  # shape=(1,2)

    index_tile = tl.load(index_ptr + index_local)
    value_tile = tl.load(value_ptr + index_local[:, None] * 2 + x1_local)

    index_put(ptr=dst_ptr, index=index_tile, value=value_tile, dim=0, index_boundary=4, end_offset=(2, 2),
              start_offset=(0, 0), dst_stride=(2, 1))


@triton.jit
def simple_gather_kernel(src_ptr, index_ptr, out_ptr):
    # index tile shape: [2,2]
    y0_local = tl.arange(0, 2)[:, None]  # [0,1] rows
    x1_local = tl.arange(0, 2)[None, :]  # [0,1] cols
    mask = (y0_local < 2) & (x1_local < 2)

    # Load index tile to UB
    index = tl.load(index_ptr + y0_local * 2 + x1_local, mask)

    # Call gather_out_to_ub: gather values from src along dim=0
    gathered = gather_out_to_ub(src=src_ptr, index=index, index_boundary=4, dim=0, src_stride=(2, 1), end_offset=(2, 2),
                                start_offset=(0, 0))

    tl.store(out_ptr + y0_local * 2 + x1_local, gathered, mask)


@triton.jit
def simple_scatter_kernel(value_ptr, index_ptr, dst_ptr):
    # index tile shape: [2,2]
    y0_local = tl.arange(0, 2)[:, None]  # [0,1] rows
    x1_local = tl.arange(0, 2)[None, :]  # [0,1] cols
    mask = (y0_local < 2) & (x1_local < 2)

    value = tl.load(value_ptr + y0_local * 2 + x1_local, mask)
    index = tl.load(index_ptr + y0_local * 2 + x1_local, mask)

    scatter_ub_to_out(ptr=dst_ptr, value=value, index=index, index_boundary=4, dim=0, dst_stride=(2, 1),
                      end_offset=(2, 2), start_offset=(0, 0))


@triton.jit
def simple_scatter_kernel(value_ptr, index_ptr, dst_ptr):
    # index tile shape: [2,2]
    y0_local = tl.arange(0, 2)[:, None]  # [0,1] rows
    x1_local = tl.arange(0, 2)[None, :]  # [0,1] cols
    mask = (y0_local < 2) & (x1_local < 2)

    value = tl.load(value_ptr + y0_local * 2 + x1_local, mask)
    index = tl.load(index_ptr + y0_local * 2 + x1_local, mask)

    scatter_ub_to_out(ptr=dst_ptr, value=value, index=index, index_boundary=4, dim=0, dst_stride=(2, 1),
                      end_offset=(2, 2), start_offset=(0, 0))


# Test function definitions
def test_core_functions():
    print("=" * 60)
    print("Test 1: Core Functions")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_core_functions,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_math_ops():
    print("=" * 60)
    print("Test 2: Math Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_math_ops,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_mem_ops():
    print("=" * 60)
    print("Test 3: Memory Operations (EXPECTED TO FAIL)")
    print("=" * 60)

    # 构造输入（必须放在测试内，防止模块加载时报错）
    value = torch.randn(3, 3, device='npu').half()
    index = torch.tensor([0, 1], device='npu')
    dst = torch.zeros(4, 2, device='npu').half()

    # ✅ 核心：把编译+运行的代码包在这里，pytest 才能捕获
    with pytest.warns(FutureWarning, match="index_put"), pytest.raises(MLIRCompilationError):
        # 这一行会触发编译，直接抛出你要的错误
        kernel_index_put_simple[(1, )](value, index, dst)

    print("✅ 成功捕获预期的 MLIRCompilationError 错误！测试通过！")


def test_mem2_ops():
    print("=" * 60)
    print("Test 3: Memory Operations (EXPECTED TO FAIL)")
    print("=" * 60)

    # 构造输入（必须放在测试内，防止模块加载时报错）
    src = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], device='npu')
    index = torch.tensor([[0, 1], [2, 3]], device='npu')
    out = torch.empty((2, 2), device='npu', dtype=torch.float32)

    with pytest.warns(FutureWarning, match="gather_out_to_ub"), pytest.raises(MLIRCompilationError):
        simple_gather_kernel[(1, )](src, index, out)

    print("✅ 成功捕获预期的 MLIRCompilationError 错误！测试通过！")


def test_mem3_ops():
    print("=" * 60)
    print("Test 3: Memory Operations (EXPECTED TO FAIL)")
    print("=" * 60)

    # 构造输入（必须放在测试内，防止模块加载时报错）
    dst = torch.zeros((4, 2), device='npu', dtype=torch.float32)
    value = torch.tensor([[1., 2.], [3., 4.]], device='npu')
    index = torch.tensor([[1, 2], [3, 0]], device='npu')

    with pytest.warns(FutureWarning, match="scatter_ub_to_out"), pytest.raises(MLIRCompilationError):
        simple_scatter_kernel[(1, )](value, index, dst)

    print("✅ 成功捕获预期的 MLIRCompilationError 错误！测试通过！")


def test_mem4_ops():
    print("=" * 60)
    print("Test 3: Memory Operations (EXPECTED TO FAIL)")
    print("=" * 60)

    # 构造输入（必须放在测试内，防止模块加载时报错）
    dst = torch.zeros((4, 2), device='npu', dtype=torch.float32)
    value = torch.tensor([[1., 2.], [3., 4.]], device='npu')
    index = torch.tensor([[1, 2], [3, 0]], device='npu')

    with pytest.warns(FutureWarning, match="scatter_ub_to_out"), pytest.raises(MLIRCompilationError):
        simple_scatter_kernel[(1, )](value, index, dst)

    print("✅ 成功捕获预期的 MLIRCompilationError 错误！测试通过！")


def test_vec_ops():
    print("=" * 60)
    print("Test 4: Vector Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_vec_ops,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_aux_ops():
    print("=" * 60)
    print("Test 5: Auxiliary Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_aux_ops,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_scope():
    print("=" * 60)
    print("Test 6: Scope Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_scope,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_sync_operations():
    print("=" * 60)
    print("Test 7: Sync Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_sync_operations,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_custom_op():
    print("=" * 60)
    print("Test 8: Custom Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_custom_op,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


# ============== Main for manual testing ==============
if __name__ == "__main__":
    test_core_functions()
    test_math_ops()
    test_mem_ops()
    test_mem2_ops()
    test_mem3_ops()
    test_mem4_ops()
    test_vec_ops()
    test_aux_ops()
    test_scope()
    test_sync_operations()
    test_custom_op()
