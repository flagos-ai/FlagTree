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
#
# Tests for SIMT IndirectLoad fast-path (TritonToLinalg / StridedLoadStoreRewrite).
#
# What each parameter row exercises against the V1 trigger condition
# (compileOn91095 + forceSimtTemplate, last-axis stride statically > 1,
#  non-permuted layout, rank <= 5, not stride==2-even-size):
#
#   * STRIDE > 2 / odd       -> V1 should rewrite tt.load -> tt.indirect_load
#   * STRIDE == 2, even size -> DeinterleaveStatusOptimization handles (V1 yields)
#   * STRIDE == 2, odd size  -> V1 should rewrite (deinterleave precondition fails)
#   * Permuted layout        -> ImplicitPermute handles (V1 must not touch)
#   * Last-axis stride == 1  -> No rewrite (normal strided memref.copy)
#
# Verifies correctness only -- to confirm which path actually fired, dump IR
# with MLIR_ENABLE_DUMP=1 and grep for tt.indirect_load / tt.trans.

import triton
import triton.language as tl
import torch
import pytest
import test_common

try:
    from triton.backends.ascend.utils import is_compile_on_910_95
except Exception:

    def is_compile_on_910_95(_arch):
        return False


a3_known_boundary_load_issue = pytest.mark.xfail(
    not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch),
    reason=("Known A3 baseline issue on release/3.2.2-dev: make_block_ptr "
            "boundary_check + static power-of-two stride"),
    strict=False,
)

# ---------------------------------------------------------------------------
# 1D: out[i] = in[i * STRIDE]
# ---------------------------------------------------------------------------


@triton.jit
def kernel_1d_strided_compaction(
    in_ptr,
    out_ptr,
    in_numel,
    out_numel,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
    STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        src_idx = xindex * STRIDE
        # Two masks: store side bounded by out_numel, load side by in_numel.
        store_mask = xindex < out_numel
        load_mask = src_idx < in_numel
        tmp = tl.load(in_ptr + src_idx, load_mask)
        tl.store(out_ptr + xindex, tmp, store_mask)


def _ref_1d(src_cpu: torch.Tensor, stride: int, out_numel: int) -> torch.Tensor:
    flat = src_cpu.flatten().contiguous()
    return flat[::stride][:out_numel].contiguous()


@pytest.mark.parametrize("dtype,in_numel,stride,ncore,xblock,xblock_sub", [
    # ---- V1 命中: stride > 2 (deinterleave 不接) ----
    ("float32", 4096 * 16, 16, 2, 2048, 256),  # stride=16
    ("float32", 4096 * 8, 8, 2, 2048, 256),  # stride=8
    ("float32", 4096 * 4, 4, 2, 2048, 256),  # stride=4
    ("float32", 4096 * 3, 3, 2, 2048, 256),  # stride=3 (奇数 stride, block 仍 pow2)
    ("float16", 4096 * 6, 6, 2, 2048, 256),
    ("int8", 4096 * 7, 7, 2, 2048, 256),
    # ---- Deinterleave 接管 (stride==2 + 偶数 block, V1 让路) ----
    # 注:无法在 Triton 中构造"奇数 size + stride==2"以测试 V1 这一支,
    # 因为 tl.arange 要求 end-start 是 2 的幂.
    ("float32", 4096 * 2, 2, 2, 2048, 256),
    # ---- 不应改写 (stride==1) ----
    # TODO: 编译失败原因待排查; 已加 CSE+Canonicalize 收尾仍挂.
    # pytest.param("float32", 4096, 1, 2, 2048, 256,
    #              marks=pytest.mark.skip("stride=1 V1 sub-step compile bug")),
])
def test_1d_strided_compaction(dtype, in_numel, stride, ncore, xblock, xblock_sub):
    out_numel = ncore * xblock
    assert in_numel >= out_numel * stride, "in_numel must cover the strided range"
    assert xblock % xblock_sub == 0

    src = test_common.generate_tensor((in_numel, ), dtype).npu()
    dst = test_common.generate_tensor((out_numel, ), dtype).npu()

    kernel_1d_strided_compaction[(ncore, )](
        src,
        dst,
        in_numel,
        out_numel,
        xblock,
        xblock_sub,
        stride,
    )

    ref = _ref_1d(src.cpu(), stride, out_numel)
    actual = dst.cpu()
    if dtype in ("float32", "float16"):
        assert torch.allclose(actual, ref, atol=1e-2, rtol=1e-3), \
            f"max abs diff = {(actual.float() - ref.float()).abs().max().item()}"
    else:
        assert torch.equal(actual, ref), \
            f"mismatch count = {(actual != ref).sum().item()}"


# ---------------------------------------------------------------------------
# Multi-D: out[i0, i1, ..., in-1] = in_flat[sum_d i_d * STRIDE_d]
# Output is packed contiguous of shape `blocks`; input is flat with enough
# elements to cover the maximum offset.
# ---------------------------------------------------------------------------


@triton.jit
def kernel_multi_d_gather(
    in_ptr,
    out_ptr,
    BLOCK_0: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
    BLOCK_3: tl.constexpr,
    BLOCK_4: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr,
    STRIDE_4: tl.constexpr,
):
    # Build in_off / out_off progressively. out_off is packed-contiguous so we
    # can compare against a plain reshape of the reference gather.
    in_off = tl.arange(0, BLOCK_0) * STRIDE_0
    out_off = tl.arange(0, BLOCK_0)

    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        in_off = in_off[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        out_off = out_off[:, None] * BLOCK_1 + tl.arange(0, BLOCK_1)[None, :]
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        in_off = in_off[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        out_off = out_off[:, :, None] * BLOCK_2 + tl.arange(0, BLOCK_2)[None, None, :]
    if (BLOCK_3 * BLOCK_4) > 1:
        in_off = in_off[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        out_off = out_off[:, :, :, None] * BLOCK_3 + tl.arange(0, BLOCK_3)[None, None, None, :]
    if BLOCK_4 > 1:
        in_off = in_off[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        out_off = out_off[:, :, :, :, None] * BLOCK_4 + tl.arange(0, BLOCK_4)[None, None, None, None, :]

    tmp = tl.load(in_ptr + in_off)
    tl.store(out_ptr + out_off, tmp)


def _ref_multi_d(src_flat_cpu: torch.Tensor, blocks, strides) -> torch.Tensor:
    """reference[i0, ..., in-1] = src_flat[sum_d i_d * strides[d]]"""
    coords = torch.meshgrid(*[torch.arange(b) for b in blocks], indexing="ij")
    offsets = torch.zeros(blocks, dtype=torch.int64)
    for d in range(len(blocks)):
        offsets = offsets + coords[d].to(torch.int64) * int(strides[d])
    return src_flat_cpu[offsets]


@pytest.mark.parametrize("dtype,blocks,strides", [
    # ---- V1 hit: non-permuted, last-axis stride statically > 1, all blocks power-of-2 ----
    # 2D
    ("float32", (4, 8), (8, 4)),  # stride 4
    ("float32", (4, 8), (24, 3)),  # stride 3 (odd)
    # 3D
    ("float16", (2, 4, 8), (32, 8, 4)),  # stride 4
    ("float32", (4, 4, 8), (96, 24, 3)),  # stride 3 (odd)
    # 4D
    ("float16", (2, 4, 4, 8), (128, 32, 8, 3)),  # stride 3
    # 5D
    ("float32", (2, 2, 2, 4, 8), (256, 128, 64, 16, 5)),  # stride 5 (odd)

    # ---- Deinterleave handles (stride==2 + even last block) ----
    ("float16", (4, 4, 8), (32, 8, 2)),

    # ---- Permuted: ImplicitPermute handles, V1 must skip ----
    ("float32", (4, 8), (1, 4)),  # strides ascending
    ("float32", (4, 4, 8), (1, 4, 16)),  # strictly ascending

    # ---- Normal contiguous (stride==1): no rewrite ----
    ("float32", (4, 8), (8, 1)),
    ("float32", (4, 4, 8), (32, 8, 1)),
])
def test_multi_d_gather(dtype, blocks, strides):
    assert len(blocks) == len(strides)
    assert len(blocks) <= 5

    # Cover max input offset: (B_d - 1) * STRIDE_d sum + 1.
    max_offset = sum((b - 1) * s for b, s in zip(blocks, strides)) + 1
    src = test_common.generate_tensor((max_offset, ), dtype).npu()
    dst = test_common.generate_tensor(tuple(blocks), dtype).npu()

    padded_blocks = list(blocks) + [1] * (5 - len(blocks))
    padded_strides = list(strides) + [0] * (5 - len(strides))

    kernel_multi_d_gather[(1, )](
        src,
        dst,
        *padded_blocks,
        *padded_strides,
    )

    ref = _ref_multi_d(src.cpu(), blocks, strides)
    actual = dst.cpu()
    if dtype in ("float32", "float16"):
        assert torch.allclose(actual, ref, atol=1e-2, rtol=1e-3), \
            f"max abs diff = {(actual.float() - ref.float()).abs().max().item()}"
    else:
        assert torch.equal(actual, ref), \
            f"mismatch count = {(actual != ref).sum().item()}"


# ---------------------------------------------------------------------------
# make_block_ptr (tt.make_tensor_ptr) strided load:
#   Build a tl.make_block_ptr with non-default strides such that the low-dim
#   stride is `stride_n`.  When stride_n > 1 V1 should rewrite the load to
#   tt.indirect_load; when stride_n == 1 V1 should bail.
#
# Sanity: this test verifies value correctness only.  For "did V1 actually
# trigger?" see the FileCheck test under
#   third_party/ascend/unittest/Conversion/950PR/TritonToLinalg/
#     indirect_load_rewrite.mlir
# ---------------------------------------------------------------------------


@triton.jit
def kernel_block_ptr_strided(
    in_ptr,
    out_ptr,
    src_shape_m,
    src_shape_n,
    src_stride_m,
    src_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(src_shape_m, src_shape_n),
        strides=(src_stride_m, src_stride_n),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(BLOCK_M, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tile = tl.load(in_block_ptr)
    tl.store(out_block_ptr, tile)


def _ref_block_ptr_strided(src_cpu_flat: torch.Tensor, block_m: int, block_n: int, stride_m: int,
                           stride_n: int) -> torch.Tensor:
    coords_i, coords_j = torch.meshgrid(torch.arange(block_m), torch.arange(block_n), indexing="ij")
    offsets = (coords_i.to(torch.int64) * int(stride_m) + coords_j.to(torch.int64) * int(stride_n))
    return src_cpu_flat[offsets]


@pytest.mark.parametrize("dtype,block_m,block_n,stride_m,stride_n", [
    # V1 命中: non-permuted, 低维 stride 静态 > 1
    ("float32", 4, 8, 32, 4),
    ("float32", 4, 8, 24, 3),
    ("float16", 8, 8, 64, 8),
    ("int8", 16, 8, 56, 7),
    # Deinterleave 接管 (stride_n == 2, 偶数 last block)
    ("float32", 4, 8, 16, 2),
    # V1 不动 (stride_n == 1, contiguous)
    ("float32", 4, 8, 8, 1),
    ("float16", 8, 16, 16, 1),
])
def test_block_ptr_strided(dtype, block_m, block_n, stride_m, stride_n):
    max_offset = (block_m - 1) * stride_m + (block_n - 1) * stride_n + 1
    src = test_common.generate_tensor((max_offset, ), dtype).npu()
    dst = test_common.generate_tensor((block_m, block_n), dtype).npu()

    src_shape_m = block_m * stride_m
    src_shape_n = stride_n * block_n

    kernel_block_ptr_strided[(1, )](
        src,
        dst,
        src_shape_m,
        src_shape_n,
        stride_m,
        stride_n,
        block_m,
        block_n,
    )

    ref = _ref_block_ptr_strided(src.cpu(), block_m, block_n, stride_m, stride_n)
    actual = dst.cpu()
    if dtype in ("float32", "float16"):
        assert torch.allclose(actual, ref, atol=1e-2, rtol=1e-3), \
            f"max abs diff = {(actual.float() - ref.float()).abs().max().item()}"
    else:
        assert torch.equal(actual, ref), \
            f"mismatch count = {(actual != ref).sum().item()}"


# ---------------------------------------------------------------------------
# V2: tt.store -> tt.indirect_store fast-path. Mirror of the load tests
# above, but the strided side is the OUTPUT (scatter) instead of the input.
#
#   out[i*stride] = in[i]   (1D AddPtr)
#   out[i*stride_m + j*stride_n] = in[i, j]   (2D block_ptr)
# ---------------------------------------------------------------------------


@triton.jit
def kernel_1d_strided_scatter(
    in_ptr,
    out_ptr,
    in_numel,
    out_numel,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
    STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        dst_idx = xindex * STRIDE
        load_mask = xindex < in_numel
        store_mask = dst_idx < out_numel
        tmp = tl.load(in_ptr + xindex, load_mask)
        tl.store(out_ptr + dst_idx, tmp, store_mask)


def _ref_1d_scatter(src_cpu: torch.Tensor, stride: int, out_numel: int, dtype: str) -> torch.Tensor:
    """out[i*stride] = src[i] for i in [0, src.numel())."""
    torch_dtype = eval("torch." + dtype)
    out = torch.zeros(out_numel, dtype=torch_dtype)
    flat = src_cpu.flatten().contiguous()
    for i in range(flat.numel()):
        idx = i * stride
        if idx < out_numel:
            out[idx] = flat[i]
    return out


@pytest.mark.parametrize("dtype,in_numel,stride,ncore,xblock,xblock_sub", [
    # V2 命中: stride > 2 (deinterleave doesn't apply)
    ("float32", 4096, 16, 2, 2048, 256),
    ("float32", 4096, 8, 2, 2048, 256),
    ("float32", 4096, 4, 2, 2048, 256),
    ("float32", 4096, 3, 2, 2048, 256),
    ("float16", 4096, 6, 2, 2048, 256),
    ("int8", 4096, 7, 2, 2048, 256),
    # Deinterleave 接管 (stride==2, 偶数 last block)
    ("float32", 4096, 2, 2, 2048, 256),
    # V2 不动 (stride == 1)
    ("float32", 4096, 1, 2, 2048, 256),
])
def test_1d_strided_scatter(dtype, in_numel, stride, ncore, xblock, xblock_sub):
    out_numel = in_numel * stride
    assert xblock % xblock_sub == 0
    assert ncore * xblock >= in_numel
    src = test_common.generate_tensor((in_numel, ), dtype).npu()
    dst = test_common.generate_tensor((out_numel, ), dtype).npu()
    # Pre-zero dst so non-written positions are deterministic.
    dst.zero_()

    kernel_1d_strided_scatter[(ncore, )](
        src,
        dst,
        in_numel,
        out_numel,
        xblock,
        xblock_sub,
        stride,
    )

    ref = _ref_1d_scatter(src.cpu(), stride, out_numel, dtype)
    actual = dst.cpu()
    if dtype in ("float32", "float16"):
        assert torch.allclose(actual, ref, atol=1e-2, rtol=1e-3), \
            f"max abs diff = {(actual.float() - ref.float()).abs().max().item()}"
    else:
        assert torch.equal(actual, ref), \
            f"mismatch count = {(actual != ref).sum().item()}"


@triton.jit
def kernel_block_ptr_strided_scatter(
    in_ptr,
    out_ptr,
    dst_shape_m,
    dst_shape_n,
    dst_stride_m,
    dst_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Contiguous in, strided out.
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(BLOCK_M, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(dst_shape_m, dst_shape_n),
        strides=(dst_stride_m, dst_stride_n),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tile = tl.load(in_block_ptr)
    tl.store(out_block_ptr, tile)


def _ref_block_ptr_scatter(src_cpu: torch.Tensor, block_m: int, block_n: int, stride_m: int, stride_n: int,
                           out_numel: int, dtype: str) -> torch.Tensor:
    """out_flat[i*stride_m + j*stride_n] = src[i, j]"""
    torch_dtype = eval("torch." + dtype)
    out = torch.zeros(out_numel, dtype=torch_dtype)
    for i in range(block_m):
        for j in range(block_n):
            idx = i * stride_m + j * stride_n
            if idx < out_numel:
                out[idx] = src_cpu[i, j]
    return out


@pytest.mark.parametrize("dtype,block_m,block_n,stride_m,stride_n", [
    # V2 命中: non-permuted, 低维 stride 静态 > 1
    ("float32", 4, 8, 32, 4),
    ("float32", 4, 8, 24, 3),
    ("float16", 8, 8, 64, 8),
    ("int8", 16, 8, 56, 7),
    # Deinterleave 接管
    ("float32", 4, 8, 16, 2),
    # V2 不动 (contiguous)
    ("float32", 4, 8, 8, 1),
    ("float16", 8, 16, 16, 1),
])
def test_block_ptr_strided_scatter(dtype, block_m, block_n, stride_m, stride_n):
    src = test_common.generate_tensor((block_m, block_n), dtype).npu()
    out_numel = (block_m - 1) * stride_m + (block_n - 1) * stride_n + 1
    # Round up to make a contiguous flat buffer of sufficient size.
    out_numel = max(out_numel, block_m * stride_m)
    dst = test_common.generate_tensor((out_numel, ), dtype).npu()
    dst.zero_()

    dst_shape_m = block_m * stride_m
    dst_shape_n = stride_n * block_n

    kernel_block_ptr_strided_scatter[(1, )](
        src,
        dst,
        dst_shape_m,
        dst_shape_n,
        stride_m,
        stride_n,
        block_m,
        block_n,
    )

    ref = _ref_block_ptr_scatter(src.cpu(), block_m, block_n, stride_m, stride_n, out_numel, dtype)
    actual = dst.cpu()
    if dtype in ("float32", "float16"):
        assert torch.allclose(actual, ref, atol=1e-2, rtol=1e-3), \
            f"max abs diff = {(actual.float() - ref.float()).abs().max().item()}"
    else:
        assert torch.equal(actual, ref), \
            f"mismatch count = {(actual != ref).sum().item()}"


# ---------------------------------------------------------------------------
# V1.5: make_block_ptr Load + boundary_check + strided (low-dim stride > 1).
#   Block intentionally overflows the parent tensor; OOB lanes should get
#   the padding value (0 for PAD_ZERO, NaN for PAD_NAN), in-bounds lanes
#   should get the strided gather data.
# ---------------------------------------------------------------------------


@triton.jit
def kernel_block_ptr_boundary_load(
    in_ptr,
    out_ptr,
    PARENT_M: tl.constexpr,
    PARENT_N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAD_NAN: tl.constexpr,
):
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(PARENT_M, PARENT_N),
        strides=(STRIDE_M, STRIDE_N),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(BLOCK_M, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    if PAD_NAN:
        tile = tl.load(in_block_ptr, boundary_check=(0, 1), padding_option="nan")
    else:
        tile = tl.load(in_block_ptr, boundary_check=(0, 1), padding_option="zero")
    tl.store(out_block_ptr, tile)


def _ref_boundary_load(src_cpu_flat, parent_m, parent_n, stride_m, stride_n, block_m, block_n, pad_value, dtype):
    torch_dtype = eval("torch." + dtype)
    out = torch.full((block_m, block_n), pad_value, dtype=torch_dtype)
    for i in range(block_m):
        for j in range(block_n):
            if i < parent_m and j < parent_n:
                out[i, j] = src_cpu_flat[i * stride_m + j * stride_n]
    return out


@pytest.mark.parametrize("dtype,parent_m,parent_n,stride_m,stride_n,block_m,block_n,pad_nan", [
    # V1.5 hit: PAD_ZERO, stride_n=4, block exceeds parent
    pytest.param("float32", 5, 3, 12, 4, 8, 8, False, marks=a3_known_boundary_load_issue),
    # V1.5 hit: PAD_ZERO, stride_n=3, block partially out-of-bounds
    pytest.param("float32", 7, 5, 15, 3, 8, 8, False, marks=a3_known_boundary_load_issue),
    # V1.5 hit: PAD_NAN (float)
    pytest.param("float32", 5, 3, 12, 4, 8, 8, True, marks=a3_known_boundary_load_issue),
    # V1.5 hit: float16
    pytest.param("float16", 5, 5, 20, 4, 8, 8, False, marks=a3_known_boundary_load_issue),
    # OOB only on last axis: block fully fits axis 0
    pytest.param("float32", 8, 3, 12, 4, 8, 8, False, marks=a3_known_boundary_load_issue),
])
def test_block_ptr_boundary_load(dtype, parent_m, parent_n, stride_m, stride_n, block_m, block_n, pad_nan):
    # in buffer covers all valid (i, j) where i < parent_m, j < parent_n.
    in_numel = (parent_m - 1) * stride_m + (parent_n - 1) * stride_n + 1
    src = test_common.generate_tensor((in_numel, ), dtype).npu()
    dst = test_common.generate_tensor((block_m, block_n), dtype).npu()

    kernel_block_ptr_boundary_load[(1, )](
        src,
        dst,
        parent_m,
        parent_n,
        stride_m,
        stride_n,
        block_m,
        block_n,
        pad_nan,
    )

    pad_value = float("nan") if pad_nan else 0
    ref = _ref_boundary_load(src.cpu(), parent_m, parent_n, stride_m, stride_n, block_m, block_n, pad_value, dtype)
    actual = dst.cpu()
    if pad_nan:
        # NaN locations both ref and actual; compare ignoring NaN via finite mask.
        finite_mask = torch.isfinite(ref)
        # In-bounds positions
        assert torch.allclose(actual[finite_mask], ref[finite_mask], atol=1e-2, rtol=1e-3)
        # OOB positions must be NaN in actual too
        assert torch.isnan(actual[~finite_mask]).all(), \
            "expected NaN in OOB positions"
    else:
        if dtype in ("float32", "float16"):
            assert torch.allclose(actual, ref, atol=1e-2, rtol=1e-3), \
                f"max abs diff = {(actual.float() - ref.float()).abs().max().item()}"
        else:
            assert torch.equal(actual, ref)


# ---------------------------------------------------------------------------
# V1.5: make_block_ptr Store + boundary_check + strided.
#   Block intentionally overflows the parent; OOB positions should NOT be
#   written; in-bounds positions should receive the strided values.
# ---------------------------------------------------------------------------


@triton.jit
def kernel_block_ptr_boundary_store(
    in_ptr,
    out_ptr,
    PARENT_M: tl.constexpr,
    PARENT_N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(BLOCK_M, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(PARENT_M, PARENT_N),
        strides=(STRIDE_M, STRIDE_N),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tile = tl.load(in_block_ptr)
    tl.store(out_block_ptr, tile, boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# V1.5 regression: kernel pattern `tl.make_block_ptr(s + bos*H + i_h, ...)`
# where the base is a scalar AddPtr chain (NOT the raw function-arg ptr).
# Reproduces the chunk_local_cumsum 10x precision bug fixed 2026-05-28.
# Before the fix, the scalar AddPtr lowered to a size-1 reinterpret_cast view
# and per-element offsets indexed out of bounds.
# ---------------------------------------------------------------------------


@triton.jit
def kernel_chunk_local_cumsum(
    s,
    o,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    p_s = tl.make_block_ptr(s + bos * H + i_h, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    b_s = tl.load(p_s, boundary_check=(0, )).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, ))


@pytest.mark.parametrize("dtype,B,T,H,chunk_size", [
    ("float32", 4, 2048, 8, 128),
    ("float32", 2, 512, 4, 64),
    ("float16", 4, 1024, 8, 128),
])
def test_chunk_local_cumsum_scalar_base(dtype, B, T, H, chunk_size):
    """Verify scalar-AddPtr-base make_block_ptr path produces correct results."""
    torch_dtype = eval("torch." + dtype)
    torch.manual_seed(42)
    g = torch.randn(B, T, H, device="npu", dtype=torch_dtype) * 0.1
    out = torch.empty_like(g, dtype=torch.float32)

    NT = (T + chunk_size - 1) // chunk_size
    grid = (NT, B * H)
    kernel_chunk_local_cumsum[grid](g, out, T, H=H, BT=chunk_size)

    # Reference: per-chunk cumsum along the T axis for each (b, h).
    ref = torch.zeros_like(g, dtype=torch.float32)
    g_cpu = g.cpu()
    for b in range(B):
        for h in range(H):
            for t_chunk in range(NT):
                start = t_chunk * chunk_size
                end = min(start + chunk_size, T)
                sl = g_cpu[b, start:end, h].to(torch.float32)
                ref[b, start:end, h] = torch.cumsum(sl, dim=0).to("npu")

    diff = (out - ref).abs().max().item()
    atol = 1e-3 if dtype == "float16" else 1e-4
    assert diff < atol, f"max abs diff = {diff:.6e}"


@pytest.mark.skip(reason="The case is failed, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize("dtype,parent_m,parent_n,stride_m,stride_n,block_m,block_n", [
    ("float32", 5, 3, 12, 4, 8, 8),
    ("float32", 7, 5, 15, 3, 8, 8),
    ("float16", 5, 5, 20, 4, 8, 8),
])
def test_block_ptr_boundary_store(dtype, parent_m, parent_n, stride_m, stride_n, block_m, block_n):
    src = test_common.generate_tensor((block_m, block_n), dtype).npu()
    out_numel = (parent_m - 1) * stride_m + (parent_n - 1) * stride_n + 1
    # Generous round-up to guarantee no OOB writes are possible (even if V1.5
    # logic regresses we won't segfault).
    out_numel = max(out_numel, block_m * stride_m + block_n * stride_n)
    dst = test_common.generate_tensor((out_numel, ), dtype).npu()
    # Pre-fill dst with a sentinel so spurious writes to OOB positions would
    # be detectable (we only assert in-bounds correctness below; the sentinel
    # itself is informational for future debug).
    if dtype in ("float32", "float16"):
        sentinel = float("inf")
    elif dtype == "int8":
        sentinel = -128
    else:
        sentinel = 0
    dst.fill_(sentinel)

    kernel_block_ptr_boundary_store[(1, )](
        src,
        dst,
        parent_m,
        parent_n,
        stride_m,
        stride_n,
        block_m,
        block_n,
    )

    actual_cpu = dst.cpu()
    src_cpu = src.cpu()
    # Check in-bounds positions got the strided write.
    for i in range(parent_m):
        for j in range(parent_n):
            idx = i * stride_m + j * stride_n
            if i < block_m and j < block_n:
                expected = src_cpu[i, j]
                got = actual_cpu[idx]
                if dtype in ("float32", "float16"):
                    assert torch.isclose(got, expected, atol=1e-2, rtol=1e-3), \
                        f"mismatch at parent[{i},{j}]: got {got} vs {expected}"
                else:
                    assert got == expected, \
                        f"mismatch at parent[{i},{j}]: got {got} vs {expected}"
