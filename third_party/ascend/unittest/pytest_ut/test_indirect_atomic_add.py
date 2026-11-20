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

# =============================================================================
# Indirect atomic_add test suite
#
# Test matrix (access pattern x tensor rank):
#
#  | access pattern                                  | 1-D  | 2-D  | 3-D  | 4-D  | 5-D  |
#  |-------------------------------------------------|------|------|------|------|------|
#  | structured pointer + discrete mask              | (A1) | (A2) | (A3) | (A4) | (A5) |
#  | partial structured: high-dim disc + low-dim cont|  -   | (B2) | (B3) | (B4) | (B5) |
#  | fully unstructured indirect offsets             | (C1) | (C2) | (C3) | (C4) | (C5) |
#  | loaded mask + masked-off out-of-bounds offsets   | (D1) |  -   |  -   |  -   |  -   |
#  | scalar-splat mask shared by load and atomic      | (E1) |  -   |  -   |  -   |  -   |
#  | scalar-splat mask + used atomic return value     | (F1) |  -   |  -   |  -   |  -   |
#
# Notes:
# 1. Case A exercises the structured-pointer discrete-mask atomic_add path.
# 2. Case B exercises a single high-dimension discrete remap with the
#    remaining lower dimensions kept contiguous.
# 3. Case C exercises fully unstructured indirect offsets.
# 4. Case D verifies that a loaded discrete mask guards loaded invalid offsets.
# 5. Case E verifies that a scalar-splat mask shared by an indirect load and
#    atomic_add still guards inactive programs.
# 6. Case F verifies that the old-value copy and atomic store use the same
#    scalar-splat-masked address range, including inactive invalid offsets.
# 7. Cases A-D and F validate both the final destination tensor and the atomic_add
#    return value, which must be the old value observed at each access.
# =============================================================================

import math

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95

_TARGET_ARCH = triton.runtime.driver.active.get_current_target().arch

SUPPORTED_DTYPES = [
    ("int8", torch.int8),
    ("int16", torch.int16),
    ("int32", torch.int32),
    ("int64", torch.int64),
    ("float16", torch.float16),
    ("float32", torch.float32),
    ("bfloat16", torch.bfloat16),
    ("uint32", torch.uint32),
    ("uint64", torch.uint64),
]

RANK_SHAPES = {
    1: (8, ),
    2: (4, 4),
    3: (2, 3, 4),
    4: (2, 2, 3, 4),
    5: (2, 2, 2, 3, 4),
}

PARTIAL_STRUCTURED_SHAPES = {
    2: (2, 16),
    3: (2, 2, 8),
    4: (2, 2, 2, 8),
    5: (2, 2, 2, 2, 8),
}

TEST_RANKS = [3]
TEST_DTYPE = [("int32", torch.int32), ("bfloat16", torch.bfloat16)]


@triton.jit
def structured_disc_mask_atomic_add_1d(
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
):
    i0 = tl.arange(0, D0)
    linear = i0
    mask = (linear * 2) < D0
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + linear, value, mask=mask)
    tl.store(old_ptr + linear, old, mask=mask)


@triton.jit
def loaded_mask_oob_offset_atomic_add_1d(
    idx_ptr,
    mask_ptr,
    val_ptr,
    out_ptr,
    old_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    linear = tl.arange(0, BLOCK_SIZE)
    offset = tl.load(idx_ptr + linear)
    mask = tl.load(mask_ptr + linear) != 0
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + offset, value, mask=mask)
    tl.store(old_ptr + linear, old, mask=mask)


@triton.jit(do_not_specialize=["numel"])
def scalar_splat_mask_atomic_add_1d(
    topk_ids_ptr,
    tokens_cnts_ptr,
    numel,
    TOKENS_PER_THREAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * TOKENS_PER_THREAD + tl.arange(0, TOKENS_PER_THREAD)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(tokens_cnts_ptr + expert_id, 1, mask=mask)


@triton.jit(do_not_specialize=["numel"])
def scalar_splat_mask_atomic_add_return_value_1d(
    idx_ptr,
    out_ptr,
    old_ptr,
    numel,
):
    pid = tl.program_id(0)
    offset = tl.load(idx_ptr + pid)
    mask = pid < numel
    old = tl.atomic_add(out_ptr + offset, 1, mask=mask)
    tl.store(old_ptr + pid, old, mask=mask)


@triton.jit
def structured_disc_mask_atomic_add_2d(
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None]
    i1 = tl.arange(0, D1)[None, :]
    linear = i0 * D1 + i1
    mask = (linear * 2) < (D0 * D1)
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + linear, value, mask=mask)
    tl.store(old_ptr + linear, old, mask=mask)


@triton.jit
def structured_disc_mask_atomic_add_3d(
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None]
    i1 = tl.arange(0, D1)[None, :, None]
    i2 = tl.arange(0, D2)[None, None, :]
    linear = (i0 * D1 + i1) * D2 + i2
    mask = (linear * 2) < (D0 * D1 * D2)
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + linear, value, mask=mask)
    tl.store(old_ptr + linear, old, mask=mask)


@triton.jit
def structured_disc_mask_atomic_add_4d(
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None, None]
    i1 = tl.arange(0, D1)[None, :, None, None]
    i2 = tl.arange(0, D2)[None, None, :, None]
    i3 = tl.arange(0, D3)[None, None, None, :]
    linear = ((i0 * D1 + i1) * D2 + i2) * D3 + i3
    mask = (linear * 2) < (D0 * D1 * D2 * D3)
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + linear, value, mask=mask)
    tl.store(old_ptr + linear, old, mask=mask)


@triton.jit
def structured_disc_mask_atomic_add_5d(
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None, None, None]
    i1 = tl.arange(0, D1)[None, :, None, None, None]
    i2 = tl.arange(0, D2)[None, None, :, None, None]
    i3 = tl.arange(0, D3)[None, None, None, :, None]
    i4 = tl.arange(0, D4)[None, None, None, None, :]
    linear = (((i0 * D1 + i1) * D2 + i2) * D3 + i3) * D4 + i4
    mask = (linear * 2) < (D0 * D1 * D2 * D3 * D4)
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + linear, value, mask=mask)
    tl.store(old_ptr + linear, old, mask=mask)


@triton.jit
def fully_unstructured_atomic_add_1d(
    idx_ptr,
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
):
    i0 = tl.arange(0, D0)
    linear = i0
    offset = tl.load(idx_ptr + linear)
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + offset, value)
    tl.store(old_ptr + linear, old)


@triton.jit
def fully_unstructured_atomic_add_2d(
    idx_ptr,
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None]
    i1 = tl.arange(0, D1)[None, :]
    linear = i0 * D1 + i1
    offset = tl.load(idx_ptr + linear)
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + offset, value)
    tl.store(old_ptr + linear, old)


@triton.jit
def fully_unstructured_atomic_add_3d(
    idx_ptr,
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None]
    i1 = tl.arange(0, D1)[None, :, None]
    i2 = tl.arange(0, D2)[None, None, :]
    linear = (i0 * D1 + i1) * D2 + i2
    offset = tl.load(idx_ptr + linear)
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + offset, value)
    tl.store(old_ptr + linear, old)


@triton.jit
def fully_unstructured_atomic_add_4d(
    idx_ptr,
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None, None]
    i1 = tl.arange(0, D1)[None, :, None, None]
    i2 = tl.arange(0, D2)[None, None, :, None]
    i3 = tl.arange(0, D3)[None, None, None, :]
    linear = ((i0 * D1 + i1) * D2 + i2) * D3 + i3
    offset = tl.load(idx_ptr + linear)
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + offset, value)
    tl.store(old_ptr + linear, old)


@triton.jit
def fully_unstructured_atomic_add_5d(
    idx_ptr,
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None, None, None]
    i1 = tl.arange(0, D1)[None, :, None, None, None]
    i2 = tl.arange(0, D2)[None, None, :, None, None]
    i3 = tl.arange(0, D3)[None, None, None, :, None]
    i4 = tl.arange(0, D4)[None, None, None, None, :]
    linear = (((i0 * D1 + i1) * D2 + i2) * D3 + i3) * D4 + i4
    offset = tl.load(idx_ptr + linear)
    value = tl.load(val_ptr + linear)
    old = tl.atomic_add(out_ptr + offset, value)
    tl.store(old_ptr + linear, old)


@triton.jit
def partial_structured_atomic_add_2d(
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None]
    remapped_i0 = (i0 + 1) % D0
    inner_idx = tl.arange(0, D1)[None, :]
    linear = i0 * D1 + inner_idx
    value = tl.load(val_ptr + linear)
    ptr = out_ptr + remapped_i0 * D1 + inner_idx
    old = tl.atomic_add(ptr, value)
    tl.store(old_ptr + linear, old)


@triton.jit
def partial_structured_atomic_add_3d(
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None]
    remapped_i0 = (i0 + 1) % D0
    i1 = tl.arange(0, D1)[None, :]
    i2 = tl.arange(0, D2)[None, None, :]
    linear = (i0 * D1 + i1)[:, :, None] * D2 + i2
    value = tl.load(val_ptr + linear)
    ptr = out_ptr + ((remapped_i0 * D1 + i1)[:, :, None] * D2 + i2)
    old = tl.atomic_add(ptr, value)
    tl.store(old_ptr + linear, old)


@triton.jit
def partial_structured_atomic_add_4d(
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None]
    remapped_i0 = (i0 + 1) % D0
    i1 = tl.arange(0, D1)[None, :, None]
    i2 = tl.arange(0, D2)[None, None, :]
    i3 = tl.arange(0, D3)[None, None, None, :]
    linear = (((i0 * D1 + i1) * D2 + i2)[:, :, :, None] * D3 + i3)
    value = tl.load(val_ptr + linear)
    ptr = out_ptr + ((((remapped_i0 * D1 + i1) * D2 + i2)[:, :, :, None] * D3) + i3)
    old = tl.atomic_add(ptr, value)
    tl.store(old_ptr + linear, old)


@triton.jit
def partial_structured_atomic_add_5d(
    val_ptr,
    out_ptr,
    old_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
):
    i0 = tl.arange(0, D0)[:, None, None, None]
    remapped_i0 = (i0 + 1) % D0
    i1 = tl.arange(0, D1)[None, :, None, None]
    i2 = tl.arange(0, D2)[None, None, :, None]
    i3 = tl.arange(0, D3)[None, None, None, :]
    i4 = tl.arange(0, D4)[None, None, None, None, :]
    linear = ((((i0 * D1 + i1) * D2 + i2) * D3 + i3)[:, :, :, :, None] * D4 + i4)
    value = tl.load(val_ptr + linear)
    ptr = out_ptr + (((((remapped_i0 * D1 + i1) * D2 + i2) * D3 + i3)[:, :, :, :, None] * D4) + i4)
    old = tl.atomic_add(ptr, value)
    tl.store(old_ptr + linear, old)


STRUCTURED_DISC_MASK_KERNELS = {
    1: structured_disc_mask_atomic_add_1d,
    2: structured_disc_mask_atomic_add_2d,
    3: structured_disc_mask_atomic_add_3d,
    4: structured_disc_mask_atomic_add_4d,
    5: structured_disc_mask_atomic_add_5d,
}

PARTIAL_STRUCTURED_KERNELS = {
    2: partial_structured_atomic_add_2d,
    3: partial_structured_atomic_add_3d,
    4: partial_structured_atomic_add_4d,
    5: partial_structured_atomic_add_5d,
}

FULLY_UNSTRUCTURED_KERNELS = {
    1: fully_unstructured_atomic_add_1d,
    2: fully_unstructured_atomic_add_2d,
    3: fully_unstructured_atomic_add_3d,
    4: fully_unstructured_atomic_add_4d,
    5: fully_unstructured_atomic_add_5d,
}


def _build_partial_structured_offsets(shape):
    assert len(shape) >= 2
    first_dim = shape[0]
    tail_shape = shape[1:]
    tail_numel = math.prod(tail_shape)
    remapped_first = ((torch.arange(first_dim, dtype=torch.int64) + 1) % first_dim)
    first_offsets = remapped_first.reshape((first_dim, ) + (1, ) * len(tail_shape))
    tail_offsets = torch.arange(tail_numel, dtype=torch.int64).reshape((1, ) + tail_shape)
    return first_offsets * tail_numel + tail_offsets, math.prod(shape)


def _build_fully_unstructured_offsets(shape):
    numel = math.prod(shape)
    output_numel = numel * 2
    base = torch.arange(numel, dtype=torch.int64)
    offsets = (base % 2) * numel + (base // 2)
    offsets = offsets.reshape(shape)
    return offsets, output_numel


def _build_discrete_mask(shape):
    numel = math.prod(shape)
    return ((torch.arange(numel, dtype=torch.int64).reshape(shape) * 2) < numel)


def _build_value_tensor(shape, dtype):
    return torch.ones(shape, dtype=dtype)


def _build_output_baseline(output_numel, dtype):
    if dtype in (torch.uint32, torch.uint64):
        return torch.arange(output_numel, dtype=torch.int64).to(dtype)
    return torch.arange(output_numel, dtype=dtype)


def _simulate_atomic_add(base_output, offsets, values, mask=None):
    compute_dtype = torch.int64 if base_output.dtype in (torch.uint32, torch.uint64) else base_output.dtype
    expected_output = base_output.reshape(-1).clone().cpu().to(compute_dtype)
    flat_offsets = offsets.reshape(-1).to(torch.int64).cpu()
    flat_values = values.reshape(-1).cpu().to(compute_dtype)
    flat_old = torch.zeros(flat_values.shape, dtype=compute_dtype)
    if mask is None:
        flat_mask = torch.ones(flat_offsets.shape, dtype=torch.bool)
    else:
        flat_mask = mask.reshape(-1).cpu()

    for idx in range(flat_offsets.numel()):
        if not bool(flat_mask[idx]):
            continue
        offset = int(flat_offsets[idx].item())
        flat_old[idx] = expected_output[offset]
        expected_output[offset] = expected_output[offset] + flat_values[idx]

    return expected_output.to(base_output.dtype), flat_old.reshape(values.shape).to(base_output.dtype)


def _assert_equal(actual, expected, dtype_name, rank, scenario):
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    assert torch.equal(actual_cpu, expected_cpu), (f"scenario={scenario}, dtype={dtype_name}, rank={rank}\n"
                                                   f"Expected:\n{expected_cpu}\nGot:\n{actual_cpu}")


def _launch_structured_discrete_mask(rank, values, output, old, shape):
    kernel = STRUCTURED_DISC_MASK_KERNELS[rank]
    kwargs = {f"D{dim}": size for dim, size in enumerate(shape)}
    kernel[(1, )](values, output, old, **kwargs)


def _launch_partial_structured(rank, values, output, old, shape):
    kernel = PARTIAL_STRUCTURED_KERNELS[rank]
    kwargs = {f"D{dim}": size for dim, size in enumerate(shape)}
    kernel[(1, )](values, output, old, **kwargs)


def _launch_fully_unstructured(rank, offsets, values, output, old, shape):
    kernel = FULLY_UNSTRUCTURED_KERNELS[rank]
    kwargs = {f"D{dim}": size for dim, size in enumerate(shape)}
    kernel[(1, )](offsets, values, output, old, **kwargs)


@pytest.mark.parametrize("dtype_name, torch_dtype", TEST_DTYPE)
@pytest.mark.parametrize("rank", TEST_RANKS)
def test_atomic_add_structured_pointer_with_discrete_mask(dtype_name, torch_dtype, rank):
    if not is_compile_on_910_95(_TARGET_ARCH) and torch_dtype in (torch.uint32, torch.uint64):
        pytest.skip("uint32 and uint64 atomics are only supported on 950")
    shape = RANK_SHAPES[rank]
    values = _build_value_tensor(shape, torch_dtype).npu()
    output_numel = math.prod(shape)
    baseline = _build_output_baseline(output_numel, torch_dtype)
    output = baseline.clone().npu()
    old = torch.zeros(shape, dtype=torch_dtype).npu()

    _launch_structured_discrete_mask(rank, values, output, old, shape)

    identity_offsets = torch.arange(output_numel, dtype=torch.int64).reshape(shape)
    mask = _build_discrete_mask(shape)
    expected_output, expected_old = _simulate_atomic_add(
        baseline,
        identity_offsets,
        _build_value_tensor(shape, torch_dtype),
        mask=mask,
    )
    _assert_equal(output, expected_output, dtype_name, rank, "structured+discrete-mask/output")
    _assert_equal(old, expected_old, dtype_name, rank, "structured+discrete-mask/old")


@pytest.mark.parametrize("dtype_name, torch_dtype", TEST_DTYPE)
@pytest.mark.parametrize("rank", TEST_RANKS)
def test_atomic_add_partially_structured_indirect_offsets(dtype_name, torch_dtype, rank):
    if rank == 1:
        pytest.skip("Partially structured test is not applicable to 1-D tensors")
    if not is_compile_on_910_95(_TARGET_ARCH) and torch_dtype in (torch.uint32, torch.uint64):
        pytest.skip("uint32 and uint64 atomics are only supported on 950")
    shape = PARTIAL_STRUCTURED_SHAPES[rank]
    offsets, output_numel = _build_partial_structured_offsets(shape)
    values = _build_value_tensor(shape, torch_dtype).npu()
    baseline = _build_output_baseline(output_numel, torch_dtype)
    output = baseline.clone().npu()
    old = torch.zeros(shape, dtype=torch_dtype).npu()

    _launch_partial_structured(rank, values, output, old, shape)

    expected_output, expected_old = _simulate_atomic_add(
        baseline,
        offsets,
        _build_value_tensor(shape, torch_dtype),
    )
    _assert_equal(output, expected_output, dtype_name, rank, "partially-structured-indirect/output")
    _assert_equal(old, expected_old, dtype_name, rank, "partially-structured-indirect/old")


@pytest.mark.parametrize("dtype_name, torch_dtype", TEST_DTYPE)
@pytest.mark.parametrize("rank", TEST_RANKS)
def test_atomic_add_fully_unstructured_indirect_offsets(dtype_name, torch_dtype, rank):
    if not is_compile_on_910_95(_TARGET_ARCH) and torch_dtype in (torch.uint32, torch.uint64):
        pytest.skip("uint32 and uint64 atomics are only supported on 950")
    shape = RANK_SHAPES[rank]
    offsets, output_numel = _build_fully_unstructured_offsets(shape)
    values = _build_value_tensor(shape, torch_dtype).npu()
    baseline = _build_output_baseline(output_numel, torch_dtype)
    output = baseline.clone().npu()
    old = torch.zeros(shape, dtype=torch_dtype).npu()

    _launch_fully_unstructured(rank, offsets.npu(), values, output, old, shape)

    expected_output, expected_old = _simulate_atomic_add(
        baseline,
        offsets,
        _build_value_tensor(shape, torch_dtype),
    )
    _assert_equal(output, expected_output, dtype_name, rank, "fully-unstructured-indirect/output")
    _assert_equal(old, expected_old, dtype_name, rank, "fully-unstructured-indirect/old")


def test_atomic_add_loaded_mask_skips_oob_offsets():
    if not is_compile_on_910_95(_TARGET_ARCH):
        pytest.skip("masked indirect atomic template is only enabled on 910_95/950")

    block_size = 8
    invalid_offset = 1 << 30
    offsets = torch.tensor(
        [0, invalid_offset, 1, invalid_offset, 2, invalid_offset, 3, invalid_offset],
        dtype=torch.int64,
    )
    mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.int32)
    values = torch.ones(block_size, dtype=torch.int32)
    baseline = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    old_sentinel = -777

    output = baseline.clone().npu()
    old = torch.full((block_size, ), old_sentinel, dtype=torch.int32).npu()
    loaded_mask_oob_offset_atomic_add_1d[(1, )](
        offsets.npu(),
        mask.npu(),
        values.npu(),
        output,
        old,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()

    expected_output = baseline + 1
    expected_old = torch.tensor(
        [10, old_sentinel, 20, old_sentinel, 30, old_sentinel, 40, old_sentinel],
        dtype=torch.int32,
    )
    _assert_equal(output, expected_output, "int32", 1, "loaded-mask-oob-offset/output")
    _assert_equal(old, expected_old, "int32", 1, "loaded-mask-oob-offset/old")


def test_atomic_add_scalar_splat_mask_skips_inactive_programs():
    numel = 3
    grid_size = 8
    tokens_per_thread = 1
    sentinel_expert = 3

    # The inactive programs point to a valid sentinel bucket so the regression
    # is detected deterministically without deliberately triggering an MTE OOB.
    topk_ids = torch.tensor(
        [0, 1, 2] + [sentinel_expert] * (grid_size - numel),
        dtype=torch.int64,
    ).npu()
    tokens_cnts = torch.zeros(sentinel_expert + 1, dtype=torch.int32).npu()

    scalar_splat_mask_atomic_add_1d[(grid_size, )](
        topk_ids,
        tokens_cnts,
        numel,
        TOKENS_PER_THREAD=tokens_per_thread,
    )
    torch.npu.synchronize()

    expected = torch.tensor([1, 1, 1, 0], dtype=torch.int32)
    assert torch.equal(tokens_cnts.cpu(), expected)


def test_atomic_add_scalar_splat_mask_return_value_skips_oob_offsets():
    numel = 3
    grid_size = 8
    invalid_offset = 1 << 30
    old_sentinel = -777

    offsets = torch.tensor(
        [0, 1, 2] + [invalid_offset] * (grid_size - numel),
        dtype=torch.int64,
    ).npu()
    baseline = torch.tensor([10, 20, 30], dtype=torch.int32)
    output = baseline.clone().npu()
    old = torch.full((grid_size, ), old_sentinel, dtype=torch.int32).npu()

    scalar_splat_mask_atomic_add_return_value_1d[(grid_size, )](
        offsets,
        output,
        old,
        numel,
    )
    torch.npu.synchronize()

    expected_output = baseline + 1
    expected_old = torch.tensor(
        [10, 20, 30] + [old_sentinel] * (grid_size - numel),
        dtype=torch.int32,
    )
    _assert_equal(output, expected_output, "int32", 1, "scalar-splat-mask-return-value/output")
    _assert_equal(old, expected_old, "int32", 1, "scalar-splat-mask-return-value/old")
