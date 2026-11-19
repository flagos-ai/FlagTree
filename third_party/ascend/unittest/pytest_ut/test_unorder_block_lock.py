import pytest
import torch
import triton
import triton.language as tl

import test_common
from triton.backends.ascend.utils import is_compile_on_910_95


@triton.jit
def store_mask_split_dim0(
    in_ptr,
    out_ptr,
    mask_ptr,
    cache_modifier: tl.constexpr,
    eviction_policy: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
    BLOCK_3: tl.constexpr,
    BLOCK_4: tl.constexpr,
    BLOCK_5: tl.constexpr,
    BLOCK_6: tl.constexpr,
    BLOCK_7: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr,
    STRIDE_4: tl.constexpr,
    STRIDE_5: tl.constexpr,
    STRIDE_6: tl.constexpr,
    STRIDE_7: tl.constexpr,
):
    # Each program processes one complete slice of dimension 0.
    offsets = tl.program_id(0) * STRIDE_0
    offsets = offsets + tl.arange(0, BLOCK_1) * STRIDE_1
    offsets = offsets[:, None] + tl.arange(0, BLOCK_2)[None, :] * STRIDE_2
    offsets = offsets[:, :, None] + tl.arange(0, BLOCK_3)[None, None, :] * STRIDE_3
    offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, :] * STRIDE_4
    offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, :] * STRIDE_5
    offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, :] * STRIDE_6
    offsets = offsets[:, :, :, :, :, :, None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, :] * STRIDE_7

    mask = tl.load(mask_ptr + offsets) != 0
    value = tl.load(in_ptr + offsets)
    tl.store(
        out_ptr + offsets,
        value,
        mask=mask,
        cache_modifier=cache_modifier,
        eviction_policy=eviction_policy,
    )


@pytest.mark.parametrize("shape", [(15, 2, 2, 2, 3, 2, 2, 2)])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_unorder_block_lock(shape, dtype):
    mask = test_common.generate_tensor(shape, "bool").npu()
    value = test_common.generate_tensor(shape, dtype).npu()
    actual = test_common.generate_tensor(shape, dtype).npu()
    original = actual.clone()

    blocks = list(value.size())
    strides = list(value.stride())
    grid = (shape[0], )
    store_mask_split_dim0[grid](
        value,
        actual,
        mask,
        None,
        None,
        *blocks[1:],
        *strides,
    )
    torch.npu.synchronize()

    expected = torch.where(mask, value, original)
    torch.testing.assert_close(actual.cpu(), expected.cpu())


CV_BLOCK_M = 64
CV_BLOCK_N = 64
CV_BLOCK_K = 64


@triton.jit
def simd_unorder_block_lock_a5(
    x_ptr,
    group_id_ptr,
    row_index_ptr,
    tile_start_ptr,
    weight_ptr,
    out_ptr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wg: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    M: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_tile_id = tl.program_id(0)
    m_tile_id = tl.program_id(1)
    tile_start = tl.load(tile_start_ptr + m_tile_id)
    # Some programs skip the guarded region, so lock participation is unordered.
    if tile_start >= M:
        return

    m_offsets = tile_start + tl.arange(0, BLOCK_M)
    group_ids = tl.load(group_id_ptr + m_offsets, mask=m_offsets < M, other=GROUPS)
    group_id = tl.min(group_ids)
    row_mask = group_ids == group_id
    # Indirect loads make the AIV path contain SIMT work.
    row_indices = tl.load(row_index_ptr + m_offsets, mask=row_mask, other=0)

    k_offsets = tl.arange(0, BLOCK_K)
    n_offsets = n_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + row_indices[:, None] * stride_xm + k_offsets[None, :] * stride_xk
    weight_ptrs = (weight_ptr + group_id * stride_wg + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn)
    x = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
    weight = tl.load(weight_ptrs)
    # Dot adds Cube work and makes this a CV mixed kernel.
    acc = tl.dot(x, weight, out_dtype=tl.float32)

    # Build rank-6 output pointers and a non-contiguous mask for the discrete
    # mask store below. Discrete-mask rank > 5 selects the SIMD lowering path.
    m_store_offsets = tile_start
    m_store_offsets = m_store_offsets + tl.arange(0, 2)[:, None, None, None, None, None] * 32
    m_store_offsets = m_store_offsets + tl.arange(0, 2)[None, :, None, None, None, None] * 16
    m_store_offsets = m_store_offsets + tl.arange(0, 2)[None, None, :, None, None, None] * 8
    m_store_offsets = m_store_offsets + tl.arange(0, 2)[None, None, None, :, None, None] * 4
    m_store_offsets = m_store_offsets + tl.arange(0, 4)[None, None, None, None, :, None]
    n_store_offsets = n_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)[None, None, None, None, None, :]
    out_ptrs = out_ptr + m_store_offsets * stride_om + n_store_offsets * stride_on
    store_group_ids = tl.load(
        group_id_ptr + m_store_offsets,
        mask=m_store_offsets < M,
        other=GROUPS,
    )
    store_mask = (m_store_offsets < M) & (store_group_ids == group_id) & (n_store_offsets < BLOCK_N)
    simd_shape: tl.constexpr = (2, 2, 2, 2, 4, 64)
    acc = tl.reshape(acc.to(out_ptr.dtype.element_ty), simd_shape)
    # This rank-6 masked store triggers SIMD discrete-mask conversion, which
    # inserts the unordered block lock around the generated vector operation.
    tl.store(out_ptrs, acc, mask=store_mask)


def test_cv_simd_unorder_block_lock_a5():
    torch.manual_seed(0)
    device = torch.device("npu")
    dtype = torch.bfloat16
    m, k, n, groups = 96, CV_BLOCK_K, CV_BLOCK_N, 2

    x = torch.randn((m, k), device=device, dtype=dtype)
    weight = torch.randn((groups, k, n), device=device, dtype=dtype)
    group_ids = torch.cat((
        torch.zeros(48, device=device, dtype=torch.int64),
        torch.ones(48, device=device, dtype=torch.int64),
    ))
    row_indices = torch.arange(m, device=device, dtype=torch.int64)
    # The last two programs intentionally skip the guarded region.
    tile_starts = torch.tensor([0, 48, m, m], device=device, dtype=torch.int64)
    actual = torch.full((m, n), float("nan"), device=device, dtype=dtype)

    simd_unorder_block_lock_a5[(1, tile_starts.numel())](
        x,
        group_ids,
        row_indices,
        tile_starts,
        weight,
        actual,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        actual.stride(0),
        actual.stride(1),
        M=m,
        GROUPS=groups,
        BLOCK_M=CV_BLOCK_M,
        BLOCK_N=CV_BLOCK_N,
        BLOCK_K=CV_BLOCK_K,
    )
    torch.npu.synchronize()

    expected = torch.empty((m, n), device=device, dtype=dtype)
    expected[:48] = torch.matmul(x[:48].float(), weight[0].float()).to(dtype)
    expected[48:] = torch.matmul(x[48:].float(), weight[1].float()).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)


@triton.jit
def simd_three_unorder_block_locks(
    in_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    mask0_ptr,
    mask1_ptr,
    mask2_ptr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
    BLOCK_3: tl.constexpr,
    BLOCK_4: tl.constexpr,
    BLOCK_5: tl.constexpr,
    BLOCK_6: tl.constexpr,
    BLOCK_7: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr,
    STRIDE_4: tl.constexpr,
    STRIDE_5: tl.constexpr,
    STRIDE_6: tl.constexpr,
    STRIDE_7: tl.constexpr,
):
    offsets = tl.program_id(0) * STRIDE_0
    offsets = offsets + tl.arange(0, BLOCK_1) * STRIDE_1
    offsets = offsets[:, None] + tl.arange(0, BLOCK_2)[None, :] * STRIDE_2
    offsets = offsets[:, :, None] + tl.arange(0, BLOCK_3)[None, None, :] * STRIDE_3
    offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, :] * STRIDE_4
    offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, :] * STRIDE_5
    offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, :] * STRIDE_6
    offsets = offsets[:, :, :, :, :, :, None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, :] * STRIDE_7

    value = tl.load(in_ptr + offsets)
    mask0 = tl.load(mask0_ptr + offsets) != 0
    mask1 = tl.load(mask1_ptr + offsets) != 0
    mask2 = tl.load(mask2_ptr + offsets) != 0

    # Each rank-7 discrete mask below selects SIMD masked-store lowering and
    # therefore creates an independent unordered block lock.
    tl.store(out0_ptr + offsets, value, mask=mask0)
    tl.store(out1_ptr + offsets, value + 1.0, mask=mask1)
    tl.store(out2_ptr + offsets, value * 2.0, mask=mask2)


@pytest.mark.parametrize('shape', [(15, 2, 2, 2, 3, 2, 2, 2)])
@pytest.mark.parametrize('dtype', ['bfloat16'])
def test_simd_three_unorder_block_locks_a3_a5(shape, dtype):
    torch.manual_seed(0)
    value = test_common.generate_tensor(shape, dtype).npu()
    masks = [test_common.generate_tensor(shape, 'bool').npu() for _ in range(3)]
    actuals = [test_common.generate_tensor(shape, dtype).npu() for _ in range(3)]
    originals = [actual.clone() for actual in actuals]

    blocks = list(value.size())
    strides = list(value.stride())
    simd_three_unorder_block_locks[(shape[0], )](
        value,
        *actuals,
        *masks,
        *blocks[1:],
        *strides,
    )
    torch.npu.synchronize()

    stored_values = (value, value + 1.0, value * 2.0)
    for actual, original, mask, stored_value in zip(actuals, originals, masks, stored_values):
        expected = torch.where(mask, stored_value, original)
        torch.testing.assert_close(actual.cpu(), expected.cpu())


@triton.jit
def simd_unorder_block_lock_a3(
    x_ptr,
    group_id_ptr,
    row_index_ptr,
    tile_start_ptr,
    weight_ptr,
    out_ptr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wg: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    M: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_tile_id = tl.program_id(0)
    m_tile_id = tl.program_id(1)
    tile_start = tl.load(tile_start_ptr + m_tile_id)
    # Some programs skip the guarded region, so lock participation is unordered.
    if tile_start >= M:
        return

    m_offsets = tile_start + tl.arange(0, BLOCK_M)
    group_ids = tl.load(group_id_ptr + m_offsets, mask=m_offsets < M, other=GROUPS)
    group_id = tl.min(group_ids)
    row_mask = group_ids == group_id
    row_indices = tl.load(row_index_ptr + m_offsets, mask=row_mask, other=0)

    k_offsets = tl.arange(0, BLOCK_K)
    n_offsets = n_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + row_indices[:, None] * stride_xm + k_offsets[None, :] * stride_xk
    weight_ptrs = (weight_ptr + group_id * stride_wg + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn)
    x = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
    weight = tl.load(weight_ptrs)
    acc = tl.dot(x, weight, out_dtype=tl.float32)

    out_ptrs = out_ptr + m_offsets[:, None] * stride_om + n_offsets[None, :] * stride_on
    store_mask = row_mask[:, None] & (n_offsets[None, :] < BLOCK_N)
    # On A3 this original rank-2 discrete masked store selects SIMD lowering
    # and inserts the unordered block lock around the vector operation.
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=store_mask)


@pytest.mark.skipif(is_compile_on_910_95(), reason="requires A3")
def test_cv_simd_unorder_block_lock_a3():
    torch.manual_seed(0)
    device = torch.device("npu")
    dtype = torch.bfloat16
    m, k, n, groups = 96, CV_BLOCK_K, CV_BLOCK_N, 2

    x = torch.randn((m, k), device=device, dtype=dtype)
    weight = torch.randn((groups, k, n), device=device, dtype=dtype)
    group_ids = torch.cat((
        torch.zeros(48, device=device, dtype=torch.int64),
        torch.ones(48, device=device, dtype=torch.int64),
    ))
    row_indices = torch.arange(m, device=device, dtype=torch.int64)
    tile_starts = torch.tensor([0, 48, m, m], device=device, dtype=torch.int64)
    actual = torch.full((m, n), float("nan"), device=device, dtype=dtype)

    simd_unorder_block_lock_a3[(1, tile_starts.numel())](
        x,
        group_ids,
        row_indices,
        tile_starts,
        weight,
        actual,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        actual.stride(0),
        actual.stride(1),
        M=m,
        GROUPS=groups,
        BLOCK_M=CV_BLOCK_M,
        BLOCK_N=CV_BLOCK_N,
        BLOCK_K=CV_BLOCK_K,
    )
    torch.npu.synchronize()

    expected = torch.empty((m, n), device=device, dtype=dtype)
    expected[:48] = torch.matmul(x[:48].float(), weight[0].float()).to(dtype)
    expected[48:] = torch.matmul(x[48:].float(), weight[1].float()).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)
