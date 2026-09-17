import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.compiler.errors import CompilationError
from triton.experimental.tle.language.gpu.mthreads import (
    MusaDotOperandEncoding,
    MusaSqmmaEncoding,
    MusaWmmaEncoding,
)

from test_tle_utils import require_mthreads_libtriton

require_mthreads_libtriton()


def _musa_runtime_available():
    return hasattr(torch, "musa") and torch.musa.is_available()


requires_musa_runtime = pytest.mark.skipif(
    not _musa_runtime_available(),
    reason="MUSA runtime is not available",
)


def _block_layout_a():
    return tle.gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0])


def _block_layout_b():
    return tle.gpu.BlockEncoding([1, 1], [32, 1], [1, 4], [0, 1])


def _slice_layout():
    return tle.gpu.SlicedEncoding(0, _block_layout_a())


def _wmma_layout():
    return MusaWmmaEncoding([3, 1], [2, 2], [16, 8, 16])


def _sqmma_layout():
    return MusaSqmmaEncoding([3, 1], [4, 1], [32, 64, 16])


@triton.jit
def _set_layout_pointwise_kernel(src, out, n: tl.constexpr, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(src + offsets, mask=mask, other=0.0)
    values = tle.gpu.set_layout(values, LAYOUT)
    adjustment = tl.where(offsets % 3 == 0, 2.0, -1.0)
    result = values * 1.5 + adjustment
    pointers = tle.gpu.set_layout(out + offsets, LAYOUT)
    tl.store(pointers, result, mask=mask)


@triton.jit
def _set_layout_transpose_kernel(
    src,
    out,
    BLOCK: tl.constexpr,
    SRC_LAYOUT: tl.constexpr,
    DST_LAYOUT: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    offsets = rows * BLOCK + cols
    src_offsets = tle.gpu.set_layout(offsets, SRC_LAYOUT)
    values = tle.gpu.set_layout(tl.load(src + src_offsets), SRC_LAYOUT)
    result = tle.gpu.set_layout(tl.trans(values) * 2.0 + 1.0, DST_LAYOUT)
    dst_offsets = tle.gpu.set_layout(offsets, DST_LAYOUT)
    tl.store(out + dst_offsets, result)


@triton.jit
def _set_layout_multi_domain_kernel(
    src,
    out_a,
    out_b,
    BLOCK: tl.constexpr,
    LAYOUT_A: tl.constexpr,
    LAYOUT_B: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    offsets = rows * BLOCK + cols
    root = tl.load(src + offsets)
    values_a = tle.gpu.set_layout(root, LAYOUT_A)
    values_b = tle.gpu.set_layout(root, LAYOUT_B)
    pointers_a = tle.gpu.set_layout(out_a + offsets, LAYOUT_A)
    pointers_b = tle.gpu.set_layout(out_b + offsets, LAYOUT_B)
    tl.store(pointers_a, values_a + 1.25)
    tl.store(pointers_b, values_b * 3.0 - 2.0)


@triton.jit
def _set_layout_expand_dims_kernel(
    out_rows,
    out_cols,
    BLOCK: tl.constexpr,
    ROW_LAYOUT: tl.constexpr,
    COL_LAYOUT: tl.constexpr,
):
    values = tl.arange(0, BLOCK)
    rows = tle.gpu.set_layout(values[None, :, None], ROW_LAYOUT)
    cols = tle.gpu.set_layout(values[None, None, :], COL_LAYOUT)
    tl.store(out_rows + rows, rows.to(tl.float32) + 0.5)
    tl.store(out_cols + cols, cols.to(tl.float32) * 2.0)


@triton.jit(noinline=True)
def _set_layout_producer(src, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    return tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)


@triton.jit(noinline=True)
def _set_layout_bridge(src, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    values = _set_layout_producer(src, BLOCK, LAYOUT)
    return values * 2.0 + 3.0


@triton.jit
def _set_layout_noinline_abi_kernel(src, out, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    values = _set_layout_bridge(src, BLOCK, LAYOUT)
    offsets = tl.arange(0, BLOCK)
    pointers = tle.gpu.set_layout(out + offsets, LAYOUT)
    tl.store(pointers, values)


@triton.jit
def _set_layout_tile_roundtrip_kernel(
    src,
    out,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    tile_index = tl.program_id(0)
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    offsets = rows * BLOCK + cols
    source = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)
    tile = tle.extract_tile(source, index=tile_index, tile_shape=(TILE, TILE))
    tile = tle.gpu.set_layout(tile, LAYOUT)
    tile = tile + (tile_index + 1).to(tl.float32) * 1000.0
    result = tle.insert_tile(source, tile, index=tile_index)
    result = tle.gpu.set_layout(result, LAYOUT)
    output_offsets = tile_index * BLOCK * BLOCK + offsets
    pointers = tle.gpu.set_layout(out + output_offsets, LAYOUT)
    tl.store(pointers, result)


@triton.jit
def _set_layout_shared_roundtrip_kernel(src, out, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)
    buffer = tle.gpu.alloc(
        (BLOCK, ),
        dtype=tl.float32,
        init_value=values,
        nv_mma_shared_layout=False,
    )
    tl.debug_barrier()
    local_pointers = tle.gpu.local_ptr(buffer, (offsets, ))
    result = tle.gpu.set_layout(tl.load(local_pointers), LAYOUT)
    output_pointers = tle.gpu.set_layout(out + offsets, LAYOUT)
    tl.store(output_pointers, result * 2.0 - 4.0)


@triton.jit
def _set_layout_copy_roundtrip_kernel(src, out, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    buffer = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(src + offsets, buffer, (BLOCK, ))
    local_pointers = tle.gpu.local_ptr(buffer, (offsets, ))
    result = tle.gpu.set_layout(tl.load(local_pointers), LAYOUT)
    output_pointers = tle.gpu.set_layout(out + offsets, LAYOUT)
    tl.store(output_pointers, result + 7.0)


@triton.jit
def _set_layout_wmma_kernel(
    out,
    MMA_LAYOUT: tl.constexpr,
    LHS_LAYOUT: tl.constexpr,
    RHS_LAYOUT: tl.constexpr,
    CHAINED: tl.constexpr,
    FP16_RESULT: tl.constexpr,
):
    size: tl.constexpr = 16
    rows = tl.arange(0, size)[:, None]
    cols = tl.arange(0, size)[None, :]
    lhs = tle.gpu.set_layout(tl.full((size, size), 1.0, tl.bfloat16), LHS_LAYOUT)
    rhs = tle.gpu.set_layout(tl.full((size, size), 2.0, tl.bfloat16), RHS_LAYOUT)
    acc = tle.gpu.set_layout(tl.zeros((size, size), tl.float32), MMA_LAYOUT)
    result = tl.dot(lhs, rhs, acc=acc, out_dtype=tl.float32)
    if CHAINED:
        result = tle.gpu.set_layout(result, MMA_LAYOUT)
        result = tl.dot(lhs, rhs, acc=result, out_dtype=tl.float32)
    result = tle.gpu.set_layout(result, MMA_LAYOUT)
    if FP16_RESULT:
        result = result.to(tl.float16)
    pointers = tle.gpu.set_layout(out + rows * size + cols, MMA_LAYOUT)
    tl.store(pointers, result)


@triton.jit
def _set_layout_sqmma_dot_kernel(
    out,
    MMA_LAYOUT: tl.constexpr,
    CHAINED: tl.constexpr,
    FP16_RESULT: tl.constexpr,
):
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = 64
    block_k: tl.constexpr = 32
    rows = tl.arange(0, block_m)[:, None]
    cols = tl.arange(0, block_n)[None, :]
    lhs = tl.full((block_m, block_k), 1.0, tl.float16)
    rhs = tl.full((block_k, block_n), 2.0, tl.float16)
    acc = tle.gpu.set_layout(tl.zeros((block_m, block_n), tl.float32), MMA_LAYOUT)
    result = tl.dot(lhs, rhs, acc=acc, out_dtype=tl.float32)
    if CHAINED:
        result = tle.gpu.set_layout(result, MMA_LAYOUT)
        result = tl.dot(lhs, rhs, acc=result, out_dtype=tl.float32)
    result = tle.gpu.set_layout(result, MMA_LAYOUT)
    if FP16_RESULT:
        result = result.to(tl.float16)
    pointers = tle.gpu.set_layout(out + rows * block_n + cols, MMA_LAYOUT)
    tl.store(pointers, result)


@triton.jit
def _set_layout_wgmma_kernel(a_desc, b_desc, out, MMA_LAYOUT: tl.constexpr):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    lhs = tle.gpu.alloc(
        (block_m, block_k),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    rhs = tle.gpu.alloc(
        (block_k, block_n),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    lhs_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * 2)
    rhs_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * 2)
    tle.gpu.copy(a_desc, lhs, (block_m, block_k), (0, 0), barrier=lhs_full)
    tle.gpu.copy(b_desc, rhs, (block_k, block_n), (0, 0), barrier=rhs_full)
    tle.gpu.barrier_wait(lhs_full, phaseIdx=0)
    tle.gpu.barrier_wait(rhs_full, phaseIdx=0)
    result = tle.gpu.wgmma(lhs, rhs, tl.zeros((block_m, block_n), dtype=tl.float32))
    result = tle.gpu.wgmma_wait(0, result)
    result = tle.gpu.set_layout(result, MMA_LAYOUT)
    rows = tl.arange(0, block_m)[:, None]
    cols = tl.arange(0, block_n)[None, :]
    pointers = tle.gpu.set_layout(out + rows * block_n + cols, MMA_LAYOUT)
    tl.store(pointers, result)


@triton.jit
def _set_layout_invalid_target_kernel(out, LAYOUT: tl.constexpr):
    values = tle.gpu.set_layout(tl.arange(0, 128), LAYOUT)
    tl.store(out + tl.arange(0, 128), values)


def test_mthreads_layout_types_are_public_and_ranked():
    blocked = _block_layout_a()
    sliced = tle.gpu.SlicedEncoding(0, blocked)
    wmma = _wmma_layout()
    sqmma = _sqmma_layout()
    lhs = MusaDotOperandEncoding(0, wmma)
    rhs = MusaDotOperandEncoding(1, wmma)

    assert blocked.rank == 2
    assert sliced.rank == 1
    assert wmma.rank == 2
    assert sqmma.rank == 2
    assert lhs.rank == rhs.rank == 2
    assert tle.gpu.mthreads.MusaWmmaEncoding is MusaWmmaEncoding
    assert tle.gpu.mthreads.MusaSqmmaEncoding is MusaSqmmaEncoding
    assert tle.gpu.mthreads.MusaDotOperandEncoding is MusaDotOperandEncoding


@pytest.mark.parametrize("operand_index", [0, 1])
def test_mthreads_sqmma_register_operand_layout_is_rejected(operand_index):
    with pytest.raises(ValueError) as excinfo:
        MusaDotOperandEncoding(operand_index, _sqmma_layout())
    diagnostic = str(excinfo.value)
    assert "SQMMA" in diagnostic
    assert "shared memory" in diagnostic


@requires_musa_runtime
@pytest.mark.parametrize("n", [127, 128])
def test_set_layout_native_pointwise_runtime(n):
    block = 128
    reference = torch.linspace(-5.0, 7.0, n, dtype=torch.float32)
    source = reference.to("musa")
    actual = torch.full((n, ), float("nan"), device="musa", dtype=torch.float32)

    _set_layout_pointwise_kernel[(1, )](source, actual, n=n, BLOCK=block, LAYOUT=_slice_layout(), num_warps=4)

    offsets = torch.arange(n)
    expected = reference * 1.5 + torch.where(offsets % 3 == 0, 2.0, -1.0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=1e-6)


@requires_musa_runtime
def test_set_layout_native_transpose_runtime():
    block = 16
    reference = torch.arange(block * block, dtype=torch.float32).reshape(block, block) * 0.25 - 4.0
    source = reference.to("musa")
    actual = torch.full_like(source, float("nan"))

    _set_layout_transpose_kernel[(1, )](
        source,
        actual,
        BLOCK=block,
        SRC_LAYOUT=_block_layout_a(),
        DST_LAYOUT=_block_layout_b(),
        num_warps=4,
    )

    torch.testing.assert_close(actual.cpu(), reference.T * 2.0 + 1.0, rtol=0, atol=0)


@requires_musa_runtime
def test_set_layout_multiple_domains_runtime():
    block = 16
    reference = torch.arange(block * block, dtype=torch.float32).reshape(block, block) / 7.0
    source = reference.to("musa")
    actual_a = torch.full_like(source, float("nan"))
    actual_b = torch.full_like(source, float("nan"))

    _set_layout_multi_domain_kernel[(1, )](
        source,
        actual_a,
        actual_b,
        BLOCK=block,
        LAYOUT_A=_block_layout_a(),
        LAYOUT_B=_block_layout_b(),
        num_warps=4,
    )

    torch.testing.assert_close(actual_a.cpu(), reference + 1.25, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual_b.cpu(), reference * 3.0 - 2.0, rtol=0, atol=1e-6)


@requires_musa_runtime
def test_set_layout_shared_expand_dims_runtime():
    block = 128
    row_layout = tle.gpu.BlockEncoding([1, 1, 1], [1, 32, 1], [1, 4, 1], [2, 1, 0])
    col_layout = tle.gpu.BlockEncoding([1, 1, 1], [1, 1, 32], [1, 1, 4], [2, 1, 0])
    actual_rows = torch.full((block, ), float("nan"), device="musa", dtype=torch.float32)
    actual_cols = torch.full_like(actual_rows, float("nan"))

    _set_layout_expand_dims_kernel[(1, )](
        actual_rows,
        actual_cols,
        BLOCK=block,
        ROW_LAYOUT=row_layout,
        COL_LAYOUT=col_layout,
        num_warps=4,
    )

    values = torch.arange(block, dtype=torch.float32)
    torch.testing.assert_close(actual_rows.cpu(), values + 0.5, rtol=0, atol=0)
    torch.testing.assert_close(actual_cols.cpu(), values * 2.0, rtol=0, atol=0)


@requires_musa_runtime
def test_set_layout_noinline_abi_runtime():
    block = 128
    reference = torch.arange(block, dtype=torch.float32) / 3.0
    source = reference.to("musa")
    actual = torch.full_like(source, float("nan"))

    _set_layout_noinline_abi_kernel[(1, )](
        source,
        actual,
        BLOCK=block,
        LAYOUT=_slice_layout(),
        num_warps=4,
    )

    torch.testing.assert_close(actual.cpu(), reference * 2.0 + 3.0, rtol=0, atol=1e-5)


@requires_musa_runtime
def test_set_layout_extract_insert_tile_runtime():
    block = 32
    tile = 16
    reference = torch.arange(block * block, dtype=torch.float32).reshape(block, block)
    source = reference.to("musa")
    actual = torch.full((4, block, block), float("nan"), device="musa", dtype=torch.float32)

    _set_layout_tile_roundtrip_kernel[(4, )](
        source,
        actual,
        BLOCK=block,
        TILE=tile,
        LAYOUT=_block_layout_a(),
        num_warps=4,
    )

    expected = reference.expand(4, block, block).clone()
    for index in range(4):
        row = (index // 2) * tile
        col = (index % 2) * tile
        expected[index, row:row + tile, col:col + tile] += (index + 1) * 1000.0
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=1e-6)


@requires_musa_runtime
@pytest.mark.parametrize("kind", ["alloc-local-ptr", "copy"])
def test_set_layout_shared_memory_runtime(kind):
    block = 128
    reference = torch.arange(block, dtype=torch.float32) * 0.5 - 9.0
    source = reference.to("musa")
    actual = torch.full_like(source, float("nan"))
    if kind == "alloc-local-ptr":
        kernel = _set_layout_shared_roundtrip_kernel
        expected = reference * 2.0 - 4.0
    else:
        kernel = _set_layout_copy_roundtrip_kernel
        expected = reference + 7.0

    kernel[(1, )](source, actual, BLOCK=block, LAYOUT=_slice_layout(), num_warps=4)

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@requires_musa_runtime
@pytest.mark.parametrize(
    "kind,chained,result_dtype",
    [
        ("wmma", False, torch.float32),
        ("wmma", False, torch.float16),
        ("wmma", True, torch.float32),
        ("sqmma", False, torch.float32),
        ("sqmma", False, torch.float16),
        ("sqmma", True, torch.float32),
    ],
)
def test_set_layout_native_dot_runtime(kind, chained, result_dtype):
    if kind == "wmma":
        block_m = block_n = block_k = 16
        mma = _wmma_layout()
        kernel = _set_layout_wmma_kernel
        layout_args = (
            mma,
            MusaDotOperandEncoding(0, mma),
            MusaDotOperandEncoding(1, mma),
        )
    else:
        block_m, block_n, block_k = 64, 64, 32
        mma = _sqmma_layout()
        kernel = _set_layout_sqmma_dot_kernel
        layout_args = (mma, )

    actual = torch.full((block_m, block_n), float("nan"), device="musa", dtype=result_dtype)
    kernel[(1, )](
        actual,
        *layout_args,
        CHAINED=chained,
        FP16_RESULT=result_dtype == torch.float16,
        num_warps=4,
    )

    expected_value = 4.0 * block_k if chained else 2.0 * block_k
    expected = torch.full((block_m, block_n), expected_value, dtype=result_dtype)
    atol = 0.5 if result_dtype == torch.float16 else 0.1
    rtol = 0.05
    torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)


@requires_musa_runtime
def test_set_layout_wgmma_runtime():
    from triton.tools.tensor_descriptor import TensorDescriptor

    torch.manual_seed(23)
    block_m = block_n = 128
    block_k = 64
    lhs_cpu = torch.randn((block_m, block_k), dtype=torch.float16)
    rhs_cpu = torch.randn((block_k, block_n), dtype=torch.float16)
    lhs = lhs_cpu.to("musa")
    rhs = rhs_cpu.to("musa")
    actual = torch.full((block_m, block_n), float("nan"), device="musa", dtype=torch.float32)
    lhs_desc = TensorDescriptor.from_tensor(lhs, block_shape=[block_m, block_k])
    rhs_desc = TensorDescriptor.from_tensor(rhs, block_shape=[block_k, block_n])
    mma = MusaSqmmaEncoding([3, 1], [4, 1], [block_m, block_n, block_k])

    _set_layout_wgmma_kernel[(1, )](
        lhs_desc,
        rhs_desc,
        actual,
        MMA_LAYOUT=mma,
        num_warps=4,
        num_stages=1,
    )

    expected = lhs_cpu.float() @ rhs_cpu.float()
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)


@requires_musa_runtime
@pytest.mark.parametrize("target", ["rank-mismatch", "shared-layout"])
def test_set_layout_rejects_invalid_target_runtime(target):
    out = torch.empty((128, ), device="musa", dtype=torch.int32)
    if target == "rank-mismatch":
        layout = _block_layout_a()
    else:
        layout = tle.gpu.swizzled_shared_layout.make_default(rank=1)

    with pytest.raises((CompilationError, RuntimeError, ValueError)) as excinfo:
        _set_layout_invalid_target_kernel[(1, )](out, LAYOUT=layout, num_warps=4)

    diagnostic = str(excinfo.value).lower()
    if target == "rank-mismatch":
        assert "rank" in diagnostic or "encoding" in diagnostic
    else:
        assert "distributed" in diagnostic or "shared" in diagnostic
