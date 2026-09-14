"""Unit coverage for explicit TLE transport-level asynchronous copies."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _require_cuda():
    try:
        torch.cuda.init()
    except Exception as exc:
        pytest.skip(f"CUDA init failed: {exc}")


@pytest.fixture(scope="module", autouse=True)
def _cuda_guard():
    _require_cuda()


@triton.jit
def _two_group_async_copy_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    stage0 = tle.gpu.alloc(
        [BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False)
    stage1 = tle.gpu.alloc(
        [BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False)

    tle.gpu.copy(in_ptr + offsets, stage0, [BLOCK], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.copy(in_ptr + BLOCK + offsets, stage1, [BLOCK], is_async=True)
    tle.gpu.async_commit_group()

    tle.gpu.async_wait_group(1)
    values0 = tl.load(tle.gpu.local_ptr(stage0, (offsets, )))
    tl.store(out_ptr + offsets, values0 + 1.0)
    tle.gpu.async_wait_group(0)
    values1 = tl.load(tle.gpu.local_ptr(stage1, (offsets, )))
    tl.store(out_ptr + BLOCK + offsets, values1 + 2.0)


@triton.jit
def _qkv_lhs_shape_async_copy_kernel(in_ptr, out_ptr):
    row = tl.arange(0, 1)[:, None]
    col = tl.arange(0, 256)[None, :]
    offsets = row * 256 + col
    smem = tle.gpu.alloc(
        [1, 256], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=True)
    tle.gpu.copy(
        in_ptr + offsets, smem, [1, 256], is_async=True,
        eviction_policy="evict_last")
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    values = tl.load(tle.gpu.local_ptr(smem, (row, col), shape=(1, 256)))
    tl.store(out_ptr + offsets, values)


@triton.jit
def _invalid_async_direction_kernel(out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc(
        [BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False)
    tle.gpu.copy(smem, out_ptr + offsets, [BLOCK], is_async=True)


@triton.jit
def _invalid_async_shape_kernel(in_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc(
        [BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False)
    tle.gpu.copy(in_ptr + offsets, smem, [BLOCK // 2], is_async=True)


@triton.jit
def _invalid_async_wait_kernel(out_ptr, MAX_PENDING: tl.constexpr):
    tle.gpu.async_wait_group(MAX_PENDING)
    tl.store(out_ptr, 0.0)


def test_async_copy_api_is_exported():
    assert hasattr(tle.gpu, "async_commit_group")
    assert hasattr(tle.gpu, "async_wait_group")


def test_two_group_async_copy_waits_conservatively_and_executes():
    block = 64
    inp = torch.arange(2 * block, device="cuda", dtype=torch.float32)
    out = torch.empty_like(inp)
    compiled = _two_group_async_copy_kernel.warmup(
        inp, out, BLOCK=block, grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]
    assert ttgir.count("ttg.async_copy_global_to_local") == 2
    assert ttgir.count("tle.required_async_copy") == 2
    assert ttgir.count("ttg.async_commit_group") == 2
    # The ordinary pipeline derives waits from dependency tokens. The TLE
    # transport API has none, so a requested partial wait may become a drain.
    wait_counts = re.findall(r"ttg\.async_wait[^\n]*num = (\d+)", ttgir)
    assert wait_counts and set(wait_counts) == {"0"}
    assert "tle.explicit_async_wait" not in ttgir
    assert re.search(r"cp\.async\.wait_group\s+0", ptx)
    assert "cp.async" in ptx

    _two_group_async_copy_kernel[(1, )](inp, out, BLOCK=block, num_warps=4)
    expected = torch.cat((inp[:block] + 1.0, inp[block:] + 2.0))
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_qkv_bf16_1x256_copy_legalizes_and_executes():
    inp = torch.arange(256, device="cuda", dtype=torch.bfloat16).reshape(1, 256)
    out = torch.empty_like(inp)
    compiled = _qkv_lhs_shape_async_copy_kernel.warmup(
        inp, out, grid=(1, ), num_warps=8)
    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]
    assert "ttg.async_copy_global_to_local" in ttgir
    assert "tle.required_async_copy" in ttgir
    assert "contiguity = 2 : i32" in ttgir
    assert re.search(r"cp\.async\.wait_group\s+0", ptx)

    _qkv_lhs_shape_async_copy_kernel[(1, )](inp, out, num_warps=8)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_async_copy_rejects_non_global_to_shared_direction():
    out = torch.empty((64, ), device="cuda", dtype=torch.float32)
    with pytest.raises(triton.CompilationError, match="supports only a tl.tensor of global pointers"):
        _invalid_async_direction_kernel.warmup(out, BLOCK=64, grid=(1, ), num_warps=4)


def test_async_copy_rejects_partial_shape():
    inp = torch.empty((64, ), device="cuda", dtype=torch.float32)
    with pytest.raises(triton.CompilationError, match="requires shape to exactly match both operands"):
        _invalid_async_shape_kernel.warmup(inp, BLOCK=64, grid=(1, ), num_warps=4)


@triton.jit
def _masked_async_copy_kernel(source, result, ROWS: tl.constexpr, COLS: tl.constexpr,
                              TILE_ROWS: tl.constexpr, TILE_COLS: tl.constexpr):
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)[:, None]
    col = tl.arange(0, TILE_COLS)[None, :]
    stage = tle.gpu.alloc([TILE_ROWS, TILE_COLS], dtype=tl.float32,
                          layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    tle.gpu.copy(source + row * COLS + col, stage, [TILE_ROWS, TILE_COLS],
                 is_async=True, mask=(row < ROWS) & (col < COLS))
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    values = tl.load(tle.gpu.local_ptr(stage, (tl.arange(0, TILE_ROWS)[:, None], col),
                                      [TILE_ROWS, TILE_COLS]))
    tl.store(result + row * TILE_COLS + col, values)


@pytest.mark.parametrize("rows,columns", [(7, 12), (0, 12), (8, 128)])
def test_masked_async_copy_zero_fills_invalid_rows_and_columns(rows, columns):
    source = torch.arange(rows * columns, device="cuda", dtype=torch.float32).reshape(rows, columns)
    output = torch.full((12, 128), float("nan"), device="cuda")
    kernel = _masked_async_copy_kernel[(3,)](source, output, rows, columns, 4, 128, num_warps=4)
    expected = torch.zeros_like(output)
    expected[:rows, :columns] = source
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    if rows:
        assert "ttg.async_copy_global_to_local" in kernel.asm["ttgir"]
        assert "cp.async" in kernel.asm["ptx"]


@pytest.mark.parametrize("max_pending", [-1, 8])
def test_async_wait_group_rejects_out_of_range_immediates(max_pending):
    out = torch.empty((1, ), device="cuda", dtype=torch.float32)
    with pytest.raises(triton.CompilationError, match=r"max_pending must be in \[0, 7\]"):
        _invalid_async_wait_kernel.warmup(
            out, MAX_PENDING=max_pending, grid=(1, ), num_warps=1)


@triton.jit
def _copy_mask_contract_kernel(source, result, MODE: tl.constexpr):
    row = tl.arange(0, 4)[:, None]
    col = tl.arange(0, 32)[None, :]
    stage = tle.gpu.alloc([4, 32], dtype=tl.float32, layout=None,
                          scope=tle.gpu.smem, nv_mma_shared_layout=False)
    if MODE == "integer":
        mask = row
    elif MODE == "bad_shape":
        mask = tl.arange(0, 8)[:, None] < 2
    elif MODE == "scalar":
        mask = False
    else:
        mask = row < 2
    if MODE == "reverse":
        tle.gpu.copy(stage, source + row * 32 + col, [4, 32], is_async=True, mask=mask)
    else:
        tle.gpu.copy(source + row * 32 + col, stage, [4, 32], is_async=MODE != "sync", mask=mask)
        tle.gpu.async_commit_group()
        tle.gpu.async_wait_group(0)
        values = tl.load(tle.gpu.local_ptr(stage, (row, col), [4, 32]))
        tl.store(result + row * 32 + col, values)


@pytest.mark.parametrize("mode,match", [
    ("integer", "mask must have boolean"), ("sync", "mask requires is_async=True"),
    ("reverse", "supports only a tl.tensor of global pointers"), ("bad_shape", "broadcast"),
])
def test_copy_mask_rejects_invalid_contract(mode, match):
    source = torch.empty((4, 32), device="cuda")
    with pytest.raises(triton.CompilationError, match=match):
        _copy_mask_contract_kernel.warmup(source, torch.empty_like(source), mode, grid=(1,), num_warps=4)


@pytest.mark.parametrize("mode", ["row", "scalar"])
def test_copy_mask_broadcasts_boolean_scalar_and_row(mode):
    source = torch.arange(128, dtype=torch.float32, device="cuda").reshape(4, 32)
    output = torch.full_like(source, float("nan"))
    _copy_mask_contract_kernel[(1,)](source, output, mode, num_warps=4)
    expected = torch.zeros_like(source)
    if mode == "row":
        expected[:2] = source[:2]
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
