# flagtree tle
"""
Unit tests for TLE cumsum helper.
"""

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
def _tle_cumsum_masked_kernel(x_ptr, exclusive_ptr, total_ptr, n, BLOCK: tl.constexpr, REVERSE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    exclusive, total = tle.cumsum(x, axis=0, reverse=REVERSE)
    tl.store(exclusive_ptr + offs, exclusive, mask=mask)
    tl.store(total_ptr, total)


@triton.jit
def _tle_cumsum_ptx_kernel(x_ptr, exclusive_ptr, total_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    exclusive, total = tle.cumsum(x, axis=0, reverse=False)
    tl.store(exclusive_ptr + offs, exclusive)
    tl.store(total_ptr, total)


@triton.jit
def _tle_cumsum_callee_shared_kernel(hist_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(hist_ptr + offs)
    exclusive, _ = tle.cumsum(x, axis=0, reverse=False)
    tl.store(hist_ptr + offs, exclusive)


@triton.jit
def _tle_cumsum_call_shared_kernel(exclusive_ptr, sentinel_ptr, BLOCK: tl.constexpr):
    sentinel_value = 123456789
    offs = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc([BLOCK * 2], dtype=tl.int32, scope=tle.gpu.smem)
    base = tle.gpu.local_ptr(smem, (0, ))

    tl.store(base + offs, offs + 1)
    tl.store(base + (BLOCK + offs), sentinel_value)
    _tle_cumsum_callee_shared_kernel(base, BLOCK=BLOCK)
    tl.debug_barrier()

    exclusive = tl.load(base + offs)
    sentinel = tl.load(base + (BLOCK + offs))
    tl.store(exclusive_ptr + offs, exclusive)
    tl.store(sentinel_ptr + offs, sentinel)


@triton.jit
def _tle_cumsum_scalar_base_addptr_kernel(exclusive_ptr, sentinel_ptr, BLOCK: tl.constexpr):
    sentinel_value = 123456789
    offs = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc([BLOCK * 2], dtype=tl.int32, scope=tle.gpu.smem)
    base = tle.gpu.local_ptr(smem, (0, ))
    data_ptrs = base + offs
    sentinel_ptrs = base + (BLOCK + offs)

    tl.store(data_ptrs, offs + 1)
    tl.store(sentinel_ptrs, sentinel_value)
    x = tl.load(data_ptrs)
    exclusive, _ = tle.cumsum(x, axis=0, reverse=False)
    tl.store(data_ptrs, exclusive)
    tl.debug_barrier()

    tl.store(exclusive_ptr + offs, tl.load(data_ptrs))
    tl.store(sentinel_ptr + offs, tl.load(sentinel_ptrs))


def _pick_expected_dtype(input_dtype: torch.dtype) -> torch.dtype:
    if input_dtype in (torch.int8, torch.int16):
        return torch.int32
    if input_dtype == torch.bfloat16:
        return torch.float32
    return input_dtype


def _make_input(dtype: torch.dtype, block: int) -> torch.Tensor:
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        return torch.randn((block, ), device="cuda", dtype=dtype)
    if dtype == torch.int8:
        return torch.randint(-32, 32, (block, ), device="cuda", dtype=dtype)
    if dtype == torch.int16:
        return torch.randint(-512, 512, (block, ), device="cuda", dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(-2048, 2048, (block, ), device="cuda", dtype=dtype)
    raise AssertionError(f"unsupported dtype for test: {dtype}")


def _entry_block(ptx: str) -> str:
    m = re.search(r"\.entry\s+([^\(]+)\(", ptx)
    assert m is not None, "failed to locate PTX entry"
    begin = ptx.find("{", m.end())
    assert begin >= 0, "failed to locate PTX entry body"
    depth = 0
    for i in range(begin, len(ptx)):
        if ptx[i] == "{":
            depth += 1
        elif ptx[i] == "}":
            depth -= 1
            if depth == 0:
                return ptx[m.start():i + 1]
    raise AssertionError("failed to parse PTX entry block")


@pytest.mark.parametrize(
    "dtype, n, block, reverse, num_warps",
    [
        (torch.int8, 511, 512, False, 16),
        (torch.int16, 257, 512, False, 16),
        (torch.int32, 512, 512, False, 16),
        (torch.int32, 256, 256, True, 8),
        (torch.int32, 512, 512, True, 16),
        (torch.float16, 127, 128, False, 4),
        (torch.float32, 128, 128, True, 4),
        (torch.float32, 512, 512, True, 16),
        (torch.bfloat16, 193, 256, False, 8),
    ],
)
def test_tle_cumsum_exclusive_and_total(dtype, n, block, reverse, num_warps):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this GPU")

    x = _make_input(dtype, block)
    out_dtype = _pick_expected_dtype(dtype)
    exclusive = torch.empty((block, ), device="cuda", dtype=out_dtype)
    total = torch.empty((1, ), device="cuda", dtype=out_dtype)

    _tle_cumsum_masked_kernel[(1, )](
        x,
        exclusive,
        total,
        n,
        BLOCK=block,
        REVERSE=reverse,
        num_warps=num_warps,
    )

    x_valid = x[:n].to(out_dtype)
    if reverse:
        expected_exclusive = torch.flip(torch.cumsum(torch.flip(x_valid, dims=[0]), dim=0, dtype=out_dtype),
                                        dims=[0]) - x_valid
    else:
        expected_exclusive = torch.cumsum(x_valid, dim=0, dtype=out_dtype) - x_valid
    expected_total = torch.sum(x_valid, dim=0, dtype=out_dtype)

    if out_dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(exclusive[:n], expected_exclusive, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(total[0], expected_total, atol=2e-2, rtol=2e-2)
    elif out_dtype == torch.float32:
        # GPU parallel scan accumulation order differs from torch's sequential
        # cumsum reference, especially in reverse mode.
        atol = 1e-5 if reverse else 2e-6
        rtol = 5e-4 if reverse else 1e-5
        torch.testing.assert_close(exclusive[:n], expected_exclusive, atol=atol, rtol=rtol)
        torch.testing.assert_close(total[0], expected_total, atol=2e-6, rtol=1e-5)
    else:
        torch.testing.assert_close(exclusive[:n], expected_exclusive)
        torch.testing.assert_close(total[0], expected_total)


def test_tle_cumsum_ptx_fastpath_regression_guard():
    block = 512
    x = torch.randint(-1024, 1024, (block, ), device="cuda", dtype=torch.int32)
    exclusive = torch.empty_like(x)
    total = torch.empty((1, ), device="cuda", dtype=torch.int32)

    compiled = _tle_cumsum_ptx_kernel.warmup(
        x,
        exclusive,
        total,
        BLOCK=block,
        grid=(1, ),
        num_warps=block // 32,
        num_stages=1,
    )

    ttgir = compiled.asm["ttgir"]
    assert "tle.exclusive_cumsum" in ttgir
    assert "\"tt.scan\"" not in ttgir
    assert ttgir.count("ttg.convert_layout") == 0

    ptx = _entry_block(compiled.asm["ptx"])
    assert len(re.findall(r"\bbar\.sync\b", ptx)) == 2
    # TRT/CUB-aligned lowering keeps a single 32-lane scan in the round path.
    assert len(re.findall(r"\bshfl\.sync\.up\b", ptx)) == 5
    assert len(re.findall(r"\bshfl\.sync\.idx\b", ptx)) == 1
    assert len(re.findall(r"\bshfl\.sync\b", ptx)) <= 6
    assert len(re.findall(r"@%p\d+\s+ld\.shared", ptx)) == 0
    assert len(re.findall(r"@%p\d+\s+st\.shared", ptx)) == 0
    assert len(re.findall(r"\bselp\b", ptx)) == 0


def test_tle_cumsum_call_shared_frame_regression():
    block = 512
    num_warps = block // 32
    exclusive = torch.empty((block, ), device="cuda", dtype=torch.int32)
    sentinel = torch.empty((block, ), device="cuda", dtype=torch.int32)

    compiled = _tle_cumsum_call_shared_kernel.warmup(
        exclusive,
        sentinel,
        BLOCK=block,
        grid=(1, ),
        num_warps=num_warps,
        num_stages=1,
    )
    ttgir = compiled.asm["ttgir"]
    assert "tt.call" in ttgir, "regression scenario requires cross-function call frame"

    _tle_cumsum_call_shared_kernel[(1, )](
        exclusive,
        sentinel,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )

    x = torch.arange(1, block + 1, device="cuda", dtype=torch.int32)
    expected_exclusive = torch.cumsum(x, dim=0, dtype=torch.int32) - x
    expected_sentinel = torch.full((block, ), 123456789, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(exclusive, expected_exclusive)
    torch.testing.assert_close(sentinel, expected_sentinel)


def test_tle_cumsum_scalar_base_addptr_alias_regression():
    block = 512
    num_warps = block // 32
    exclusive = torch.empty((block, ), device="cuda", dtype=torch.int32)
    sentinel = torch.empty((block, ), device="cuda", dtype=torch.int32)

    _tle_cumsum_scalar_base_addptr_kernel[(1, )](
        exclusive,
        sentinel,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )

    x = torch.arange(1, block + 1, device="cuda", dtype=torch.int32)
    expected_exclusive = torch.cumsum(x, dim=0, dtype=torch.int32) - x
    expected_sentinel = torch.full((block, ), 123456789, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(exclusive, expected_exclusive)
    torch.testing.assert_close(sentinel, expected_sentinel)
