"""
TLE tle.cumsum test (thrive), adapted from python/test/tle/unit/test_tle_cumsum.py
========================================================================================================

Coverage ported from the upstream TLE unit test:
  - 9 (dtype, n, BLOCK, reverse) combinations, including dtypes the frontend
    widens (int8/int16 -> int32) or promotes (bfloat16 -> float32)
  - runtime (non-constexpr) ``n`` with masked load/store
  - unaligned n/BLOCK boundaries (511/512, 257/512, 127/128, 193/256)
  - int32 fast-path checks

Differences from the upstream TLE test:
  - ``device`` comes from the thrive test-suite fixture instead of ``cuda``
  - ``num_cores`` is an extra parameterization axis (1 and 2)
  - the two IR/ISA regression guards (``..._ptx_fastpath_regression_guard``,
    ``..._amdgcn_fastpath_regression_guard``) are not ported
  - the shared-memory kernels (``..._callee_shared_kernel``,
    ``..._helper_shared_kernel``, ``..._scalar_base_addptr_kernel``) are not
    ported: they need ``tle.gpu.alloc`` / ``tle.gpu.local_ptr`` /
    ``tl.debug_barrier``, and thrive's TleToThvTile conversion has no
    local-alloc lowering

reference: exclusive[i] = sum(x[:i]); total = sum(x).
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _tle_cumsum_masked_kernel(x_ptr, exclusive_ptr, total_ptr, n, BLOCK: tl.constexpr, REVERSE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    exclusive, total = tle.cumsum(x, axis=0, reverse=REVERSE)
    tl.store(exclusive_ptr + offs, exclusive, mask=mask)
    tl.store(total_ptr, total)


def _pick_expected_dtype(input_dtype: torch.dtype) -> torch.dtype:
    if input_dtype in (torch.int8, torch.int16):
        return torch.int32
    if input_dtype == torch.bfloat16:
        return torch.float32
    return input_dtype


def _make_input(dtype: torch.dtype, block: int) -> torch.Tensor:
    # Built on CPU and moved to the device afterwards, like test_cumsum.py.
    if dtype == torch.float16:
        # Keep every prefix sum exactly representable in fp16 so the
        # correctness test is independent of parallel scan accumulation order.
        return torch.randint(-2, 3, (block, ), dtype=torch.int32).to(dtype)
    if dtype in (torch.float32, torch.bfloat16):
        return torch.randn((block, ), dtype=dtype)
    if dtype == torch.int8:
        return torch.randint(-32, 32, (block, ), dtype=dtype)
    if dtype == torch.int16:
        return torch.randint(-512, 512, (block, ), dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(-2048, 2048, (block, ), dtype=dtype)
    raise AssertionError(f"unsupported dtype for test: {dtype}")


def _ref_exclusive(x, reverse, out_dtype):
    if reverse:
        return torch.flip(torch.cumsum(torch.flip(x, dims=[0]), dim=0, dtype=out_dtype), dims=[0]) - x
    return torch.cumsum(x, dim=0, dtype=out_dtype) - x


def _check(exclusive, total, ref_excl, ref_total, n, reverse, out_dtype):
    if out_dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(exclusive[:n], ref_excl, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(total[0], ref_total, atol=2e-2, rtol=2e-2)
    elif out_dtype == torch.float32:
        # Parallel scan accumulation order differs from torch's sequential
        # cumsum reference, especially in reverse mode.
        atol = 1e-5 if reverse else 2e-6
        rtol = 5e-4 if reverse else 1e-5
        torch.testing.assert_close(exclusive[:n], ref_excl, atol=atol, rtol=rtol)
        torch.testing.assert_close(total[0], ref_total, atol=2e-6, rtol=1e-5)
    else:
        torch.testing.assert_close(exclusive[:n], ref_excl)
        torch.testing.assert_close(total[0], ref_total)


@pytest.mark.parametrize(
    "dtype, n, block, reverse",
    [
        (torch.int8, 511, 512, False),
        (torch.int16, 257, 512, False),
        (torch.int32, 512, 512, False),
        (torch.int32, 256, 256, True),
        (torch.int32, 512, 512, True),
        (torch.float16, 127, 128, False),
        (torch.float32, 128, 128, True),
        (torch.float32, 512, 512, True),
        (torch.bfloat16, 193, 256, False),
    ],
)
@pytest.mark.parametrize("num_cores", [1, 2])
def test_tle_cumsum_exclusive_and_total(dtype, n, block, reverse, num_cores, device):
    torch.manual_seed(0)
    out_dtype = _pick_expected_dtype(dtype)
    x = _make_input(dtype, block).to(device)
    exclusive = torch.empty((block, ), device=device, dtype=out_dtype)
    total = torch.empty((1, ), device=device, dtype=out_dtype)

    _tle_cumsum_masked_kernel[(1, )](
        x,
        exclusive,
        total,
        n,
        BLOCK=block,
        REVERSE=reverse,
        num_cores=num_cores,
    )

    exclusive = exclusive.cpu()
    total = total.cpu()
    x_valid = x[:n].cpu().to(out_dtype)
    _check(exclusive, total, _ref_exclusive(x_valid, reverse, out_dtype), torch.sum(x_valid, dim=0, dtype=out_dtype), n,
           reverse, out_dtype)
