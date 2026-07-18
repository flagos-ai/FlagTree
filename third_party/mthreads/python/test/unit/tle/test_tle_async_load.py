import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _tle_load_asm_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr, IS_ASYNC: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = tle.load(x_ptr + offs, is_async=IS_ASYNC)
    tl.store(out_ptr + offs, vals)


@triton.jit
def _tle_load_hinted_asm_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    vals = tle.load(x_ptr + offs, is_async=True)
    tl.store(out_ptr + offs, vals)


@triton.jit
def _tle_load_block_ptr_asm_kernel(x_ptr, out_ptr):
    block = tl.make_block_ptr(x_ptr, shape=(64, ), strides=(1, ), offsets=(0, ), block_shape=(64, ), order=(0, ))
    vals = tle.load(block, boundary_check=(0, ), padding_option="zero", is_async=True)
    offs = tl.arange(0, 64)
    tl.store(out_ptr + offs, vals)


@triton.jit
def _tle_load_mask_other_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr, IS_ASYNC: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    vals = tle.load(x_ptr + offs, mask=mask, other=7.0, is_async=IS_ASYNC)
    tl.store(out_ptr + offs, vals)


@pytest.mark.parametrize("is_async", [False, True])
def test_tle_load_async_copy_codegen(is_async):
    compiled = compile_musa(
        _tle_load_asm_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr", "IS_ASYNC": "constexpr"},
        constexprs={"BLOCK": 64, "IS_ASYNC": is_async},
    )

    ttgir = compiled.asm["ttgir"]
    has_async_copy = "ttg.async_copy_global_to_local" in ttgir
    assert has_async_copy is is_async, ttgir
    assert "tt.load.async" not in ttgir
    if is_async:
        assert "llvm.musa.memcpy.g2s" in compiled.asm["llir"]


@pytest.mark.parametrize(
    ("signature", "block", "expect_async"),
    [
        ("*fp16", 64, False),
        ("*i8", 64, False),
        ("*fp16", 1, False),
        ("*i8", 1, False),
    ],
)
def test_tle_load_async_unsupported_widths_fall_back(signature, block, expect_async):
    compiled = compile_musa(
        _tle_load_hinted_asm_kernel,
        signature={"x_ptr": signature, "out_ptr": signature, "BLOCK": "constexpr"},
        constexprs={"BLOCK": block},
    )

    ttgir = compiled.asm["ttgir"]
    assert ("ttg.async_copy_global_to_local" in ttgir) is expect_async, ttgir
    assert "tt.load.async" not in ttgir


def test_tle_load_async_survives_block_ptr_rewrite():
    compiled = compile_musa(
        _tle_load_block_ptr_asm_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
    )

    ttgir = compiled.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" in ttgir, ttgir
    assert "tt.load.async" not in ttgir


@pytest.mark.parametrize("is_async", [False, True])
def test_tle_load_mask_other_matches_tl_load(is_async):
    block = 64
    n = 37
    x = torch.arange(n, device="musa", dtype=torch.float32)
    out = torch.empty((block, ), device="musa", dtype=torch.float32)

    _tle_load_mask_other_kernel[(1, )](x, out, n=n, BLOCK=block, IS_ASYNC=is_async, num_warps=1)

    ref = torch.full((block, ), 7.0, dtype=torch.float32)
    ref[:n] = torch.arange(n, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
