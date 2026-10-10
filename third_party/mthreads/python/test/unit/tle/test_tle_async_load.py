import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.backends import backends

from test_tle_utils import compile_musa, musa_target, require_mthreads_libtriton

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


def _aligned_ptr_attrs(signature):
    """16-byte pointer divisibility, as JIT specialization infers from torch tensors.

    Without it the coalesced layout caps per-thread vectors at one element, so
    sub-4-byte dtypes never meet the async-copy width gate."""
    return {(index, ): [["tt.divisibility", 16]]
            for index, ty in enumerate(signature.values())
            if isinstance(ty, str) and ty.startswith("*")}


@pytest.mark.parametrize(
    ("signature", "block", "expect_async"),
    [
        # The gate is the per-thread copy vector (>= 4 bytes, cp.async-style),
        # not the element dtype: narrow copies fall back regardless of dtype.
        ("*fp16", 64, False),
        ("*i8", 64, False),
        ("*fp8e4nv", 64, False),
        ("*fp16", 1, False),
        ("*i8", 1, False),
        ("*fp16", 4096, True),
        ("*i8", 4096, True),
        ("*fp8e4nv", 4096, True),
        ("*fp8e5", 4096, True),
        ("*i64", 4096, True),
        ("*fp64", 4096, True),
    ],
)
def test_tle_load_async_copy_vector_width_gate(signature, block, expect_async):
    kernel_signature = {"x_ptr": signature, "out_ptr": signature, "BLOCK": "constexpr"}
    compiled = compile_musa(
        _tle_load_hinted_asm_kernel,
        signature=kernel_signature,
        constexprs={"BLOCK": block},
        attrs=_aligned_ptr_attrs(kernel_signature),
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


_SIGNATURE_BY_DTYPE = {
    "bf16": "*bf16",
    "fp16": "*fp16",
    "fp32": "*fp32",
    "fp64": "*fp64",
    "fp8e4b15": "*fp8e4b15",
    "fp8e4b8": "*fp8e4b8",
    "fp8e4nv": "*fp8e4nv",
    "fp8e5": "*fp8e5",
    "fp8e5b16": "*fp8e5b16",
    "int8": "*i8",
    "int16": "*i16",
    "int32": "*i32",
    "int64": "*i64",
    "uint8": "*u8",
    "uint16": "*u16",
    "uint32": "*u32",
    "uint64": "*u64",
}

_DECLARED_ASYNC_DTYPES = backends["mthreads"].compiler(musa_target()).parse_options({}).async_copy_dtypes


@pytest.mark.parametrize("dtype_name", _DECLARED_ASYNC_DTYPES)
def test_tle_load_async_declared_dtypes_promote(dtype_name):
    # Every declared dtype must reach the async-copy path once the >= 4-byte
    # copy-vector gate is met, so declaration and lowering cannot drift apart
    # (cf. PPU issue #1050). A KeyError means a dtype was declared without
    # extending this test.
    signature = _SIGNATURE_BY_DTYPE[dtype_name]
    kernel_signature = {"x_ptr": signature, "out_ptr": signature, "BLOCK": "constexpr"}
    compiled = compile_musa(
        _tle_load_hinted_asm_kernel,
        signature=kernel_signature,
        constexprs={"BLOCK": 4096},
        attrs=_aligned_ptr_attrs(kernel_signature),
    )

    ttgir = compiled.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" in ttgir, ttgir
    assert "tt.load.async" not in ttgir
    assert "llvm.musa.memcpy.g2s" in compiled.asm["llir"]


@pytest.mark.parametrize("dtype_name", ["float8_e4m3fn", "float8_e5m2", "float16", "int8", "int64"])
def test_tle_load_async_roundtrip_bitwise(dtype_name):
    # Numeric backing for the declaration test: the promoted copy moves bytes
    # verbatim (fp8 on uint8 views: torch_musa cannot fill fp8 tensors).
    block = 4096
    torch_dtype = getattr(torch, dtype_name)
    if dtype_name.startswith("float8"):
        x = torch.randint(0, 256, (block, ), dtype=torch.uint8, device="musa").view(torch_dtype)
        out = torch.empty(block, dtype=torch.uint8, device="musa").view(torch_dtype)
    elif torch_dtype.is_floating_point:
        x = torch.randn(block, dtype=torch_dtype, device="musa")
        out = torch.empty(block, dtype=torch_dtype, device="musa")
    else:
        x = torch.randint(-100, 100, (block, ), dtype=torch_dtype, device="musa")
        out = torch.empty(block, dtype=torch_dtype, device="musa")

    compiled = _tle_load_hinted_asm_kernel[(1, )](x, out, BLOCK=block)

    assert "llvm.musa.memcpy.g2s" in compiled.asm["llir"]
    assert torch.equal(out.view(torch.uint8), x.view(torch.uint8))
