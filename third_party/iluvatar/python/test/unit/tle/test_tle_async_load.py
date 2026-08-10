import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from utils import compile_iluvatar

DTYPE_CASES = [
    pytest.param("fp32", tl.float32, torch.float32, id="fp32"),
    pytest.param("fp16", tl.float16, torch.float16, id="fp16"),
    pytest.param("bf16", tl.bfloat16, torch.bfloat16, id="bf16"),
]

BLOCK = 64
N_I32 = BLOCK // 2


@triton.jit
def _tle_load_fp32_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr, IS_ASYNC: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = tle.load(x_ptr + offs, is_async=IS_ASYNC)
    tl.store(out_ptr + offs, vals)


def _make_tle_load_half_kernel():

    @triton.jit
    def _tle_load_half_kernel(x_ptr, out_ptr, N_I32: tl.constexpr, IS_ASYNC: tl.constexpr):
        offs = tl.arange(0, N_I32)
        x_i32 = tl.cast(x_ptr, tl.pointer_type(tl.int32))
        out_i32 = tl.cast(out_ptr, tl.pointer_type(tl.int32))
        vals = tle.load(x_i32 + offs, is_async=IS_ASYNC)
        tl.store(out_i32 + offs, vals)

    return _tle_load_half_kernel


TLE_LOAD_KERNELS = {
    "fp32": _tle_load_fp32_kernel,
    "fp16": _make_tle_load_half_kernel(),
    "bf16": _make_tle_load_half_kernel(),
}


@triton.jit
def _direct_load_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, vals)


@triton.jit
def _tle_load_block_ptr_fp32_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    block = tl.make_block_ptr(x_ptr, shape=(BLOCK, ), strides=(1, ), offsets=(0, ), block_shape=(BLOCK, ), order=(0, ))
    vals = tle.load(block, boundary_check=(0, ), padding_option="zero", is_async=True)
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, vals)


def _make_tle_load_block_ptr_half_kernel():

    @triton.jit
    def _tle_load_block_ptr_half_kernel(x_ptr, out_ptr, N_I32: tl.constexpr):
        x_i32 = tl.cast(x_ptr, tl.pointer_type(tl.int32))
        out_i32 = tl.cast(out_ptr, tl.pointer_type(tl.int32))
        block = tl.make_block_ptr(x_i32, shape=(N_I32, ), strides=(1, ), offsets=(0, ), block_shape=(N_I32, ),
                                  order=(0, ))
        packed = tle.load(block, boundary_check=(0, ), padding_option="zero", is_async=True)
        offs = tl.arange(0, N_I32)
        tl.store(out_i32 + offs, packed)

    return _tle_load_block_ptr_half_kernel


TLE_LOAD_BLOCK_PTR_KERNELS = {
    "fp32": _tle_load_block_ptr_fp32_kernel,
    "fp16": _make_tle_load_block_ptr_half_kernel(),
    "bf16": _make_tle_load_block_ptr_half_kernel(),
}


def _load_compile_constexprs(dtype_str: str, is_async: bool):
    if dtype_str == "fp32":
        return {"BLOCK": BLOCK, "IS_ASYNC": is_async}
    return {"N_I32": N_I32, "IS_ASYNC": is_async}


def _load_compile_signature(dtype_str: str):
    if dtype_str == "fp32":
        return {
            "x_ptr": f"*{dtype_str}",
            "out_ptr": f"*{dtype_str}",
            "BLOCK": "constexpr",
            "IS_ASYNC": "constexpr",
        }
    return {
        "x_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "N_I32": "constexpr",
        "IS_ASYNC": "constexpr",
    }


@pytest.mark.parametrize("dtype_str,tl_dtype,torch_dtype", DTYPE_CASES)
def test_tle_load_async_attr(dtype_str, tl_dtype, torch_dtype):
    async_ttir = compile_iluvatar(
        TLE_LOAD_KERNELS[dtype_str],
        signature=_load_compile_signature(dtype_str),
        constexprs=_load_compile_constexprs(dtype_str, is_async=True),
    ).asm["ttir"]
    assert "tt.load.async = true" in async_ttir, async_ttir


@pytest.mark.parametrize("dtype_str,tl_dtype,torch_dtype", DTYPE_CASES)
@pytest.mark.parametrize("is_async", [False, True])
def test_tle_load_async_codegen(dtype_str, tl_dtype, torch_dtype, is_async):
    compiled = compile_iluvatar(
        TLE_LOAD_KERNELS[dtype_str],
        signature=_load_compile_signature(dtype_str),
        constexprs=_load_compile_constexprs(dtype_str, is_async),
    )

    ttgir = compiled.asm["ttgir"]
    has_async_copy = "ttg.async_copy_global_to_local" in ttgir
    assert has_async_copy is is_async, ttgir
    assert "tt.load.async" not in ttgir


@pytest.mark.parametrize("dtype_str,tl_dtype,torch_dtype", DTYPE_CASES)
def test_tle_load_block_ptr_codegen(dtype_str, tl_dtype, torch_dtype):
    if dtype_str == "fp32":
        signature = {"x_ptr": f"*{dtype_str}", "out_ptr": f"*{dtype_str}", "BLOCK": "constexpr"}
        constexprs = {"BLOCK": BLOCK}
    else:
        signature = {"x_ptr": f"*{dtype_str}", "out_ptr": f"*{dtype_str}", "N_I32": "constexpr"}
        constexprs = {"N_I32": N_I32}

    compiled = compile_iluvatar(
        TLE_LOAD_BLOCK_PTR_KERNELS[dtype_str],
        signature=signature,
        constexprs=constexprs,
    )

    ttgir = compiled.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" in ttgir, ttgir
    assert "tt.load.async" not in ttgir


@pytest.mark.parametrize("dtype_str,tl_dtype,torch_dtype", DTYPE_CASES)
@pytest.mark.parametrize("is_async", [False, True])
def test_tle_load(device, dtype_str, tl_dtype, torch_dtype, is_async):
    torch.manual_seed(0)
    x = torch.randn(BLOCK, device=device, dtype=torch_dtype)
    out_tle = torch.empty_like(x)
    out_ref = torch.empty_like(x)

    grid = (1, )
    kernel = TLE_LOAD_KERNELS[dtype_str]
    if dtype_str == "fp32":
        kernel[grid](x, out_tle, BLOCK=BLOCK, IS_ASYNC=is_async)
    else:
        kernel[grid](x, out_tle, N_I32=N_I32, IS_ASYNC=is_async)
    _direct_load_kernel[grid](x, out_ref, BLOCK=BLOCK)

    torch.testing.assert_close(out_tle.cpu(), out_ref.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(out_tle.cpu(), x.cpu(), atol=0, rtol=0)
