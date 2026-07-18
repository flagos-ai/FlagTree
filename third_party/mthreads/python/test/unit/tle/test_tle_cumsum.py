import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.errors import CompilationError

from test_tle_utils import compile_musa, compile_to_ttir, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _cumsum_kernel(src, exclusive_out, total_out, n: tl.constexpr, BLOCK: tl.constexpr, REVERSE: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(src + offsets, mask=mask, other=0)
    exclusive, total = tle.cumsum(values, axis=0, reverse=REVERSE)
    tl.store(exclusive_out + offsets, exclusive, mask=mask)
    tl.store(total_out, total)


@triton.jit
def _cumsum_2d_axis_kernel(src, out, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.arange(0, ROWS)[:, None]
    cols = tl.arange(0, COLS)[None, :]
    values = tl.load(src + rows * COLS + cols)
    exclusive, _ = tle.cumsum(values, axis=1)
    tl.store(out + rows * COLS + cols, exclusive)


@triton.jit(noinline=True)
def _shared_cumsum_callee(buf, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    ptrs = tle.gpu.local_ptr(buf, (offsets, ))
    values = tl.load(ptrs)
    exclusive, _ = tle.cumsum(values, axis=0)
    tl.store(ptrs, exclusive)


@triton.jit
def _shared_noinline_sentinel_kernel(out, sentinel_out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    buf = tle.gpu.alloc((BLOCK * 2, ), dtype=tl.int32, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(buf, (offsets, ))
    sentinel = tle.gpu.local_ptr(buf, (BLOCK, ))
    tl.store(ptrs, tl.full((BLOCK, ), 1, tl.int32))
    tl.store(sentinel, 123456)
    _shared_cumsum_callee(buf, BLOCK)
    tl.store(out + offsets, tl.load(ptrs))
    tl.store(sentinel_out, tl.load(sentinel))


@triton.jit
def _shared_scalar_base_addptr_sentinel_kernel(out, sentinel_out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    buf = tle.gpu.alloc((BLOCK * 2, ), dtype=tl.int32, nv_mma_shared_layout=False)
    base = tle.gpu.local_ptr(buf, (0, ))
    ptrs = base + offsets
    sentinel = tle.gpu.local_ptr(buf, (BLOCK, ))
    tl.store(ptrs, tl.full((BLOCK, ), 1, tl.int32))
    tl.store(sentinel, 654321)
    values = tl.load(ptrs)
    exclusive, _ = tle.cumsum(values, axis=0)
    tl.store(ptrs, exclusive)
    tl.store(out + offsets, tl.load(ptrs))
    tl.store(sentinel_out, tl.load(sentinel))


def test_tle_cumsum_builder_binding_is_backend_local():
    from triton._C import libtriton
    from triton._C.libtriton import ir

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    builder = ir.builder(context)

    assert hasattr(builder, "create_exclusive_cumsum")
    assert hasattr(libtriton, "mthreads")
    assert not hasattr(libtriton, "tle")


def test_tle_cumsum_ttir_uses_musa_tle_op():
    ttir = compile_to_ttir(
        _cumsum_kernel,
        signature={
            "src": "*fp32",
            "exclusive_out": "*fp32",
            "total_out": "*fp32",
            "n": "constexpr",
            "BLOCK": "constexpr",
            "REVERSE": "constexpr",
        },
        constexprs={"n": 127, "BLOCK": 128, "REVERSE": False},
    )

    assert "musa_tle.exclusive_cumsum" in ttir, ttir
    assert re.search(r"(?<!musa_)\btle\.exclusive_cumsum\b", ttir) is None, ttir


def test_tle_cumsum_ttgir_lowers_to_scan_reduce():
    compiled = compile_musa(
        _cumsum_kernel,
        signature={
            "src": "*fp32",
            "exclusive_out": "*fp32",
            "total_out": "*fp32",
            "n": "constexpr",
            "BLOCK": "constexpr",
            "REVERSE": "constexpr",
        },
        constexprs={"n": 127, "BLOCK": 128, "REVERSE": False},
    )
    ttgir = compiled.asm["ttgir"]

    assert "musa_tle.exclusive_cumsum" not in ttgir, ttgir
    assert '"tt.scan"' in ttgir, ttgir
    assert '"tt.reduce"' in ttgir, ttgir
    assert ("arith.subi" in ttgir) or ("arith.subf" in ttgir), ttgir


def test_tle_cumsum_reverse_ttgir_uses_reverse_scan():
    compiled = compile_musa(
        _cumsum_kernel,
        signature={
            "src": "*fp32",
            "exclusive_out": "*fp32",
            "total_out": "*fp32",
            "n": "constexpr",
            "BLOCK": "constexpr",
            "REVERSE": "constexpr",
        },
        constexprs={"n": 127, "BLOCK": 128, "REVERSE": True},
    )
    ttgir = compiled.asm["ttgir"]

    assert '"tt.scan"' in ttgir, ttgir
    assert "reverse = true" in ttgir, ttgir


def test_tle_cumsum_no_musa_tle_reaches_llir():
    compiled = compile_musa(
        _cumsum_kernel,
        signature={
            "src": "*fp32",
            "exclusive_out": "*fp32",
            "total_out": "*fp32",
            "n": "constexpr",
            "BLOCK": "constexpr",
            "REVERSE": "constexpr",
        },
        constexprs={"n": 127, "BLOCK": 128, "REVERSE": False},
    )

    assert "musa_tle.exclusive_cumsum" not in compiled.asm["llir"], compiled.asm["llir"]


def test_tle_cumsum_rejects_2d_axis_1(capfd):
    with pytest.raises((CompilationError, RuntimeError)) as excinfo:
        compile_musa(
            _cumsum_2d_axis_kernel,
            signature={
                "src": "*fp32",
                "out": "*fp32",
                "ROWS": "constexpr",
                "COLS": "constexpr",
            },
            constexprs={"ROWS": 16, "COLS": 16},
        )
    diagnostics = str(excinfo.value) + capfd.readouterr().err
    assert ("currently only rank-1 tensors are supported" in diagnostics
            or "currently only axis=0 is supported" in diagnostics
            or "PassManager::run failed" in diagnostics), diagnostics


def _runtime_reference(x, n, reverse, out_dtype):
    valid = x[:n].cpu()
    if valid.dtype in (torch.int8, torch.int16, torch.int32):
        work = valid.to(torch.int32)
        if reverse:
            exclusive = torch.flip(torch.cumsum(torch.flip(work, (0, )), 0, dtype=torch.int32), (0, )) - work
        else:
            exclusive = torch.cumsum(work, 0, dtype=torch.int32) - work
        total = work.sum(dtype=torch.int32)
    else:
        work = valid.to(torch.float32)
        if reverse:
            exclusive = torch.flip(torch.cumsum(torch.flip(work, (0, )), 0), (0, )) - work
        else:
            exclusive = torch.cumsum(work, 0) - work
        exclusive = exclusive.to(out_dtype)
        total = work.sum().to(out_dtype)
    return exclusive.to(out_dtype), total.reshape((1, ))


def _make_runtime_input(n, torch_dtype):
    if torch_dtype in (torch.float16, torch.float32, torch.bfloat16):
        return torch.randint(0, 4, (n, ), device="musa").to(torch_dtype)
    return torch.randint(-3, 4, (n, ), device="musa", dtype=torch_dtype)


def _supports_musa_bfloat16():
    if not torch.musa.is_available():
        return False
    try:
        torch.empty((1, ), device="musa", dtype=torch.bfloat16)
    except Exception:
        return False
    return True


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,signature_dtype,out_dtype,out_signature",
    [
        ("int8", torch.int8, "*i8", torch.int32, "*i32"),
        ("int16", torch.int16, "*i16", torch.int32, "*i32"),
        ("int32", torch.int32, "*i32", torch.int32, "*i32"),
        ("float16", torch.float16, "*fp16", torch.float16, "*fp16"),
        ("float32", torch.float32, "*fp32", torch.float32, "*fp32"),
        ("bfloat16", torch.bfloat16, "*bf16", torch.float32, "*fp32"),
    ],
)
@pytest.mark.parametrize("block,n", [(128, 127), (256, 256), (512, 511)])
@pytest.mark.parametrize("reverse", [False, True])
def test_tle_cumsum_runtime_matches_torch(dtype_name, torch_dtype, signature_dtype, out_dtype, out_signature, block, n,
                                          reverse):
    if dtype_name == "bfloat16" and not _supports_musa_bfloat16():
        pytest.skip("MUSA bfloat16 is not available")

    x = _make_runtime_input(n, torch_dtype)
    exclusive_out = torch.empty((n, ), device="musa", dtype=out_dtype)
    total_out = torch.empty((1, ), device="musa", dtype=out_dtype)

    _cumsum_kernel[(1, )](
        x,
        exclusive_out,
        total_out,
        n,
        BLOCK=block,
        REVERSE=reverse,
        num_warps=8,
    )

    expected_exclusive, expected_total = _runtime_reference(x, n, reverse, out_dtype)
    atol = 0 if out_dtype in (torch.int32, torch.float32) else 1e-3
    rtol = 0 if out_dtype in (torch.int32, torch.float32) else 1e-3
    torch.testing.assert_close(exclusive_out.cpu(), expected_exclusive, rtol=rtol, atol=atol)
    torch.testing.assert_close(total_out.cpu(), expected_total, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_cumsum_shared_noinline_preserves_adjacent_sentinel():
    block = 128
    out = torch.empty((block, ), device="musa", dtype=torch.int32)
    sentinel = torch.empty((1, ), device="musa", dtype=torch.int32)

    _shared_noinline_sentinel_kernel[(1, )](out, sentinel, BLOCK=block, num_warps=4)

    torch.testing.assert_close(out.cpu(), torch.arange(0, block, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(sentinel.cpu(), torch.tensor([123456], dtype=torch.int32), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_cumsum_shared_scalar_base_addptr_preserves_adjacent_sentinel():
    block = 128
    out = torch.empty((block, ), device="musa", dtype=torch.int32)
    sentinel = torch.empty((1, ), device="musa", dtype=torch.int32)

    _shared_scalar_base_addptr_sentinel_kernel[(1, )](out, sentinel, BLOCK=block, num_warps=4)

    torch.testing.assert_close(out.cpu(), torch.arange(0, block, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(sentinel.cpu(), torch.tensor([654321], dtype=torch.int32), rtol=0, atol=0)
