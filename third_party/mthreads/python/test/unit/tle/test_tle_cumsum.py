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


@triton.jit
def _auto_noinline_histogram_callee(
    data_ptr,
    hist_ptr,
    out_total,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    VEC: tl.constexpr = 4
    lane = tl.arange(0, BLOCK)
    vec = tl.arange(0, VEC)
    for t in tl.range(0, N):
        base = t * BLOCK * VEC + lane * VEC
        offs = base[:, None] + vec[None, :]
        x = tl.load(data_ptr + offs)
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        bin_idx = (bits >> 5).to(tl.uint32)
        tl.atomic_add(hist_ptr + bin_idx, 1, sem="relaxed", scope="cta")
    tl.debug_barrier()
    counts = tl.load(hist_ptr + lane)
    _, total = tle.cumsum(counts, axis=0)
    tl.store(out_total, total)


@triton.jit
def _auto_noinline_histogram_kernel(
    data_ptr,
    total_out,
    hist_dump_out,
    NBINS: tl.constexpr,
    BLOCK: tl.constexpr,
    N: tl.constexpr,
):
    hist = tle.gpu.alloc((NBINS, ), dtype=tl.int32, nv_mma_shared_layout=False)
    hist_ptr = tle.gpu.local_ptr(hist, (0, ))
    tl.store(hist_ptr + tl.arange(0, NBINS), 0)
    tl.debug_barrier()
    _auto_noinline_histogram_callee(data_ptr, hist_ptr, total_out, N, BLOCK)
    tl.store(hist_dump_out + tl.arange(0, NBINS), tl.load(hist_ptr + tl.arange(0, NBINS)))


def test_tle_cumsum_auto_noinline_callee_is_not_inlined():
    compiled = compile_musa(
        _auto_noinline_histogram_kernel,
        signature={
            "data_ptr": "*fp32",
            "total_out": "*i32",
            "hist_dump_out": "*i32",
            "NBINS": "constexpr",
            "BLOCK": "constexpr",
            "N": "constexpr",
        },
        constexprs={"NBINS": 2048, "BLOCK": 512, "N": 1},
    )
    ttir = compiled.asm["ttir"]
    assert "tt.call" in ttir, ("callee containing tle.cumsum should not be inlined on MUSA")


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_cumsum_auto_noinline_histogram_not_corrupted():
    block = 512
    nbins = 2048
    n_tiles = 1
    data = torch.randn(block * 4 * n_tiles, device="musa", dtype=torch.float32)
    total = torch.empty((1, ), device="musa", dtype=torch.int32)
    hist_dump = torch.empty((nbins, ), device="musa", dtype=torch.int32)

    _auto_noinline_histogram_kernel[(1, )](
        data,
        total,
        hist_dump,
        NBINS=nbins,
        BLOCK=block,
        N=n_tiles,
        num_warps=block // 32,
    )

    expected_count = block * 4 * n_tiles
    assert hist_dump.sum().item() == expected_count, (
        f"histogram sum should be {expected_count}, got {hist_dump.sum().item()}")
