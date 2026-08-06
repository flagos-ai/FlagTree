"""Compile coverage for the initial mthreads TLE SQMMA contract."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.errors import CompilationError

from test_tle_utils import compile_musa, compile_to_ttir, require_mthreads_libtriton

require_mthreads_libtriton()

_SQMMA_MN_SHAPES = (
    (16, 64),
    (32, 32),
    (32, 64),
    (32, 128),
    (64, 16),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 32),
    (128, 64),
    (128, 128),
)
_SQMMA_DTYPE_CASES = (
    ("float16", 0, 2, "fmma", (16, 32, 64), 1.0, 1.0e-1, 5.0e-2),
    ("bfloat16", 1, 2, "bfmma", (16, 32, 64), 1.0, 3.0e-1, 1.0e-1),
    ("float8_e4m3fn", 2, 1, "e4m3", (32, 64, 128), 0.5, 5.0e-1, 1.5e-1),
)
_SQMMA_SHAPE_CASES = tuple(
    pytest.param(
        torch_dtype_name,
        dtype_kind,
        input_bytes,
        intrinsic_tag,
        m,
        n,
        k,
        scale,
        atol,
        rtol,
        id=f"{intrinsic_tag}-m{m}-n{n}-k{k}",
    )
    for torch_dtype_name, dtype_kind, input_bytes, intrinsic_tag, k_shapes, scale, atol, rtol in _SQMMA_DTYPE_CASES
    for m, n in _SQMMA_MN_SHAPES
    for k in k_shapes)


@triton.jit
def _tle_sqmma_kernel(out):
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None, nv_mma_shared_layout=False)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None, nv_mma_shared_layout=False)
    acc = tl.zeros((128, 128), dtype=tl.float32)
    acc = tle.gpu.wgmma(a, b, acc)
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_nonzero_wait_kernel(out):
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None, nv_mma_shared_layout=False)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None, nv_mma_shared_layout=False)
    acc = tle.gpu.wgmma(a, b, tl.zeros((128, 128), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(1, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_runtime_kernel(a_desc, b_desc, out):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    a = tle.gpu.alloc(
        (block_m, block_k),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=False,
    )
    b = tle.gpu.alloc(
        (block_k, block_n),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=False,
    )
    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * 2)
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * 2)

    tle.gpu.copy(a_desc, a, (block_m, block_k), (0, 0), barrier=a_full)
    tle.gpu.copy(b_desc, b, (block_k, block_n), (0, 0), barrier=b_full)
    tle.gpu.barrier_wait(a_full, phaseIdx=0)
    tle.gpu.barrier_wait(b_full, phaseIdx=0)

    acc = tle.gpu.wgmma(a, b, tl.zeros((block_m, block_n), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_for_loop_runtime_kernel(a_desc, b_desc, out, k_tiles: tl.constexpr):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    a = tle.gpu.alloc(
        (block_m, block_k),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=False,
    )
    b = tle.gpu.alloc(
        (block_k, block_n),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=False,
    )
    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * 2)
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * 2)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_iter in range(0, k_tiles):
        k_offset = k_iter * block_k
        tle.gpu.copy(a_desc, a, (block_m, block_k), (0, k_offset), barrier=a_full)
        tle.gpu.copy(b_desc, b, (block_k, block_n), (k_offset, 0), barrier=b_full)
        tle.gpu.barrier_wait(a_full, phaseIdx=k_iter)
        tle.gpu.barrier_wait(b_full, phaseIdx=k_iter)
        acc = tle.gpu.wgmma(a, b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)

    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_dtype_runtime_kernel(
    a_desc,
    b_desc,
    out,
    input_bytes: tl.constexpr,
):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    if input_bytes == 1:
        a = tle.gpu.alloc(
            (block_m, block_k),
            dtype=tl.float8e4nv,
            layout=None,
            nv_mma_shared_layout=False,
        )
        b = tle.gpu.alloc(
            (block_k, block_n),
            dtype=tl.float8e4nv,
            layout=None,
            nv_mma_shared_layout=False,
        )
    else:
        a = tle.gpu.alloc(
            (block_m, block_k),
            dtype=tl.bfloat16,
            layout=None,
            nv_mma_shared_layout=False,
        )
        b = tle.gpu.alloc(
            (block_k, block_n),
            dtype=tl.bfloat16,
            layout=None,
            nv_mma_shared_layout=False,
        )
    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * input_bytes, )
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * input_bytes, )

    tle.gpu.copy(a_desc, a, (block_m, block_k), (0, 0), barrier=a_full)
    tle.gpu.copy(b_desc, b, (block_k, block_n), (0, 0), barrier=b_full)
    tle.gpu.barrier_wait(a_full, phaseIdx=0)
    tle.gpu.barrier_wait(b_full, phaseIdx=0)
    acc = tle.gpu.wgmma(a, b, tl.zeros((block_m, block_n), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_all_shapes_runtime_kernel(
    a_desc,
    b_desc,
    out,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    dtype_kind: tl.constexpr,
    input_bytes: tl.constexpr,
):
    if dtype_kind == 0:
        a = tle.gpu.alloc(
            (block_m, block_k),
            dtype=tl.float16,
            layout=None,
            nv_mma_shared_layout=False,
        )
        b = tle.gpu.alloc(
            (block_k, block_n),
            dtype=tl.float16,
            layout=None,
            nv_mma_shared_layout=False,
        )
    elif dtype_kind == 1:
        a = tle.gpu.alloc(
            (block_m, block_k),
            dtype=tl.bfloat16,
            layout=None,
            nv_mma_shared_layout=False,
        )
        b = tle.gpu.alloc(
            (block_k, block_n),
            dtype=tl.bfloat16,
            layout=None,
            nv_mma_shared_layout=False,
        )
    else:
        a = tle.gpu.alloc(
            (block_m, block_k),
            dtype=tl.float8e4nv,
            layout=None,
            nv_mma_shared_layout=False,
        )
        b = tle.gpu.alloc(
            (block_k, block_n),
            dtype=tl.float8e4nv,
            layout=None,
            nv_mma_shared_layout=False,
        )

    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * input_bytes, )
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * input_bytes, )
    tle.gpu.copy(a_desc, a, (block_m, block_k), (0, 0), barrier=a_full)
    tle.gpu.copy(b_desc, b, (block_k, block_n), (0, 0), barrier=b_full)
    tle.gpu.barrier_wait(a_full, phaseIdx=0)
    tle.gpu.barrier_wait(b_full, phaseIdx=0)
    acc = tle.gpu.wgmma(a, b, tl.zeros((block_m, block_n), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


def test_mthreads_tle_sqmma_ttir_uses_backend_local_names():
    ttir = compile_to_ttir(_tle_sqmma_kernel, {"out": "*fp32"})
    assert "musa_tle.sqmma" in ttir, ttir
    assert "musa_tle.sqmma_wait" in ttir, ttir
    assert "musa_tle.wgmma" not in ttir, ttir
    assert ttir.count("musa_tle.auto_shared_layout") == 2, ttir


def test_mthreads_tle_sqmma_rejects_nonzero_pending_groups():
    with pytest.raises(CompilationError, match="requires pendings=0"):
        compile_to_ttir(_tle_sqmma_nonzero_wait_kernel, {"out": "*fp32"})


def test_mthreads_tle_sqmma_lowers_to_parameterless_wait():
    compiled = compile_musa(_tle_sqmma_kernel, {"out": "*fp32"})
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "musa_tle.sqmma" not in ttgir, ttgir
    assert "ttmg.squad_dot" in ttgir, ttgir
    assert ttgir.count("ttmg.squad_dot_wait") == 1, ttgir
    assert "ttg.local_load" not in ttgir, ttgir
    assert "llvm.musa.sqmma" in llir, llir
    assert "call void @llvm.musa.sqmma.wait()" in llir, llir


def test_mthreads_tle_sqmma_runtime_precision():
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(1234)
    a_cpu = torch.randn((128, 64), dtype=torch.float16)
    b_cpu = torch.randn((64, 128), dtype=torch.float16)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((128, 128), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[128, 64])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[64, 128])

    kernel = _tle_sqmma_runtime_kernel[(1, )](a_desc, b_desc, out, num_warps=4, num_stages=1)
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=5e-2, rtol=5e-2)
    assert "llvm.musa.tme.ld.tile.2d" in kernel.asm["llir"]
    assert "llvm.musa.sqmma" in kernel.asm["llir"]
    assert "call void @llvm.musa.sqmma.wait()" in kernel.asm["llir"]


def test_mthreads_tle_sqmma_for_loop_runtime_precision():
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(1234)
    a_cpu = torch.randn((128, 128), dtype=torch.float16)
    b_cpu = torch.randn((128, 128), dtype=torch.float16)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((128, 128), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[128, 64])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[64, 128])

    kernel = _tle_sqmma_for_loop_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        2,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=7e-2, rtol=5e-2)
    assert "scf.for" in kernel.asm["ttgir"]
    assert "llvm.musa.tme.ld.tile.2d" in kernel.asm["llir"]
    assert "llvm.musa.sqmma" in kernel.asm["llir"]
    assert "call void @llvm.musa.sqmma.wait()" in kernel.asm["llir"]


@pytest.mark.parametrize(
    "torch_dtype_name,input_bytes,intrinsic_tag,atol,rtol",
    [
        ("bfloat16", 2, "bfmma", 1.5e-1, 5e-2),
        ("float8_e4m3fn", 1, "e4m3", 2.5e-1, 1e-1),
    ],
)
def test_mthreads_tle_sqmma_dtype_runtime_precision(
    torch_dtype_name,
    input_bytes,
    intrinsic_tag,
    atol,
    rtol,
):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(1234)
    torch_dtype = getattr(torch, torch_dtype_name)
    scale = 0.5 if input_bytes == 1 else 1.0
    a_cpu = (torch.randn((128, 64), dtype=torch.float32) * scale).to(torch_dtype)
    b_cpu = (torch.randn((64, 128), dtype=torch.float32) * scale).to(torch_dtype)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((128, 128), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[128, 64])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[64, 128])

    kernel = _tle_sqmma_dtype_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        input_bytes,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=atol, rtol=rtol)
    assert "llvm.musa.tme.ld.tile.2d" in kernel.asm["llir"]
    assert f"llvm.musa.sqmma.{intrinsic_tag}" in kernel.asm["llir"]
    assert "call void @llvm.musa.sqmma.wait()" in kernel.asm["llir"]


@pytest.mark.parametrize(
    "torch_dtype_name,dtype_kind,input_bytes,intrinsic_tag,m,n,k,scale,atol,rtol",
    _SQMMA_SHAPE_CASES,
)
def test_mthreads_tle_sqmma_all_supported_shapes_runtime(
    torch_dtype_name,
    dtype_kind,
    input_bytes,
    intrinsic_tag,
    m,
    n,
    k,
    scale,
    atol,
    rtol,
):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(1234)
    torch_dtype = getattr(torch, torch_dtype_name)
    a_cpu = (torch.randn((m, k), dtype=torch.float32) * scale).to(torch_dtype)
    b_cpu = (torch.randn((k, n), dtype=torch.float32) * scale).to(torch_dtype)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((m, n), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[m, k])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[k, n])

    kernel = _tle_sqmma_all_shapes_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        m,
        n,
        k,
        dtype_kind,
        input_bytes,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=atol, rtol=rtol)
    intrinsic = f"llvm.musa.sqmma.{intrinsic_tag}.m{m}n{n}k{k}.mma"
    assert intrinsic in kernel.asm["llir"]
    assert "call void @llvm.musa.sqmma.wait()" in kernel.asm["llir"]


def test_mthreads_tle_sqmma_macro_tile_auto_decomposition_runtime():
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    m, n, k = 256, 256, 64
    torch.manual_seed(5256)
    a_cpu = torch.randn((m, k), dtype=torch.float16)
    b_cpu = torch.randn((k, n), dtype=torch.float16)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((m, n), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[m, k])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[k, n])

    kernel = _tle_sqmma_all_shapes_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        m,
        n,
        k,
        0,
        2,
        num_warps=16,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=1.0e-1, rtol=5.0e-2)

    ttir = kernel.asm["ttir"]
    ttgir = kernel.asm["ttgir"]
    llir = kernel.asm["llir"]
    assert ttir.count("musa_tle.sqmma ") == 1
    assert ttir.count("musa_tle.sqmma_wait ") == 1
    assert "warpsPerCTA = [8, 2]" in ttgir
    assert "instrShape = [128, 128, 64]" in ttgir
    assert ttgir.count("ttmg.squad_dot ") == 1
    intrinsic = "llvm.musa.sqmma.fmma.m128n128k64.mma"
    assert llir.count(f"@{intrinsic}(") == 2  # One call plus one declaration.
    assert "call void @llvm.musa.sqmma.wait()" in llir
