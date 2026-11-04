"""Frontend, lowering, and runtime coverage for buffered tensor subslices."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_utils import compile_musa, compile_to_ttir, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _staged_subslice_kernel(
    out,
    START: tl.constexpr,
    LENGTH: tl.constexpr,
    DIM: tl.constexpr,
):
    storage = tle.gpu.alloc((4, 2, 128), dtype=tl.float16, nv_mma_shared_layout=False)
    slot = storage.slot(0)
    view = slot.subslice(START, LENGTH, DIM)
    tl.static_assert(view.type.alloc_shape[0] == 2)
    tl.static_assert(view.type.alloc_shape[1] == 128)
    tl.store(out, view.shape[0] * view.shape[1])


@triton.jit
def _subslice_round_trip_kernel(out):
    values = tl.arange(0, 128).to(tl.float32) + 1.0
    storage = tle.gpu.alloc(
        (128, ),
        dtype=tl.float32,
        init_value=values,
        nv_mma_shared_layout=False,
    )
    # Shared-layout subslices must begin at a tile boundary.  A half-tile
    # offset would be rejected by MemDescSubsliceOp's verifier before LLVM
    # lowering, so use the same 64-element partition used by GEMM operands.
    view = storage.subslice(64, 64)
    loaded = tl.load(tle.gpu.local_ptr(view))
    tl.store(out + tl.arange(0, 64), loaded)


@triton.jit
def _staged_rectangular_sqmma_kernel(a_desc, b_desc, out):
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = 256
    block_k: tl.constexpr = 64
    half_n: tl.constexpr = 128
    a_storage = tle.gpu.alloc(
        (1, block_m, block_k),
        dtype=tl.bfloat16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b_storage = tle.gpu.alloc(
        (1, block_k, block_n),
        dtype=tl.bfloat16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    a = a_storage.slot(0)
    b = b_storage.slot(0)
    a_ready = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * 2)
    b_ready = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * 2)
    tle.gpu.copy(a_desc, a, (block_m, block_k), (0, 0), barrier=a_ready)
    tle.gpu.copy(b_desc, b, (block_k, block_n), (0, 0), barrier=b_ready)
    tle.gpu.barrier_wait(a_ready, phaseIdx=0)
    tle.gpu.barrier_wait(b_ready, phaseIdx=0)

    left = b.subslice(0, half_n, 1)
    right = b.subslice(half_n, half_n, 1)
    left_acc = tle.gpu.wgmma(a, left, tl.zeros((block_m, half_n), dtype=tl.float32))
    right_acc = tle.gpu.wgmma(a, right, tl.zeros((block_m, half_n), dtype=tl.float32))
    left_acc = tle.gpu.wgmma_wait(0, left_acc)
    right_acc = tle.gpu.wgmma_wait(0, right_acc)

    rows = tl.arange(0, block_m)
    cols = tl.arange(0, half_n)
    tl.store(out + rows[:, None] * block_n + cols[None, :], left_acc)
    tl.store(out + rows[:, None] * block_n + half_n + cols[None, :], right_acc)


@triton.jit
def _partitioned_sqmma_top(reader, out):
    ready = reader.wait(0)
    acc = tle.gpu.wgmma(
        ready.slot.a_top,
        ready.slot.b,
        tl.zeros((256, 256), dtype=tl.float32),
    )
    acc = tle.gpu.wgmma_wait(0, acc)
    reader.release(0)
    rows = tl.arange(0, 256)
    cols = tl.arange(0, 256)
    tl.store(out + rows[:, None] * 256 + cols[None, :], acc)


@triton.jit
def _partitioned_sqmma_bottom(reader, out):
    ready = reader.wait(0)
    acc = tle.gpu.wgmma(
        ready.slot.a_bottom,
        ready.slot.b,
        tl.zeros((128, 256), dtype=tl.float32),
    )
    acc = tle.gpu.wgmma_wait(0, acc)
    reader.release(0)
    rows = tl.arange(0, 128)
    cols = tl.arange(0, 256)
    tl.store(out + (256 + rows[:, None]) * 256 + cols[None, :], acc)


@triton.jit
def _partitioned_sqmma_producer(writer, a_top_desc, a_bottom_desc, b_desc):
    slot = writer.acquire(0)
    tle.gpu.copy(a_top_desc, slot.a_top, (256, 32), (0, 0))
    tle.gpu.copy(a_bottom_desc, slot.a_bottom, (128, 32), (0, 0))
    tle.gpu.copy(b_desc, slot.b, (32, 256), (0, 0))
    writer.commit(0)


@triton.jit
def _partitioned_sqmma_kernel(a_top_desc, a_bottom_desc, b_desc, out):
    a_top = tle.gpu.alloc(
        (1, 256, 32),
        dtype=tl.bfloat16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    a_bottom = tle.gpu.alloc(
        (1, 128, 32),
        dtype=tl.bfloat16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b = tle.gpu.alloc(
        (1, 32, 256),
        dtype=tl.bfloat16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="partitioned_sqmma",
        readers=("top", "bottom"),
        a_top=a_top,
        a_bottom=a_bottom,
        b=b,
    )
    tle.gpu.warp_specialize(
        [
            (_partitioned_sqmma_top, (pipe.reader("top"), out)),
            (_partitioned_sqmma_bottom, (pipe.reader("bottom"), out)),
            (
                _partitioned_sqmma_producer,
                (pipe.writer(), a_top_desc, a_bottom_desc, b_desc),
            ),
        ],
        worker_num_warps=[8, 4],
        worker_num_regs=[168, 24],
    )


@pytest.mark.parametrize(
    "start,length,dim",
    [(0, 1, 0), (1, 1, -2), (64, 64, 1), (64, 64, -1)],
)
def test_tle_buffered_tensor_subslice_builds_native_memdesc_view(start, length, dim):
    ttir = compile_to_ttir(
        _staged_subslice_kernel,
        {"out": "*i32"},
        {
            "START": start,
            "LENGTH": length,
            "DIM": dim,
        },
    )
    assert "ttg.memdesc_index" in ttir
    assert "ttg.memdesc_subslice" in ttir


def test_tle_buffered_tensor_subslice_lowers_to_llvm():
    compiled = compile_musa(_subslice_round_trip_kernel, {"out": "*fp32"})
    assert "ttg.memdesc_subslice" in compiled.asm["ttir"]
    assert "ttg.memdesc_subslice" not in compiled.asm["llir"]


def test_tle_staged_rectangular_slice_lowers_through_sqmma():
    compiled = compile_musa(
        _staged_rectangular_sqmma_kernel,
        {
            "a_desc": "tensordesc<bf16[64, 64]>",
            "b_desc": "tensordesc<bf16[64, 256]>",
            "out": "*fp32",
        },
    )
    ttir = compiled.asm["ttir"]
    assert ttir.count("ttg.memdesc_subslice") == 2
    assert "llvm.musa.sqmma.bfmma." in compiled.asm["llir"]


def test_tle_partitioned_sqmma_uses_contextual_warp_layouts_and_captures():
    compiled = compile_musa(
        _partitioned_sqmma_kernel,
        {
            "a_top_desc": "tensordesc<bf16[256, 32]>",
            "a_bottom_desc": "tensordesc<bf16[128, 32]>",
            "b_desc": "tensordesc<bf16[32, 256]>",
            "out": "*fp32",
        },
    )
    ttgir = compiled.asm["ttgir"]
    assert "partition0" in ttgir and "num_warps(8)" in ttgir
    assert "partition1" in ttgir and "num_warps(4)" in ttgir
    assert ttgir.count("ttmg.squad_dot") >= 2
    assert "ttg.warp_specialize" not in compiled.asm["llir"]


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_buffered_tensor_subslice_runtime_uses_view_offset():
    out = torch.empty((64, ), device="musa", dtype=torch.float32)
    _subslice_round_trip_kernel[(1, )](out, num_warps=1)
    torch.musa.synchronize()

    expected = torch.arange(65, 129, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_staged_rectangular_slice_sqmma_runtime_matches_torch():
    from triton.tools.tensor_descriptor import TensorDescriptor

    block_m, block_n, block_k = 64, 256, 64
    torch.manual_seed(20260925)
    a_cpu = torch.randn((block_m, block_k), dtype=torch.bfloat16)
    b_cpu = torch.randn((block_k, block_n), dtype=torch.bfloat16)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((block_m, block_n), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n])

    _staged_rectangular_sqmma_kernel[(1, )](a_desc, b_desc, out, num_warps=4, num_stages=1)
    torch.musa.synchronize()

    expected = a_cpu.float() @ b_cpu.float()
    torch.testing.assert_close(out.cpu(), expected, atol=3e-1, rtol=1e-1)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_partitioned_sqmma_runtime_matches_torch():
    from triton.tools.tensor_descriptor import TensorDescriptor

    torch.manual_seed(20260925)
    a_top_cpu = torch.randn((256, 32), dtype=torch.bfloat16)
    a_bottom_cpu = torch.randn((128, 32), dtype=torch.bfloat16)
    b_cpu = torch.randn((32, 256), dtype=torch.bfloat16)
    a_top = a_top_cpu.to("musa")
    a_bottom = a_bottom_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((384, 256), device="musa", dtype=torch.float32)
    a_top_desc = TensorDescriptor.from_tensor(a_top, [256, 32])
    a_bottom_desc = TensorDescriptor.from_tensor(a_bottom, [128, 32])
    b_desc = TensorDescriptor.from_tensor(b, [32, 256])

    _partitioned_sqmma_kernel[(1, )](
        a_top_desc,
        a_bottom_desc,
        b_desc,
        out,
        num_warps=16,
        num_stages=1,
    )
    torch.musa.synchronize()

    a_cpu = torch.cat((a_top_cpu, a_bottom_cpu), dim=0)
    expected = a_cpu.float() @ b_cpu.float()
    torch.testing.assert_close(out.cpu(), expected, atol=3e-1, rtol=1e-1)
