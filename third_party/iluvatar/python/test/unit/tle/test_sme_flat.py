# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Natural-order SME bytes, warp ownership, FIFO tails and slot reuse."""
import pytest
import re
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu.iluvatar import sme
from triton.compiler.compiler import ASTSource, GPUTarget
from triton.language.core import extern, extern_elementwise


@extern
def _flat_sme_clock(_semantic=None):
    return extern_elementwise(
        "", "", [], {(): ("llvm.bi.sl.gettime64", tl.int64)},
        is_pure=False, _semantic=_semantic,
    )


@triton.jit(noinline=True)
def _flat_sme_collective(src, dst):
    producers: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [1, 64], [16, 1], [1, 0])
    consumers: tl.constexpr = tle.gpu.BlockEncoding([1, 8], [1, 64], [16, 1], [1, 0])
    warp = tle.gpu.set_layout(tl.arange(0, 16)[:, None], producers)
    arena = tle.gpu.alloc(
        [8192], dtype=tl.uint16,
        layout=tle.gpu.swizzled_shared_layout(1, 1, 1, [0], 1, [1], [1], [0]),
        nv_mma_shared_layout=False,
    )
    base = tle.gpu.local_ptr(arena, (0,))
    sme.copy(src + warp * 512, base + warp * 512)
    sme.wait()
    owner = tl.arange(0, 16)[:, None]
    element = tl.arange(0, 512)[None, :]
    index = tle.gpu.set_layout(tl.broadcast_to(owner * 512 + element, [16, 512]), consumers)
    value = tl.load(tle.gpu.set_layout(base + index, consumers))
    tl.store(tle.gpu.set_layout(dst + index, consumers), value)
    sme.wait(drain_reads=True)


@triton.jit
def _flat_sme_profiled_call(src, dst, stamps):
    tl.store(stamps, _flat_sme_clock())
    _flat_sme_collective(src, dst)
    tl.store(stamps + 1, _flat_sme_clock())


@triton.jit(noinline=True)
def _flat_sme_nested_wrapper(src, dst):
    _flat_sme_collective(src, dst)


@triton.jit(noinline=True)
def _flat_sme_scalar_helper(src, dst):
    tl.store(dst, tl.load(src).to(tl.int64))


@triton.jit
def _flat_sme_nested_profiled_call(src, dst, stamps):
    _flat_sme_scalar_helper(src, stamps + 2)
    tl.store(stamps, _flat_sme_clock())
    _flat_sme_nested_wrapper(src, dst)
    tl.store(stamps + 1, _flat_sme_clock())


def test_convergence_reaches_nested_callers_without_marking_scalar_helper():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = torch.arange(8192, dtype=torch.int32, device="cuda").to(torch.uint16)
    output = torch.empty_like(source)
    stamps = torch.zeros(3, dtype=torch.int64, device="cuda")
    compiled = _flat_sme_nested_profiled_call[(1,)](source, output, stamps, num_warps=16)
    torch.cuda.synchronize()
    assert torch.equal(output, source)
    llir = compiled.asm["llir"]
    for name, convergent in (("_flat_sme_collective", True),
                             ("_flat_sme_nested_wrapper", True),
                             ("_flat_sme_scalar_helper", False)):
        function = re.search(r"define[^\n]*\." + name + r"__[^\n]*#(\d+)", llir)
        assert function, name
        attributes = re.search(r"attributes #" + function[1] + r" = \{([^\n]*)\}", llir)
        assert attributes and ("convergent" in attributes[1]) == convergent


def test_predicated_timestamp_preserves_convergent_sme_callee(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = torch.arange(8192, dtype=torch.int32, device="cuda").to(torch.uint16)
    output = torch.full((8192 + 17,), 65535, dtype=torch.int32, device="cuda").to(torch.uint16)
    stamps = torch.zeros(2, dtype=torch.int64, device="cuda")
    compiled = _flat_sme_profiled_call[(1,)](source, output, stamps, num_warps=16)
    torch.cuda.synchronize()
    (tmp_path / "predicated-sme-callee.llir").write_text(compiled.asm["llir"])
    assert torch.equal(output[:8192], source)
    assert torch.equal(output[8192:], torch.full_like(output[8192:], 65535))
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        _flat_sme_profiled_call[(1,)](source, output, stamps, num_warps=16)
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output[:8192], source)


@triton.jit
def _flat_sme_recycled_scratch(src, dst, DRAIN_READS: tl.constexpr,
                               CARRIED: tl.constexpr):
    producers: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [1, 64], [16, 1], [1, 0])
    consumers: tl.constexpr = tle.gpu.BlockEncoding([1, 8], [32, 2], [8, 2], [1, 0])
    reduction: tl.constexpr = tle.gpu.BlockEncoding([1, 2], [4, 16], [16, 1], [1, 0])
    warp = tle.gpu.set_layout(tl.arange(0, 16)[:, None], producers)
    arena = tle.gpu.alloc(
        [8192], dtype=tl.uint16,
        layout=tle.gpu.swizzled_shared_layout(1, 1, 1, [0], 1, [1], [1], [0]),
        nv_mma_shared_layout=False,
    )
    base = tle.gpu.local_ptr(arena, (0,))
    n = tl.arange(0, 512)[:, None]
    k = tl.arange(0, 32)[None, :]
    value = tle.gpu.set_layout(tl.broadcast_to(n * 32 + k, (512, 32)).to(tl.float32), consumers)
    for step in range(8):
        sme.copy(src + step * 8192 + warp * 512, base + warp * 512)
        if CARRIED:
            source_value = tl.load(src + step * 8192 + tl.arange(0, 32)).to(tl.float32)
            value = tle.gpu.set_layout(value + source_value[None, :], consumers)
    # The payload need not be read, but its async writes still own the arena.
    # A warp-local drain cannot let another warp recycle it as scratch yet.
    sme.wait(0, drain_reads=DRAIN_READS)
    value = tle.gpu.set_layout(value, reduction)
    tl.store(dst + tl.arange(0, 512), tl.sum(value, 1))


@pytest.mark.parametrize("drain_reads", [False, True])
@pytest.mark.parametrize("carried", [False, True])
def test_flat_sme_pending_writes_precede_recycled_conversion_scratch(drain_reads, carried):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = torch.full((8 * 8192,), 65535, dtype=torch.int32, device="cuda").to(torch.uint16)
    output = torch.empty(512, device="cuda")
    expected = (torch.arange(512, device="cuda") * 1024 + 496 +
                (8 * 32 * 65535 if carried else 0)).float()
    _flat_sme_recycled_scratch[(1,)](source, output, drain_reads, carried,
                                    num_warps=16, enable_fp_fusion=False)
    torch.cuda.synchronize()
    assert torch.equal(output, expected)
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        _flat_sme_recycled_scratch[(1,)](source, output, drain_reads, carried,
                                        num_warps=16, enable_fp_fusion=False)
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected)


@triton.jit
def _flat_pipeline(src, dst, STEPS: tl.constexpr):
    producers: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [1, 64], [8, 1], [1, 0])
    consumers: tl.constexpr = tle.gpu.BlockEncoding([1, 8], [1, 64], [8, 1], [1, 0])
    warp = tle.gpu.set_layout(tl.arange(0, 8)[:, None], producers)
    arena = tle.gpu.alloc(
        [4096], dtype=tl.uint16,
        layout=tle.gpu.swizzled_shared_layout(1, 1, 1, [0], 1, [1], [1], [0]),
        nv_mma_shared_layout=False,
    )
    base = tle.gpu.local_ptr(arena, (0,))
    for initial in tl.static_range(2):
        sme.copy(src + initial * 2048 + warp * 512,
                 base + initial * 2048 + warp * 512, active=warp < 4)
    element = tl.arange(0, 512)[None, :]
    owner = tl.arange(0, 8)[:, None]
    indices = tle.gpu.set_layout(tl.broadcast_to(owner * 512 + element, [8, 512]), consumers)
    for step in range(STEPS):
        slot = step % 2 * 2048
        if step < STEPS - 1:
            sme.wait(1)
        else:
            sme.wait(0)
        value = tl.load(tle.gpu.set_layout(base + slot + indices, consumers),
                        mask=owner < 4, other=0)
        tl.store(dst + step * 2048 + owner * 512 + element, value, mask=owner < 4)
        # Keep the next stage in flight while draining each consumer lane's
        # own shared reads. It is then safe to recycle the current stage.
        sme.wait(1, drain_reads=True)
        if step + 2 < STEPS:
            sme.copy(src + (step + 2) * 2048 + warp * 512,
                     base + slot + warp * 512, active=warp < 4)
    sme.wait(0, drain_reads=True)


@pytest.mark.parametrize("steps", [2, 17, 65])
def test_flat_sme_fifo_preserves_every_payload_byte(steps):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    torch.manual_seed(209)
    storage = torch.randint(-32768, 32768, (steps * 2048,), device="cuda", dtype=torch.int16)
    source = storage.view(torch.uint16)
    destination = torch.empty_like(source)
    compiled = _flat_pipeline[(1,)](source, destination, steps, num_warps=8)
    torch.cuda.synchronize()
    assert torch.equal(source, destination)
    assert "llvm.bi.sme.load.1x16b64" in compiled.asm["llir"]
    assert "llvm.bi.sl.waitcnt" in compiled.asm["llir"]
    # A terminal wait in the loop tail must not circulate through the
    # dependency analysis and serialize independent warp-owned generations.
    assert "llvm.nvvm.barrier.cta.sync.aligned.all" not in compiled.asm["llir"]
    assert "iluvatar_tle.sme_copy" not in compiled.asm["llir"]
    # Arbitrary uint16 values expose byte reordering, including every BF16
    # exponent/sign pattern. Repeated graph executions recycle all slots.
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        _flat_pipeline[(1,)](source, destination, steps, num_warps=8)
    for _ in range(5):
        storage.random_(-32768, 32768)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(source, destination)


def test_flat_sme_wait_keeps_independent_memory_queues_in_flight():
    compiled = triton.compile(
        ASTSource(_flat_pipeline, signature={"src": "*u16", "dst": "*u16"},
                  constexprs={"STEPS": 17}),
        target=GPUTarget("corex", 71, 64), options={"num_warps": 8},
    )
    # A warp rendezvous must order the elected issuer without a CTA memory
    # fence that drains unrelated global, scalar and shared load queues.
    assert "llvm.bi.sch.barrier" in compiled.asm["llir"]
    assert "llvm.nvvm.membar.cta" not in compiled.asm["llir"]


@triton.jit
def _nonuniform_pointer(src, dst):
    arena = tle.gpu.alloc(
        [512], dtype=tl.uint16,
        layout=tle.gpu.swizzled_shared_layout(1, 1, 1, [0], 1, [1], [1], [0]),
        nv_mma_shared_layout=False,
    )
    base = tle.gpu.local_ptr(arena, (0,))
    lane = tl.arange(0, 64)
    sme.copy(src + lane, base + lane)


def test_flat_sme_rejects_lane_varying_pointers(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        triton.compile(ASTSource(_nonuniform_pointer,
                                 signature={"src": "*u16", "dst": "*u16"}, constexprs={}),
                       target=GPUTarget("corex", 71, 64), options={"num_warps": 8})
    assert "must be warp-uniform" in capfd.readouterr().err


@triton.jit
def _invalid_wait(src, PENDING: tl.constexpr):
    sme.wait(PENDING)


@pytest.mark.parametrize("pending", [-1, 64, True])
def test_flat_sme_rejects_invalid_pending_count(pending):
    with pytest.raises(Exception, match="SME max_pending must be"):
        triton.compile(ASTSource(_invalid_wait, signature={"src": "*u16"},
                                 constexprs={"PENDING": pending}),
                       target=GPUTarget("corex", 71, 64), options={"num_warps": 8})


@triton.jit
def _flat_non_power_arena(src, dst):
    producers: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [1, 64], [8, 1], [1, 0])
    consumers: tl.constexpr = tle.gpu.BlockEncoding([1, 8], [1, 64], [8, 1], [1, 0])
    # MemDesc permits a non-power-of-two leading dimension. Natural-order
    # local pointers need affine byte offsets, not a binary layout inverse.
    arena = tle.gpu.alloc(
        [3072], dtype=tl.uint16,
        layout=tle.gpu.swizzled_shared_layout(1, 1, 1, [0], 1, [1], [1], [0]),
        nv_mma_shared_layout=False,
    )
    base = tle.gpu.local_ptr(arena, (0,))
    owner = tl.arange(0, 8)[:, None]
    warp = tle.gpu.set_layout(owner, producers)
    sme.copy(src + warp * 512, base + warp * 512, active=warp < 6)
    sme.wait()
    element = tl.arange(0, 512)[None, :]
    indices = tle.gpu.set_layout(tl.broadcast_to(owner * 512 + element, [8, 512]), consumers)
    value = tl.load(tle.gpu.set_layout(base + indices, consumers), mask=owner < 6, other=0)
    tl.store(dst + owner * 512 + element, value, mask=owner < 6)
    sme.wait(drain_reads=True)


def test_flat_sme_uses_exact_non_power_of_two_arena():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = torch.randint(-32768, 32768, (3072,), device="cuda", dtype=torch.int16)
    destination = torch.empty_like(source)
    compiled = _flat_non_power_arena[(1,)](source.view(torch.uint16), destination.view(torch.uint16),
                                          num_warps=8)
    torch.cuda.synchronize()
    assert compiled.metadata.shared == 3072 * 2
    assert torch.equal(source, destination)


@triton.jit
def _flat_copy_extent(src, dst, BYTES: tl.constexpr):
    arena = tle.gpu.alloc(
        [512], dtype=tl.uint16,
        layout=tle.gpu.swizzled_shared_layout(1, 1, 1, [0], 1, [1], [1], [0]),
        nv_mma_shared_layout=False,
    )
    base = tle.gpu.local_ptr(arena, (0,))
    sme.copy(src, base, nbytes=BYTES)
    sme.wait()
    element = tl.arange(0, 512)
    value = tl.load(base + element, mask=element < BYTES // 2, other=0)
    tl.store(dst + element, value, mask=element < BYTES // 2)
    sme.wait(drain_reads=True)


@pytest.mark.parametrize("nbytes", [512, 1024])
def test_flat_sme_explicit_extent_preserves_exact_payload(nbytes):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = torch.randint(-32768, 32768, (512,), device="cuda", dtype=torch.int16)
    destination = torch.full_like(source, 777)
    compiled = _flat_copy_extent[(1,)](source.view(torch.uint16), destination.view(torch.uint16),
                                       nbytes, num_warps=1)
    torch.cuda.synchronize()
    assert torch.equal(source[:nbytes // 2], destination[:nbytes // 2])
    assert torch.equal(destination[nbytes // 2:], torch.full_like(destination[nbytes // 2:], 777))
    intrinsic = "llvm.bi.sme.load.1x8b64" if nbytes == 512 else "llvm.bi.sme.load.1x16b64"
    assert intrinsic in compiled.asm["llir"]


@pytest.mark.parametrize("nbytes", [0, 256, 2048, True])
def test_flat_sme_rejects_unsupported_extent(nbytes):
    with pytest.raises(Exception, match="Flat SME nbytes must be"):
        triton.compile(ASTSource(_flat_copy_extent, signature={"src": "*u16", "dst": "*u16"},
                                 constexprs={"BYTES": nbytes}),
                       target=GPUTarget("corex", 71, 64), options={"num_warps": 1})
