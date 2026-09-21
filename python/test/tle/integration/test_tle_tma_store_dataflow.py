# flagtree tle
"""TMA store source lifetime across loops, branches, and shared-memory reuse."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 9,
        reason="TMA store dataflow requires NVIDIA Hopper or newer",
    ),
]


@pytest.fixture
def with_allocator():
    from triton.runtime._allocation import NullAllocator

    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())


@triton.jit(do_not_specialize=["outer", "inner"])
def nested_store(desc, outer, inner, BM: tl.constexpr, BN: tl.constexpr, USE_WHILE: tl.constexpr):
    a = tle.gpu.alloc([BM, BN], tl.float32, scope=tle.gpu.smem)
    b = tle.gpu.alloc([BM, BN], tl.float32, scope=tle.gpu.smem)
    base = (tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]).to(tl.float32)
    pid = tl.program_id(0)
    for i in tl.range(0, outer, disable_licm=True, loop_unroll_factor=1):
        t = (pid * outer + i) * (inner + 2)
        tl.store(tle.gpu.local_ptr(a), base + t)
        tle.gpu.copy(a, desc, [BM, BN], [t * BM, 0])
        if USE_WHILE:
            j = 0
            while j < inner:
                u = t + 1 + j
                tl.store(tle.gpu.local_ptr(b), base + u)
                tle.gpu.copy(b, desc, [BM, BN], [u * BM, 0])
                j += 1
        else:
            for j in tl.range(0, inner, disable_licm=True, loop_unroll_factor=1):
                u = t + 1 + j
                tl.store(tle.gpu.local_ptr(b), base + u)
                tle.gpu.copy(b, desc, [BM, BN], [u * BM, 0])
        u = t + 1 + inner
        tl.store(tle.gpu.local_ptr(b), base + u)
        tle.gpu.copy(b, desc, [BM, BN], [u * BM, 0])


@triton.jit(do_not_specialize=["phase"])
def branch_store(desc, phase, ITERS: tl.constexpr, MODE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    a = tle.gpu.alloc([BM, BN], tl.float32, scope=tle.gpu.smem)
    b = tle.gpu.alloc([BM, BN], tl.float32, scope=tle.gpu.smem)
    c = tle.gpu.alloc([BM, BN], tl.float32, scope=tle.gpu.smem)
    base = (tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]).to(tl.float32)
    pid = tl.program_id(0)
    for i in tl.range(0, ITERS, disable_licm=True, loop_unroll_factor=1):
        t = (pid * ITERS + i) * 4
        pred = (pid + i + phase) % 2 != 0
        if MODE == 2:
            if pred:
                tl.store(tle.gpu.local_ptr(a), base + t)
                tle.gpu.copy(a, desc, [BM, BN], [t * BM, 0])
                tl.store(tle.gpu.local_ptr(b), base + t + 1)
                tle.gpu.copy(b, desc, [BM, BN], [(t + 1) * BM, 0])
            else:
                tl.store(tle.gpu.local_ptr(b), base + t)
                tle.gpu.copy(b, desc, [BM, BN], [t * BM, 0])
                tl.store(tle.gpu.local_ptr(a), base + t + 1)
                tle.gpu.copy(a, desc, [BM, BN], [(t + 1) * BM, 0])
        else:
            tl.store(tle.gpu.local_ptr(a), base + t)
            tle.gpu.copy(a, desc, [BM, BN], [t * BM, 0])
            if pred:
                tl.store(tle.gpu.local_ptr(b), base + t + 1)
                tle.gpu.copy(b, desc, [BM, BN], [(t + 1) * BM, 0])
            elif MODE == 1:
                tl.store(tle.gpu.local_ptr(c), base + t + 1)
                tle.gpu.copy(c, desc, [BM, BN], [(t + 1) * BM, 0])
        tl.store(tle.gpu.local_ptr(a), base + t + 2)
        tle.gpu.copy(a, desc, [BM, BN], [(t + 2) * BM, 0])
        tl.store(tle.gpu.local_ptr(b), base + t + 3)
        tle.gpu.copy(b, desc, [BM, BN], [(t + 3) * BM, 0])


@triton.jit(do_not_specialize=["outer", "inner"])
def store_across_wgmma(desc, outer, inner, BM: tl.constexpr, BN: tl.constexpr):
    a = tle.gpu.alloc([BM, 16], tl.float16, scope=tle.gpu.smem)
    b = tle.gpu.alloc([16, BN], tl.float16, scope=tle.gpu.smem)
    staging = tle.gpu.alloc([BM, BN], tl.float32, scope=tle.gpu.smem)
    tl.store(tle.gpu.local_ptr(a), tl.full((BM, 16), 1, tl.float16))
    tl.store(tle.gpu.local_ptr(b), tl.full((16, BN), 1, tl.float16))
    pid = tl.program_id(0)
    for i in tl.range(0, outer, disable_licm=True, loop_unroll_factor=1):
        acc = tl.zeros((BM, BN), tl.float32)
        for j in tl.range(0, inner, disable_licm=True, loop_unroll_factor=1):
            acc = tle.gpu.wgmma(a, b, acc)
            acc = tle.gpu.wgmma_wait(0, acc)
        tile = pid * outer + i
        tl.store(tle.gpu.local_ptr(staging), acc + tile)
        tle.gpu.copy(staging, desc, [BM, BN], [tile * BM, 0])


def check_replays(launch, output, reference):

    def check():
        torch.cuda.synchronize()
        torch.testing.assert_close(output, reference, rtol=0, atol=0, equal_nan=True)

    for _ in range(3):
        output.fill_(float("nan"))
        launch()
        check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(3):
        output.fill_(float("nan"))
        graph.replay()
        check()


def make_output(tiles, bm, bn):
    output = torch.empty(max(tiles, 1) * bm, bn, device="cuda", dtype=torch.float32)
    if tiles:
        reference = (torch.arange(tiles, device="cuda")[:, None] +
                     torch.arange(bm * bn, device="cuda")[None, :]).float().view_as(output)
    else:
        reference = torch.full_like(output, float("nan"))
    return output, reference, TensorDescriptor.from_tensor(output, block_shape=[bm, bn])


@pytest.mark.parametrize("outer", [0, 1, 8])
@pytest.mark.parametrize("inner", [0, 1, 8])
@pytest.mark.parametrize("bn", [16, 128])
@pytest.mark.parametrize("use_while", [False, True], ids=["for", "while"])
@pytest.mark.require_tle("gpu.alloc", "gpu.copy", "gpu.local_ptr")
def test_nested_store_dataflow(outer, inner, bn, use_while, with_allocator):
    bm = 64
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    output, reference, desc = make_output(sms * outer * (inner + 2), bm, bn)
    check_replays(lambda: nested_store[(sms, )](desc, outer, inner, bm, bn, use_while, num_warps=4), output, reference)


@pytest.mark.parametrize("mode", [0, 1, 2], ids=["unequal", "equal", "reversed"])
@pytest.mark.parametrize("phase", [0, 1])
@pytest.mark.parametrize("bn", [16, 128])
@pytest.mark.require_tle("gpu.alloc", "gpu.copy", "gpu.local_ptr")
def test_branch_store_dataflow(mode, phase, bn, with_allocator):
    bm, iters = 64, 8
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    output, reference, desc = make_output(sms * iters * 4, bm, bn)
    if mode == 0:
        step = torch.arange(sms * iters, device="cuda")
        missing = ((step // iters + step % iters + phase) % 2) == 0
        reference.view(sms * iters, 4, bm, bn)[missing, 1] = float("nan")
    check_replays(lambda: branch_store[(sms, )](desc, phase, iters, mode, bm, bn, num_warps=4), output, reference)


@pytest.mark.parametrize("outer", [0, 1, 8])
@pytest.mark.parametrize("inner", [0, 1, 8])
@pytest.mark.parametrize("bn", [16, 128])
@pytest.mark.require_tle("gpu.alloc", "gpu.copy", "gpu.local_ptr")
def test_store_across_wgmma(outer, inner, bn, with_allocator):
    bm = 64
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    tiles = sms * outer
    output = torch.empty((max(tiles, 1) + 1) * bm, bn, device="cuda", dtype=torch.float32)
    reference = torch.full_like(output, float("nan"))
    if tiles:
        reference[:tiles * bm].view(tiles, bm, bn).copy_(
            (torch.arange(tiles, device="cuda", dtype=torch.float32) + 16 * inner)[:, None, None])
    desc = TensorDescriptor.from_tensor(output, block_shape=[bm, bn])
    check_replays(lambda: store_across_wgmma[(sms, )](desc, outer, inner, bm, bn, num_warps=4), output, reference)


@triton.jit(do_not_specialize=["flag"])
def source_offset_reuse(desc, sink, x_ptr, flag, BM: tl.constexpr, BN: tl.constexpr, GROUPS: tl.constexpr,
                        IN_REGION: tl.constexpr, SCRATCH_REUSER: tl.constexpr):
    pid = tl.program_id(0)
    if IN_REGION:
        # The source dies with the region; the allocator may hand its offset on.
        if flag != 0:
            src = tle.gpu.alloc([BM, BN], tl.float32, init_value=tl.full((BM, BN), 1.0, tl.float32))
            for g in tl.static_range(GROUPS):
                tle.gpu.copy(src, desc, [BM, BN], [pid * BM, g * BN])
    else:
        src = tle.gpu.alloc([BM, BN], tl.float32, init_value=tl.full((BM, BN), 1.0, tl.float32))
        for g in tl.static_range(GROUPS):
            tle.gpu.copy(src, desc, [BM, BN], [pid * BM, g * BN])
    rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BN))
    cols = tl.broadcast_to(tl.arange(0, BN)[None, :], (BM, BN))
    if SCRATCH_REUSER:
        # No new buffer, but the transposed store needs a layout conversion,
        # and its scratch shared memory may land on the dead source's offset.
        x = tl.load(x_ptr + rows * BN + cols)
        rows_t = tl.broadcast_to(tl.arange(0, BN)[:, None], (BN, BM))
        cols_t = tl.broadcast_to(tl.arange(0, BM)[None, :], (BN, BM))
        tl.store(sink + rows_t * BM + cols_t, tl.trans(x))
    else:
        # A new buffer may land on the dead source's offset while the TMA engine
        # still reads it, so its initialization must not race the stores.
        dst = tle.gpu.alloc([BM, BN], tl.float32, init_value=tl.zeros((BM, BN), tl.float32))
        tl.store(sink + rows * BN + cols, tl.load(tle.gpu.local_ptr(dst, (rows, cols))))


@pytest.mark.require_tle("gpu.alloc", "gpu.copy", "gpu.local_ptr")
@pytest.mark.parametrize("scratch_reuser", [False, True], ids=["local_alloc", "convert_layout"])
@pytest.mark.parametrize("in_region", [False, True], ids=["straight_line", "in_region"])
def test_store_source_offset_reuse(in_region, scratch_reuser, with_allocator):
    bm, bn, groups = 128, 32, 8
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    output = torch.empty(sms * bm, groups * bn, device="cuda", dtype=torch.float32)
    reference = torch.ones_like(output)
    sink = torch.empty(bm * bn, device="cuda", dtype=torch.float32)
    x = torch.randn(bm * bn, device="cuda", dtype=torch.float32)
    desc = TensorDescriptor.from_tensor(output, block_shape=[bm, bn])
    check_replays(
        lambda: source_offset_reuse[(sms, )](desc, sink, x, 3, bm, bn, groups, in_region, scratch_reuser, num_warps=4),
        output, reference)


@triton.jit(do_not_specialize=["iters"])
def hinted_pending_store(desc, iters, BM: tl.constexpr, BN: tl.constexpr, HINTED: tl.constexpr):
    staging = tle.gpu.alloc([BM, BN], tl.float32, scope=tle.gpu.smem)
    pid = tl.program_id(0)
    base = (tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]).to(tl.float32)
    tl.store(tle.gpu.local_ptr(staging), base + pid)
    for i in tl.range(0, iters, disable_licm=True, loop_unroll_factor=1):
        if HINTED:
            tle.gpu.copy(staging, desc, [BM, BN], [(pid * iters + i) * BM, 0])  # @hint: tma_store_pending=8
        else:
            tle.gpu.copy(staging, desc, [BM, BN], [(pid * iters + i) * BM, 0])


@triton.jit(do_not_specialize=["iters"])
def conflicting_hint_store(desc, iters, flag, BM: tl.constexpr, BN: tl.constexpr):
    staging = tle.gpu.alloc([BM, BN], tl.float32, scope=tle.gpu.smem)
    pid = tl.program_id(0)
    base = (tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]).to(tl.float32)
    tl.store(tle.gpu.local_ptr(staging), base + pid)
    for i in tl.range(0, iters, disable_licm=True, loop_unroll_factor=1):
        # Both sides of the branch ask for a different bound; the larger one
        # applies to the whole kernel.
        if flag != 0:
            tle.gpu.copy(staging, desc, [BM, BN], [(pid * iters + i) * BM, 0])  # @hint: tma_store_pending=2
        else:
            tle.gpu.copy(staging, desc, [BM, BN], [(pid * iters + i) * BM, 0])  # @hint: tma_store_pending=4


def pending_waits(binary):
    return sorted(int(n) for n in re.findall(r"cp\.async\.bulk\.wait_group\.read\s+(\d+)", binary.asm["ptx"]))


@pytest.mark.require_tle("gpu.alloc", "gpu.copy", "gpu.local_ptr")
@pytest.mark.parametrize("hinted", [False, True])
def test_tma_store_pending_hint(hinted, with_allocator):
    bm, bn, iters = 64, 128, 8
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    output = torch.empty(sms * iters * bm, bn, device="cuda", dtype=torch.float32)
    reference = (torch.arange(sms, device="cuda", dtype=torch.float32).repeat_interleave(iters)[:, None, None] +
                 torch.arange(bm * bn, device="cuda", dtype=torch.float32).view(1, bm, bn)).view_as(output)
    desc = TensorDescriptor.from_tensor(output, block_shape=[bm, bn])
    output.fill_(float("nan"))
    binary = hinted_pending_store[(sms, )](desc, iters, bm, bn, hinted, num_warps=4)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    # Without the hint every group is waited for before the next commit.
    assert pending_waits(binary) == ([0, 7] if hinted else [0, 0])
    check_replays(lambda: hinted_pending_store[(sms, )](desc, iters, bm, bn, hinted, num_warps=4), output, reference)


@pytest.mark.require_tle("gpu.alloc", "gpu.copy", "gpu.local_ptr")
def test_tma_store_pending_hint_takes_the_maximum(with_allocator):
    bm, bn, iters = 64, 128, 8
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    output = torch.empty(sms * iters * bm, bn, device="cuda", dtype=torch.float32)
    reference = (torch.arange(sms, device="cuda", dtype=torch.float32).repeat_interleave(iters)[:, None, None] +
                 torch.arange(bm * bn, device="cuda", dtype=torch.float32).view(1, bm, bn)).view_as(output)
    desc = TensorDescriptor.from_tensor(output, block_shape=[bm, bn])
    output.fill_(float("nan"))
    binary = conflicting_hint_store[(sms, )](desc, iters, 3, bm, bn, num_warps=4)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    # Hints of 2 and 4 combine into 4, so the loop waits with three groups left.
    assert pending_waits(binary) == [0, 3]
