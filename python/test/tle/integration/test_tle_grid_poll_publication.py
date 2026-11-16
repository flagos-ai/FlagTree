"""Grid-barrier publication after readers have cached old peer data.

Only the writer acquires readiness. Readers publish readiness and immediately
enter the tested barrier, so readiness does not invalidate their primed L1.
The omitted-acquire branch is a diagnostic sensitivity control, never a valid
implementation of distributed_barrier and never used by the passing test.
"""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

MESH = tl.constexpr(tle.device_mesh({'block': [('block_x', 128)]}))


@triton.jit
def _load_ca(pointer):
    return tl.inline_asm_elementwise('ld.global.ca.b32 $0, [$1];',
                                     constraints='=r,l', args=[pointer],
                                     dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _omit_acquire_barrier(counter):
    # Explicitly incorrect diagnostic protocol: identical release/sense test,
    # with the terminal acquire deliberately omitted. CTA barriers remain.
    tl.debug_barrier()
    tl.inline_asm_elementwise(
        '''{
        .reg .b32 tid, bid, n, add, old, cur, diff;
        .reg .pred nonleader, master, pending;
        mov.u32 tid, %tid.x;
        setp.ne.u32 nonleader, tid, 0;
        @nonleader bra DONE;
        mov.u32 bid, %ctaid.x;
        mov.u32 n, %nctaid.x;
        sub.u32 n, n, 1;
        sub.u32 add, 0x80000000, n;
        setp.eq.u32 master, bid, 0;
        selp.u32 add, add, 1, master;
        atom.add.release.gpu.u32 old, [$1], add;
        WAIT:
        ld.relaxed.gpu.u32 cur, [$1];
        xor.b32 diff, old, cur;
        and.b32 diff, diff, 0x80000000;
        setp.eq.u32 pending, diff, 0;
        @pending bra WAIT;
        DONE:
        mov.u32 $0, 0;
        }''', constraints='=r,l', args=[counter], dtype=tl.int32,
        is_pure=False, pack=1)
    tl.debug_barrier()


@triton.jit
def grid_poll_publication(data, ready, primed, observed, negative_counter,
                          EPOCHS: tl.constexpr, OMIT_ACQUIRE: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, 32)
    for epoch in tl.range(0, EPOCHS, loop_unroll_factor=1):
        old = epoch * 16384 + pid * 32 + lane
        tl.store(data + pid * 32 + lane, old)
        tle.distributed_barrier(MESH)
        if pid != 0:
            cached = _load_ca(data + pid * 32 + lane)
            tl.store(primed + epoch * 4096 + pid * 32 + lane, cached)
            tl.debug_barrier()
            tl.inline_asm_elementwise('st.release.gpu.u32 [$1], $2; mov.u32 $0, 0;',
                                      constraints='=r,l,r', args=[ready + pid * 32, epoch + 1],
                                      dtype=tl.int32, is_pure=False, pack=1)
        else:
            peers = tl.arange(0, 128)
            flags = tl.inline_asm_elementwise('ld.acquire.gpu.u32 $0, [$1];',
                                              constraints='=r,l', args=[ready + peers * 32],
                                              dtype=tl.int32, is_pure=False, pack=1)
            while tl.sum(((flags != epoch + 1) & (peers != 0)).to(tl.int32), 0) != 0:
                flags = tl.inline_asm_elementwise('ld.acquire.gpu.u32 $0, [$1];',
                                                  constraints='=r,l', args=[ready + peers * 32],
                                                  dtype=tl.int32, is_pure=False, pack=1)
            offsets = tl.arange(0, 4096)
            tl.store(data + offsets, epoch * 16384 + offsets + 8192, offsets >= 32)
        if OMIT_ACQUIRE:
            _omit_acquire_barrier(negative_counter)
        else:
            tle.distributed_barrier(MESH)
        actual = _load_ca(data + pid * 32 + lane)
        tl.store(observed + epoch * 4096 + pid * 32 + lane, actual)
        tle.distributed_barrier(MESH)


@pytest.mark.skip_if_tle_unsupported('distributed_barrier')
def test_grid_poll_publishes_to_primed_readers(with_allocator):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('requires SM90 and a resident cooperative grid')
    if torch.cuda.get_device_properties(0).multi_processor_count < 128:
        pytest.skip('128-CTA publication witness requires 128 SMs')
    epochs = 4
    data = torch.zeros(4096, dtype=torch.int32, device='cuda')
    ready = torch.zeros_like(data)
    primed = torch.full((epochs * 4096,), -1, dtype=torch.int32, device='cuda')
    observed = torch.empty_like(primed)
    counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    grid_poll_publication[(128,)](data, ready, primed, observed, counter,
                                  epochs, False, num_warps=4)
    expected = torch.arange(4096, device='cuda')[None, :] + torch.arange(epochs, device='cuda')[:, None] * 16384
    assert torch.equal(primed.reshape(epochs, 4096)[:, 32:], expected[:, 32:].to(torch.int32))
    expected[:, 32:] += 8192
    assert torch.equal(observed.reshape(epochs, 4096), expected.to(torch.int32))
