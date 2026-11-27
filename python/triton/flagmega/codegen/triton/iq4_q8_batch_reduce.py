"""Batch the accepted tree; shared stores include warp0's original accumulator.

Each owner warp starts with original warp0, adds warp1/2/3 in that order,
then calls the unmodified accepted butterfly. Synchronization stays count128.
"""
import triton as tr
import triton.language as tl
import triton.experimental.tle.language.gpu as gpu
from .iq4_q8_kernels import E, warp0_sum

@tr.jit
def batch_partial_store(p, u, S, slot, w, l):
    index = slot * 128 + w * 32 + l
    pp = gpu.set_layout(gpu.local_ptr(S, (index,), shape=(4,32)), E)
    packed = p.to(tl.uint32, bitcast=True).to(tl.uint64) | (u.to(tl.uint32, bitcast=True).to(tl.uint64) << 32)
    tl.store(pp, packed)

@tr.jit
def batch_partial_read(S, slot, l, active):
    index = slot * 128 + l
    ptr = gpu.set_layout(gpu.local_ptr(S, (index,), shape=(4,32)), E)
    pair = tl.load(ptr, active, 0)
    p = (pair & 0xffffffff).to(tl.uint32).to(tl.float32, bitcast=True)
    u = (pair >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    for t in tl.static_range(3):
        ptr = gpu.set_layout(gpu.local_ptr(S, (index + 32 * (t + 1),), shape=(4,32)), E)
        pair = tl.load(ptr, active, 0)
        v = (pair & 0xffffffff).to(tl.uint32).to(tl.float32, bitcast=True)
        p = tl.inline_asm_elementwise(
            'add.rn.f32 $0,$1,$2;', constraints='=f,f,f',
            args=[p, v], dtype=tl.float32, is_pure=True, pack=1)
        v = (pair >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
        u = tl.inline_asm_elementwise(
            'add.rn.f32 $0,$1,$2;', constraints='=f,f,f',
            args=[u, v], dtype=tl.float32, is_pure=True, pack=1)
    return gpu.set_layout(p,E),gpu.set_layout(u,E)

@tr.jit
def batch_typed_reduce(S, owner_slot, w, l, G: tl.constexpr):
    # Safe materialized pointer even in the unused G=2 warps.
    safe_slot = tl.minimum(owner_slot, G - 1)
    active = owner_slot < G
    p,u = batch_partial_read(S, safe_slot, l, active)
    owner = gpu.set_layout(tl.where(active, 0, 1), E)
    p = warp0_sum(p, owner, l)
    u = warp0_sum(u, owner, l)
    return p, gpu.set_layout(tl.where((owner == 0) & (l == 0), u, 0.0), E)
