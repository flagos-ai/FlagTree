import triton as tr
import triton.language as tl
import triton.experimental.tle.language.gpu as gpu
from .iq4_q8_kernels import E, partial_store, partial_read, warp0_sum
CONSUMER_BARRIER_ABI='H003ConsumerPartial-f32-3x32/bar3-count128/v4-compact-padding'
@tr.jit
def persistent_typed_reduce(p, u, w, l, GLU: tl.constexpr):
    B = gpu.alloc_barrier(arrive_count=128)
    S = gpu.alloc((192 if GLU else 96,), tl.float32, layout=None, nv_mma_shared_layout=False, alignment_bytes=16)
    partial_store(p, S, w, l, 0)
    if GLU:
        partial_store(u, S, w, l, 1)
    gpu.barrier_wait(B)
    p = partial_read(p, S, w, l, 0)
    if GLU:
        u = partial_read(u, S, w, l, 1)
    p = warp0_sum(p, w, l)
    if GLU:
        u = warp0_sum(u, w, l)
    gpu.barrier_wait(B)
    return (p, gpu.set_layout(tl.where((w == 0) & (l == 0), u, 0.0), E))
