# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Parallel token owners; the Q8_1 record arithmetic and bytes stay frozen."""
import triton as tr
import triton.language as tl
import triton.experimental.tle.language.gpu as gpu
from .iq4_q8_kernels import E, q8_quantize_lanes

@tr.jit
def q8_prepare_rows(X,Q,K:tl.constexpr,ROWS:tl.constexpr):
    tl.static_assert((ROWS>1)&(ROWS<=5))
    tl.static_assert(K%128==0)
    w=gpu.set_layout(tl.broadcast_to(tl.arange(0,4)[:,None],(4,32)),E)
    l=gpu.set_layout(tl.broadcast_to(tl.arange(0,32)[None,:],(4,32)),E)
    # Flatten token and group ownership into one static launch dimension.
    owner=tl.program_id(0)//(K//128)
    group=4*(tl.program_id(0)%(K//128))+w
    k=32*group+l
    x=gpu.set_layout(tl.load(X+owner*K+k.to(tl.int64),k<K,0).to(tl.float32),E)
    a,d,r,v,q,sq,header=q8_quantize_lanes(x)
    tl.store(Q+owner*(K//32)*36+36*group.to(tl.int64)+4+l,q.to(tl.uint8),group<K//32)
    tl.store(Q.to(tl.pointer_type(tl.uint32))+owner*(K//32)*9+9*group.to(tl.int64),header,(group<K//32)&(l==0))
