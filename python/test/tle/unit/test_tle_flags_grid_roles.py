# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Every consumer warp publishes and consumes distinct words around the grid cut."""
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from test_tle_flags_grid import _selected_barrier
GRID=tl.constexpr(tle.device_mesh({'block':[('block_x',128)]}))

@triton.jit(noinline=True)
def _allwarp_leaf(data,output,epoch,SELECTED:tl.constexpr):
    owner=tl.program_id(0)
    lane=tl.arange(0,256)
    value=epoch*65536+owner*256+lane
    tl.store(data+owner*256+lane,value)
    _selected_barrier(False,128,SELECTED)
    observed=tl.load(data+((owner+1)%128)*256+lane,cache_modifier='.ca')
    tl.store(output+epoch*32768+owner*256+lane,observed)
    _selected_barrier(False,128,SELECTED)

@triton.jit(noinline=True)
def _allwarp_consumer(data,output,SELECTED:tl.constexpr):
    for epoch in range(3):
        _allwarp_leaf(data,output,epoch,SELECTED)

@triton.jit(noinline=True)
def _allwarp_producer(data,output,SELECTED:tl.constexpr):
    for epoch in range(5):
        _allwarp_leaf(data+32768,output+3*32768,epoch,SELECTED)

@triton.jit
def allwarp_publication(data,output,SELECTED:tl.constexpr):
    tle.gpu.warp_specialize([(_allwarp_consumer,(data,output,SELECTED)),(_allwarp_producer,(data,output,SELECTED))],[1],[32])

def compile_allwarp(selected):
    return triton.compile(ASTSource(allwarp_publication,{'data':'*i32','output':'*i32'},constexprs={'SELECTED':selected},attrs={(i,):[('tt.divisibility',16)] for i in range(2)}),target=GPUTarget('cuda',90,32),options={'num_warps':8,'ptx_version':87})
