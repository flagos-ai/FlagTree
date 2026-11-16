# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Sparse census preserves the real barrier and role-local participant ABI."""
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

GRID = tl.constexpr(tle.device_mesh({'block': [('block_y', 8), ('block_x', 16)]}))


@triton.jit(noinline=True)
def _census_leaf(data, output, occurrence):
    owner = tl.program_id(0)
    tl.store(data + owner, owner + 7)
    tle.distributed_barrier(GRID, census=occurrence)
    value = tl.load(data + (owner + 1) % 128)
    tl.store(output + owner, value)


@triton.jit(noinline=True)
def _census_consumer(data, output):
    tle.gpu.census_mapping(0, 0)
    _census_leaf(data, output, 0)
    tle.gpu.census_mark(1, 0)
    tle.gpu.census_mapping(0, 1)


@triton.jit(noinline=True)
def _census_producer(data, output):
    tle.gpu.census_mapping(1, 0)
    _census_leaf(data + 128, output + 128, 2)
    tle.gpu.census_mark(3, 0)
    tle.gpu.census_mapping(1, 1)


@triton.jit
def census_boundary(data, output):
    tle.gpu.census_configure(4, 128)
    tle.gpu.warp_specialize([
        (_census_consumer, (data, output)),
        (_census_producer, (data, output)),
    ], [1], [32])


def compile_census_boundary():
    return triton.compile(
        ASTSource(census_boundary, {'data': '*i32', 'output': '*i32'},
                  attrs={(i,): [('tt.divisibility', 16)] for i in range(2)}),
        target=GPUTarget('cuda', 90, 32), options={'num_warps': 8, 'ptx_version': 87})


def test_publication_census_normal_native():
    compiled = compile_census_boundary()
    ptx = compiled.asm['ptx']
    assert 'mov.u64 timestamp, %globaltimer' in ptx
    assert 'atom.add.release.gpu.u32' in ptx
    assert 'fence.acquire.gpu' in ptx
    assert 'TRACE_OVERFLOW' not in ptx
    assert compiled.metadata.num_warps == 12
    assert compiled.metadata.profile_scratch_size > 0
    assert compiled.metadata.global_scratch_size == 8
