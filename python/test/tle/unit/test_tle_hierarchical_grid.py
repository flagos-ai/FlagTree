# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Normal native selection, layout and capability boundaries; no GPU launch."""
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

GRID = tle.device_mesh({'block': [('block_y', 8), ('block_x', 16)]})
AXIS8 = GRID.axis_group(('block_y',), group_shape=(8,))


@triton.jit
def publication(data, output, MESH: tl.constexpr, PROTOCOL: tl.constexpr,
                STRIDE: tl.constexpr, EPOCHS: tl.constexpr):
    owner = tl.program_id(0) + tl.num_programs(0) * (
        tl.program_id(1) + tl.num_programs(1) * tl.program_id(2))
    total = tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2)
    lane = tl.arange(0, 32)
    for epoch in tl.range(0, EPOCHS, loop_unroll_factor=1):
        value = epoch * 65536 + owner * 32 + lane
        tl.store(data + owner * 32 + lane, value)
        tle.distributed_barrier(MESH, protocol=PROTOCOL)
        peer = (owner + STRIDE) % total
        observed = tl.load(data + peer * 32 + lane, cache_modifier='.ca')
        tl.store(output + epoch * total * 32 + owner * 32 + lane, observed)
        tle.distributed_barrier(MESH, protocol=PROTOCOL)


def compile_protocol(protocol, *, axis=False, sm=90, ptx=87, epochs=3):
    return triton.compile(
        ASTSource(publication, {'data': '*i32', 'output': '*i32'},
                  constexprs={'MESH': AXIS8 if axis else GRID, 'PROTOCOL': protocol,
                              'STRIDE': 16 if axis else 1, 'EPOCHS': epochs},
                  attrs={(i,): [('tt.divisibility', 16)] for i in range(2)}),
        target=GPUTarget('cuda', sm, 32), options={'num_warps': 8, 'ptx_version': ptx})


@pytest.mark.parametrize('axis', [False, True])
@pytest.mark.parametrize('protocol', [None, 'hierarchical8'])
def test_normal_protocol_layout(axis, protocol):
    compiled = compile_protocol(protocol, axis=axis)
    groups = 16 if axis else 1
    selected = protocol is not None
    assert compiled.metadata.global_scratch_size == groups * 2 * (1152 if selected else 4)
    assert compiled.metadata.global_scratch_align == (128 if selected else 4)
    assert compiled.metadata.profile_scratch_size == 0
    assert compiled.metadata.launch_cooperative_grid
    assert 'atom.add.release.gpu.u32' in compiled.asm['ptx']
    assert 'fence.acquire.gpu' in compiled.asm['ptx']
    assert 'ld.acquire.gpu.u32' not in compiled.asm['ptx']
    assert bool('tle.grid_barrier_protocol' in compiled.asm['ttgir']) == selected


@pytest.mark.parametrize('sm,ptx', [(80, 87), (89, 87), (90, 85)])
def test_selected_unsupported_target_is_rejected(sm, ptx, capfd):
    with pytest.raises(RuntimeError, match='PassManager::run failed'):
        compile_protocol('hierarchical8', sm=sm, ptx=ptx)
    assert 'hierarchical8 requires NVIDIA SM90+ and PTX ISA8.6+' in capfd.readouterr().err


def test_minimum_selected_ptx86():
    compiled = compile_protocol('hierarchical8', ptx=86)
    assert compiled.metadata.global_scratch_align == 128


def test_unknown_protocol_is_rejected():
    with pytest.raises(Exception, match='protocol must be None'):
        compile_protocol('unknown')


WS_AXIS8 = tl.constexpr(AXIS8)
GRID1 = tl.constexpr(tle.device_mesh({'block': [('block_x', 1)]}))
GRID3 = tl.constexpr(tle.device_mesh({'block': [('block_x', 3)]}))
GRID8 = tl.constexpr(tle.device_mesh({'block': [('block_x', 8)]}))
GRID10 = tl.constexpr(tle.device_mesh({'block': [('block_x', 10)]}))
GRID128 = tl.constexpr(tle.device_mesh({'block': [('block_x', 128)]}))


@triton.jit
def _selected_barrier(AXIS: tl.constexpr, COUNT: tl.constexpr,
                      PROTOCOL: tl.constexpr):
    if AXIS:
        tle.distributed_barrier(WS_AXIS8, protocol='hierarchical8' if PROTOCOL else None)
    elif COUNT == 1:
        tle.distributed_barrier(GRID1, protocol='hierarchical8' if PROTOCOL else None)
    elif COUNT == 3:
        tle.distributed_barrier(GRID3, protocol='hierarchical8' if PROTOCOL else None)
    elif COUNT == 8:
        tle.distributed_barrier(GRID8, protocol='hierarchical8' if PROTOCOL else None)
    elif COUNT == 10:
        tle.distributed_barrier(GRID10, protocol='hierarchical8' if PROTOCOL else None)
    else:
        tl.static_assert(COUNT == 128)
        tle.distributed_barrier(GRID128, protocol='hierarchical8' if PROTOCOL else None)


@triton.jit(noinline=True)
def _epoch_leaf(data, output, epoch, AXIS: tl.constexpr, COUNT: tl.constexpr,
                PROTOCOL: tl.constexpr, STRIDE: tl.constexpr,
                SKEW_STEPS: tl.constexpr):
    owner = tl.program_id(0) + tl.num_programs(0) * (
        tl.program_id(1) + tl.num_programs(1) * tl.program_id(2))
    total = tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2)
    lane = tl.arange(0, 32)
    value = (epoch * 65536 + owner * 32 + lane).to(tl.uint32)
    skew_owner = tl.where((epoch & 1) == 0, 0, total - 1)
    if owner == skew_owner:
        for step in tl.range(0, SKEW_STEPS, loop_unroll_factor=1):
            value = value * 1664525 + 1013904223
    tl.store(data + owner * 32 + lane, value.to(tl.int32, bitcast=True))
    _selected_barrier(AXIS, COUNT, PROTOCOL)
    peer = (owner + STRIDE) % total
    observed = tl.load(data + peer * 32 + lane, cache_modifier='.ca')
    tl.store(output + epoch * total * 32 + owner * 32 + lane, observed)
    _selected_barrier(AXIS, COUNT, PROTOCOL)


@triton.jit(noinline=True)
def _nested_epoch(data, output, epoch, AXIS: tl.constexpr, COUNT: tl.constexpr,
                  PROTOCOL: tl.constexpr, STRIDE: tl.constexpr,
                  SKEW_STEPS: tl.constexpr):
    _epoch_leaf(data, output, epoch, AXIS, COUNT, PROTOCOL, STRIDE, SKEW_STEPS)


@triton.jit(noinline=True)
def _role(data, output, AXIS: tl.constexpr, COUNT: tl.constexpr, PROTOCOL: tl.constexpr,
          STRIDE: tl.constexpr, EPOCHS: tl.constexpr, SKEW_STEPS: tl.constexpr):
    owner = tl.program_id(0) + tl.num_programs(0) * (
        tl.program_id(1) + tl.num_programs(1) * tl.program_id(2))
    if AXIS:
        group_extra = (owner % 16) % 3
    else:
        group_extra = 0
    for epoch in tl.range(0, EPOCHS + group_extra, loop_unroll_factor=1):
        _nested_epoch(data, output, epoch, AXIS, COUNT, PROTOCOL, STRIDE, SKEW_STEPS)


@triton.jit(noinline=True)
def _producer_role(data, output, AXIS: tl.constexpr, COUNT: tl.constexpr, PROTOCOL: tl.constexpr,
                   STRIDE: tl.constexpr, EPOCHS: tl.constexpr,
                   SKEW_STEPS: tl.constexpr):
    total = tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2)
    consumer_extent: tl.constexpr = EPOCHS + (2 if AXIS else 0)
    _role(data + total * 32, output + consumer_extent * total * 32,
          AXIS, COUNT, PROTOCOL, STRIDE, EPOCHS + 2, SKEW_STEPS)


@triton.jit
def ws_publication(data, output, AXIS: tl.constexpr, COUNT: tl.constexpr, PROTOCOL: tl.constexpr,
                   STRIDE: tl.constexpr, EPOCHS: tl.constexpr,
                   SKEW_STEPS: tl.constexpr):
    tle.gpu.warp_specialize([
        (_role, (data, output, AXIS, COUNT, PROTOCOL, STRIDE, EPOCHS, SKEW_STEPS)),
        (_producer_role, (data, output, AXIS, COUNT, PROTOCOL, STRIDE, EPOCHS, SKEW_STEPS)),
    ], [1], [32])


def compile_ws(protocol, *, axis=False, count=128, skew=0, epochs=9):
    return triton.compile(
        ASTSource(ws_publication, {'data': '*i32', 'output': '*i32'},
                  constexprs={'AXIS': axis, 'COUNT': count, 'PROTOCOL': protocol is not None,
                              'STRIDE': 16 if axis else 1, 'EPOCHS': epochs,
                              'SKEW_STEPS': skew},
                  attrs={(i,): [('tt.divisibility', 16)] for i in range(2)}),
        target=GPUTarget('cuda', 90, 32), options={'num_warps': 8, 'ptx_version': 87})


@pytest.mark.parametrize('protocol', [None, 'hierarchical8'])
def test_nested_roles_receive_separate_frames(protocol):
    compiled = compile_ws(protocol)
    assert compiled.metadata.num_warps == 12
    assert compiled.metadata.global_scratch_size == 4 * (1152 if protocol else 4)
    assert compiled.metadata.global_scratch_align == (128 if protocol else 4)
    assert compiled.metadata.launch_cooperative_grid
