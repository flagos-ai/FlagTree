# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The byte-preserving layout transfer for mapped readonly source files."""
import triton
import triton.language as tl


@triton.jit
def _map_axis(coord, Parameters, owner, PARAMS: tl.constexpr, AXIS: tl.constexpr,
              BEGIN: tl.constexpr, END: tl.constexpr,
              STAGE_BLOCKS: tl.constexpr, STAGE_SHARDS: tl.constexpr, STAGE_CAPACITIES: tl.constexpr):
    for stage in tl.static_range(BEGIN, END):
        index = tl.load(Parameters + owner * PARAMS + AXIS + BEGIN + stage - BEGIN + 1)
        if STAGE_BLOCKS[stage] == 0:
            coord += index * STAGE_CAPACITIES[stage]
        else:
            coord = (coord // STAGE_BLOCKS[stage] * (STAGE_SHARDS[stage] * STAGE_BLOCKS[stage])
                     + index * STAGE_BLOCKS[stage] + coord % STAGE_BLOCKS[stage])
    return coord


@triton.jit
def _apply_transform(flat, BEGIN: tl.constexpr, END: tl.constexpr,
                     DIMS: tl.constexpr, OUTPUT_STRIDES: tl.constexpr, SOURCE_STRIDES: tl.constexpr):
    result = tl.full(flat.shape, 0, tl.int64)
    for axis in tl.static_range(BEGIN, END):
        result += flat // OUTPUT_STRIDES[axis] % DIMS[axis] * SOURCE_STRIDES[axis]
    return result


@triton.jit
def gather_source_bytes(Output, Sources, Prefix, Offsets, Parameters,
                        SHAPE: tl.constexpr, GLOBAL_SHAPE: tl.constexpr,
                        LOCAL_STRIDES: tl.constexpr, GLOBAL_STRIDES: tl.constexpr,
                        STAGE_COUNTS: tl.constexpr, STAGE_BEGIN: tl.constexpr,
                        STAGE_BLOCKS: tl.constexpr, STAGE_SHARDS: tl.constexpr, STAGE_CAPACITIES: tl.constexpr,
                        TRANSFORM_BEGIN: tl.constexpr, TRANSFORM_END: tl.constexpr, TRANSFORM_DIMS: tl.constexpr,
                        TRANSFORM_OUTPUT_STRIDES: tl.constexpr, TRANSFORM_SOURCE_STRIDES: tl.constexpr,
                        PARAMS: tl.constexpr, CONCAT: tl.constexpr, AXIS: tl.constexpr,
                        OPERANDS: tl.constexpr, UNIFORM: tl.constexpr, SEARCH_STEPS: tl.constexpr,
                        TRAILING: tl.constexpr, COMPONENT_NBYTES: tl.constexpr, BLOCK: tl.constexpr):
    local = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    owner = tl.program_id(1)
    flat = tl.full((BLOCK,), 0, tl.int64)
    valid = tl.full((BLOCK,), True, tl.int1)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = local // LOCAL_STRIDES[axis] % SHAPE[axis]
        if axis < len(STAGE_COUNTS):
            active = tl.load(Parameters + owner * PARAMS + axis + STAGE_BEGIN[axis])
            valid &= coord < active
            coord = _map_axis(coord, Parameters, owner, PARAMS, axis,
                              STAGE_BEGIN[axis], STAGE_BEGIN[axis] + STAGE_COUNTS[axis],
                              STAGE_BLOCKS, STAGE_SHARDS, STAGE_CAPACITIES)
        flat += coord * GLOBAL_STRIDES[axis]
    for transform in tl.static_range(len(TRANSFORM_BEGIN)):
        flat = _apply_transform(flat, TRANSFORM_BEGIN[transform], TRANSFORM_END[transform],
                                TRANSFORM_DIMS, TRANSFORM_OUTPUT_STRIDES, TRANSFORM_SOURCE_STRIDES)
    axis_coord = flat // TRAILING % CONCAT[AXIS]
    if UNIFORM:
        child = axis_coord // UNIFORM
        begin = child * UNIFORM
        extent = UNIFORM
    else:
        lo = tl.full((BLOCK,), 0, tl.int64)
        hi = tl.full((BLOCK,), OPERANDS + 1, tl.int64)
        for _ in range(SEARCH_STEPS):
            mid = (lo + hi) // 2
            value = tl.load(Prefix + mid)
            right = value <= axis_coord
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(right, hi, mid)
        child = lo - 1
        begin = tl.load(Prefix + child)
        extent = tl.load(Prefix + child + 1) - begin
    address = ((flat // (CONCAT[AXIS] * TRAILING)) * extent + axis_coord - begin) * TRAILING + flat % TRAILING
    pointer = (tl.load(Sources + child) + address.to(tl.uint64)).to(tl.pointer_type(tl.uint8))
    mask = local < COMPONENT_NBYTES
    value = tl.load(pointer, mask=mask & valid, other=0)
    offset = tl.load(Offsets + owner)
    tl.store(Output + offset + local, value, mask=mask)
