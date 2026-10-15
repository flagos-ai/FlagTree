# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Load source-composed rdata directly into the unchanged dense device ABI."""
from __future__ import annotations

from contextlib import ExitStack
from math import prod
from pathlib import Path
import ctypes

from triton.flagmega.errors import ArtifactError
from triton.flagmega.artifacts.source_layout import component_layouts


def byte_addresses(program, shape, maps, start, count):
    """Bounded CPU reference for the same byte-address map used on CUDA."""
    import numpy as np
    local = np.arange(start, start + count, dtype=np.int64)
    flat = np.zeros(count, dtype=np.int64)
    valid = np.ones(count, dtype=np.bool_)
    for axis, capacity in enumerate(shape):
        coord = local // prod(shape[axis + 1:]) % capacity
        if axis < len(maps):
            active, stages = maps[axis]
            valid &= coord < active
            for index, block, shards, capacity in stages:
                coord = coord + index * capacity if block == 0 else (coord // block) * (shards * block) + index * block + coord % block
        flat += coord * prod(program["shape"][axis + 1:])
    for transform in program["transforms"]:
        shape = transform["shape"]
        flat = sum((flat // prod(shape[axis + 1:]) % capacity) * transform["source_strides"][axis]
                   for axis, capacity in enumerate(shape))
    concat = program["concat_shape"]
    axis = program["axis"]
    trailing = prod(concat[axis + 1:])
    axis_coord = flat // trailing % concat[axis]
    prefix = np.concatenate(([0], np.cumsum([operand["extent"] for operand in program["operands"]])))
    child = np.searchsorted(prefix, axis_coord, side="right") - 1
    extent = np.diff(prefix)[child]
    address = ((flat // (concat[axis] * trailing)) * extent + axis_coord - prefix[child]) * trailing + flat % trailing
    return child, address, valid


class _RegisteredFile:
    """Read-only file pages mapped into CUDA, with explicit pin lifetime."""
    def __init__(self, path):
        import numpy as np
        self.array = np.memmap(path, dtype=np.uint8, mode="r")
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.driver.cuMemHostRegister_v2.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
        self.driver.cuMemHostGetDevicePointer_v2.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint)
        self.driver.cuMemHostUnregister.argtypes = (ctypes.c_void_p,)
        self.address = int(self.array.ctypes.data)
        result = self.driver.cuMemHostRegister_v2(self.address, self.array.nbytes, 0x01 | 0x02 | 0x08)
        if result:
            self.array._mmap.close()
            raise ArtifactError(f"CUDA cannot register readonly source pages: driver status {result}.")
        pointer = ctypes.c_uint64()
        result = self.driver.cuMemHostGetDevicePointer_v2(ctypes.byref(pointer), self.address, 0)
        if result:
            self.close()
            raise ArtifactError(f"CUDA cannot map readonly source pages: driver status {result}.")
        self.device_pointer = pointer.value

    def close(self):
        result = self.driver.cuMemHostUnregister(self.address)
        self.array._mmap.close()
        if result:
            raise ArtifactError(f"CUDA cannot unregister readonly source pages: driver status {result}.")


def load_source_rdata(root, index, module, *, chip, device):
    """Read only physical chunks; fuse virtual views with the required H2D copy.

    The caller verifies payloads before invoking this loader. CPU is an
    explicitly selected reference target, never a CUDA failure fallback.
    """
    import numpy as np
    import torch
    from triton.flagmega.ir import verify_buffer_plan
    root = Path(root)
    target = torch.device(device)
    size = int(index["segment_nbytes"])
    pool = torch.zeros(size, dtype=torch.uint8, device=target)
    host = torch.from_file(str(root / index["image"]), shared=False, size=index["nbytes"], dtype=torch.uint8)
    for chunk in index["segment_chunks"][chip]:
        offset, count = chunk["offset"], chunk["nbytes"]
        if chunk.get("zero"):
            continue
        # This view owns no model-sized CPU staging storage. copy_ transfers
        # an already materialized range, bounded by the writer's chunk size.
        pool.narrow(0, offset, count).copy_(host.narrow(0, chip * size + offset, count))
    buffers = verify_buffer_plan(module).buffer_map
    if target.type == "cpu":
        sources = [np.memmap(root / source["path"], dtype=np.uint8, mode="r") for source in index["sources"]]
        output = pool.numpy()
        for program in index["programs"]:
            for offset, shape, maps in component_layouts(buffers[program["buffer"]], chip):
                for start in range(0, prod(shape), 1024 * 1024):
                    count = min(1024 * 1024, prod(shape) - start)
                    child, addresses, valid = byte_addresses(program, shape, maps, start, count)
                    piece = np.zeros(count, dtype=np.uint8)
                    for i, operand in enumerate(program["operands"]):
                        selected = valid & (child == i)
                        piece[selected] = sources[operand["source"]][operand["offset"] + addresses[selected]]
                    output[offset + start:offset + start + count] = piece
        return pool
    if target.type != "cuda":
        raise ArtifactError(f"Source-composed rdata does not support device {target}.")
    from triton.flagmega.runtime.source_rdata_kernel import gather_source_bytes
    with torch.cuda.device(target), ExitStack() as stack:
        mappings = []
        for source in index["sources"]:
            mapping = _RegisteredFile(root / source["path"])
            stack.callback(mapping.close)
            mappings.append(mapping)
        # Finish all reads before unregistering any source, including on a
        # subsequent launch exception. Register this callback last (LIFO).
        stack.callback(torch.cuda.synchronize, target)
        for program in index["programs"]:
            layouts = list(component_layouts(buffers[program["buffer"]], chip))
            shape = tuple(layouts[0][1])
            maps = layouts[0][2]
            stages = tuple(tuple((block, shards, capacity) for _, block, shards, capacity in axis[1]) for axis in maps)
            rows = []
            for _, other_shape, owner_maps in layouts:
                other_stages = tuple(tuple((block, shards, capacity) for _, block, shards, capacity in axis[1]) for axis in owner_maps)
                if tuple(other_shape) != shape or other_stages != stages:
                    raise ArtifactError("Readonly source owner components have incompatible static geometry.")
                rows.append([value for active, stage_list in owner_maps
                             for value in (active, *(stage[0] for stage in stage_list))])
            parameters = torch.tensor(rows or [[]], dtype=torch.int64, device=target)
            offsets = torch.tensor([layout[0] for layout in layouts], dtype=torch.int64, device=target)
            addresses = torch.tensor([mappings[operand["source"]].device_pointer + operand["offset"]
                                      for operand in program["operands"]], dtype=torch.uint64, device=target)
            extents = [operand["extent"] for operand in program["operands"]]
            prefix = torch.tensor([0, *np.cumsum(extents).tolist()], dtype=torch.int64, device=target)
            stage_counts = tuple(len(axis) for axis in stages)
            stage_begin = tuple(sum(stage_counts[:axis]) for axis in range(len(stages)))
            flat_stages = tuple(stage for axis in stages for stage in axis)
            transforms = program["transforms"]
            transform_counts = [len(transform["shape"]) for transform in transforms]
            transform_begin = tuple(sum(transform_counts[:i]) for i in range(len(transforms)))
            gather_source_bytes[( (prod(shape) + 1023) // 1024, len(layouts))](
                pool, addresses, prefix, offsets, parameters,
                SHAPE=shape, GLOBAL_SHAPE=tuple(program["shape"]),
                LOCAL_STRIDES=tuple(prod(shape[axis + 1:]) for axis in range(len(shape))),
                GLOBAL_STRIDES=tuple(prod(program["shape"][axis + 1:]) for axis in range(len(shape))),
                STAGE_COUNTS=stage_counts, STAGE_BEGIN=stage_begin,
                STAGE_BLOCKS=tuple(stage[0] for stage in flat_stages),
                STAGE_SHARDS=tuple(stage[1] for stage in flat_stages),
                STAGE_CAPACITIES=tuple(stage[2] for stage in flat_stages),
                TRANSFORM_BEGIN=transform_begin,
                TRANSFORM_END=tuple(begin + count for begin, count in zip(transform_begin, transform_counts)),
                TRANSFORM_DIMS=tuple(dim for transform in transforms for dim in transform["shape"]),
                TRANSFORM_OUTPUT_STRIDES=tuple(prod(transform["shape"][axis + 1:])
                                               for transform in transforms for axis in range(len(transform["shape"]))),
                TRANSFORM_SOURCE_STRIDES=tuple(stride for transform in transforms for stride in transform["source_strides"]),
                PARAMS=parameters.shape[1], CONCAT=tuple(program["concat_shape"]), AXIS=program["axis"],
                OPERANDS=len(extents), UNIFORM=extents[0] if len(set(extents)) == 1 else 0,
                SEARCH_STEPS=(len(extents) + 1).bit_length(),
                TRAILING=prod(program["concat_shape"][program["axis"] + 1:]),
                COMPONENT_NBYTES=prod(shape), BLOCK=1024,
            )
    return pool
