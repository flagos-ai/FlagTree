# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Load source-composed rdata directly into the unchanged dense device ABI."""
from __future__ import annotations

from contextlib import ExitStack
from math import prod
from pathlib import Path
import ctypes
import mmap

from triton.flagmega.errors import ArtifactError
from triton.flagmega.artifacts.source_layout import component_layouts
from triton.flagmega.runtime.source_rdata_windows import source_transfers


PINNED_SOURCE_LIMIT = 1024 * 1024 * 1024
SOURCE_COPY_CHUNK_BYTES = 8 * 1024 * 1024


def _contiguous_source_copy(program, layouts, sources, destination_nbytes):
    """Prove a complete byte copy, without inspecting source payloads.

    The normal artifact validator remains responsible for source/program
    integrity. A nonempty map, transform, concat or partial owner needs the
    generic address kernel even if its total size matches a dense tensor.
    """
    shape = tuple(program["shape"])
    axis = program["axis"]
    if (len(layouts) != 1 or not shape
            or any(type(n) is not int or n <= 0 for n in shape)
            or type(axis) is not int or not 0 <= axis < len(shape)
            or any(type(n) is not int for n in program["concat_shape"])
            or tuple(program["concat_shape"]) != shape
            or program["transforms"] not in ([], ())
            or len(program["operands"]) != 1):
        return None
    destination, component_shape, maps = layouts[0]
    if (any(type(n) is not int for n in component_shape)
            or tuple(component_shape) != shape or maps not in ([], ())):
        return None
    count = prod(shape)
    operand = program["operands"][0]
    source, offset = operand["source"], operand["offset"]
    if (any(type(n) is not int for n in
            (destination, destination_nbytes, source, offset, operand["nbytes"], operand["extent"]))
            or not 0 <= source < len(sources)
            or not 0 <= destination <= destination_nbytes - count
            or operand["nbytes"] != count or operand["extent"] != shape[axis]):
        return None
    source_nbytes = sources[source]["nbytes"]
    ranges = program["ranges"]
    if (type(source_nbytes) is not int or not 0 <= offset <= source_nbytes - count
            or len(ranges) != 1 or len(ranges[0]) != 2
            or any(type(n) is not int for n in ranges[0])
            or tuple(ranges[0]) != (destination, count)):
        return None
    # maps=[] makes local flat index j the global flat index; no transforms
    # and one full concat operand preserve j. All byte ranges were checked.
    return source, offset, destination, count


def _copy_source_bytes(pool, source, source_offset, destination, count):
    """Keep the file view alive through each bounded, synchronous H2D copy."""
    for begin in range(0, count, SOURCE_COPY_CHUNK_BYTES):
        size = min(SOURCE_COPY_CHUNK_BYTES, count - begin)
        pool.narrow(0, destination + begin, size).copy_(
            source.narrow(0, source_offset + begin, size), non_blocking=False)


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
    def __init__(self, path, offset=0, nbytes=None):
        import numpy as np
        size = Path(path).stat().st_size
        if nbytes is None:
            nbytes = size - offset
        if offset < 0 or offset % mmap.PAGESIZE or nbytes <= 0 or offset + nbytes > size:
            raise ArtifactError("Readonly source registration needs an aligned, nonempty file range.")
        self.offset = offset
        self.array = np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(nbytes,))
        self.registered = False
        import torch
        # ROCm's torch reports device type "cuda" but ships no libcuda; the HIP
        # host-register API mirrors the CUDA one, so route by build. Mapped
        # (0x02) is required before a device pointer can be taken on either.
        if getattr(torch.version, "hip", None):
            self.driver = ctypes.CDLL("libamdhip64.so")
            register = self.driver.hipHostRegister
            get_pointer = self.driver.hipHostGetDevicePointer
            self._unregister = self.driver.hipHostUnregister
            register_flags = 0x01 | 0x02
        else:
            self.driver = ctypes.CDLL("libcuda.so.1")
            register = self.driver.cuMemHostRegister_v2
            get_pointer = self.driver.cuMemHostGetDevicePointer_v2
            self._unregister = self.driver.cuMemHostUnregister
            register_flags = 0x01 | 0x02 | 0x08
        register.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
        get_pointer.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint)
        self._unregister.argtypes = (ctypes.c_void_p,)
        self.address = int(self.array.ctypes.data)
        result = register(self.address, self.array.nbytes, register_flags)
        if result:
            self.array._mmap.close()
            raise ArtifactError(f"GPU cannot register readonly source pages: driver status {result}.")
        self.registered = True
        pointer = ctypes.c_uint64()
        result = get_pointer(ctypes.byref(pointer), self.address, 0)
        if result:
            self.close()
            raise ArtifactError(f"GPU cannot map readonly source pages: driver status {result}.")
        self.device_pointer = pointer.value

    def close(self):
        if not self.registered:
            return
        result = self._unregister(self.address)
        self.registered = False
        self.array._mmap.close()
        if result:
            raise ArtifactError(f"GPU cannot unregister readonly source pages: driver status {result}.")


def _selected_physical_chunks(chunks, ranges):
    """Intersect verified writer chunks with every selected owner component."""
    merged = []
    for start, count in sorted(ranges):
        end = start + count
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        elif count:
            merged.append((start, end))
    position = 0
    for chunk in chunks:
        begin, end = chunk["offset"], chunk["offset"] + chunk["nbytes"]
        while position < len(merged) and merged[position][1] <= begin:
            position += 1
        current = position
        while current < len(merged) and merged[current][0] < end:
            start, stop = max(begin, merged[current][0]), min(end, merged[current][1])
            if start < stop:
                yield {**chunk, "offset": start, "nbytes": stop - start}
            if merged[current][1] > end:
                break
            current += 1
        position = current


def load_source_rdata(root, index, module, *, chip, device, pool=None, buffer_ids=None):
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
    shared = pool is not None
    if shared:
        if (not isinstance(pool, torch.Tensor) or pool.dtype != torch.uint8
                or tuple(pool.shape) != (size,) or pool.device != target or not pool.is_contiguous()):
            raise ArtifactError("Readonly destination must be a matching contiguous byte pool.")
    else:
        pool = torch.zeros(size, dtype=torch.uint8, device=target)
    entries = {entry["buffer"]: entry for entry in index["entries"]}
    selected = set(entries) if buffer_ids is None else set(buffer_ids)
    if selected - set(entries):
        raise ArtifactError("Readonly selection contains an unknown entry.")
    buffers = verify_buffer_plan(module).buffer_map
    programs = [p for p in index["programs"] if p["buffer"] in selected]
    host = torch.from_file(str(root / index["image"]), shared=False, size=index["nbytes"], dtype=torch.uint8)
    if shared or buffer_ids is not None:
        virtual = {p["buffer"] for p in index["programs"]}
        ranges = ((offset, prod(shape)) for name in selected - virtual
                  for offset, shape, _ in component_layouts(buffers[name], chip))
        chunks = _selected_physical_chunks(index["segment_chunks"][chip], ranges)
    else:
        chunks = index["segment_chunks"][chip]
    for chunk in chunks:
        offset, count = chunk["offset"], chunk["nbytes"]
        if chunk.get("zero"):
            if shared:
                pool.narrow(0, offset, count).zero_()
            continue
        # This view owns no model-sized CPU staging storage. copy_ transfers
        # an already materialized range, bounded by the writer's chunk size.
        pool.narrow(0, offset, count).copy_(host.narrow(0, chip * size + offset, count))
    if target.type == "cpu":
        sources = [np.memmap(root / source["path"], dtype=np.uint8, mode="r") for source in index["sources"]]
        output = pool.numpy()
        for program in programs:
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
    source_views = {}
    with torch.cuda.device(target):
        for program in programs:
            layouts = list(component_layouts(buffers[program["buffer"]], chip))
            direct = _contiguous_source_copy(program, layouts, index["sources"], size)
            if direct is not None:
                source, offset, destination, count = direct
                if source not in source_views:
                    descriptor = index["sources"][source]
                    source_views[source] = torch.from_file(
                        str(root / descriptor["path"]), shared=False,
                        size=descriptor["nbytes"], dtype=torch.uint8)
                _copy_source_bytes(pool, source_views[source], offset, destination, count)
                continue
            from triton.flagmega.runtime.source_rdata_kernel import gather_source_bytes
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
            extents = [operand["extent"] for operand in program["operands"]]
            prefix = torch.tensor([0, *np.cumsum(extents).tolist()], dtype=torch.int64, device=target)
            stage_counts = tuple(len(axis) for axis in stages)
            stage_begin = tuple(sum(stage_counts[:axis]) for axis in range(len(stages)))
            flat_stages = tuple(stage for axis in stages for stage in axis)
            transforms = program["transforms"]
            transform_counts = [len(transform["shape"]) for transform in transforms]
            transform_begin = tuple(sum(transform_counts[:i]) for i in range(len(transforms)))
            for transfer in source_transfers(program, layouts, index["sources"], budget=PINNED_SOURCE_LIMIT):
                with ExitStack() as stack:
                    mappings = []
                    for window in transfer.windows:
                        mapping = _RegisteredFile(root / index["sources"][window.source]["path"],
                                                  window.begin, window.end - window.begin)
                        stack.callback(mapping.close)
                        mappings.append(mapping)
                    # All transfers finish before any window is unregistered,
                    # including when allocation or a later launch raises.
                    stack.callback(torch.cuda.synchronize, target)
                    addresses = torch.tensor([
                        [0 if slot is None else mappings[slot].device_pointer - mappings[slot].offset + operand["offset"]
                         for operand, slot in zip(program["operands"], row, strict=True)]
                        for row in transfer.window_indices
                    ], dtype=torch.uint64, device=target)
                    gather_source_bytes[((transfer.count + 1023) // 1024, transfer.owner_end - transfer.owner_begin)](
                        pool, addresses, prefix, offsets[transfer.owner_begin:transfer.owner_end],
                        parameters[transfer.owner_begin:transfer.owner_end], transfer.start, transfer.count,
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
