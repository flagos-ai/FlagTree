# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bound the lifetime and size of readonly host registrations by byte geometry."""
import mmap
from dataclasses import dataclass
from math import prod

from triton.flagmega.errors import ArtifactError


@dataclass(frozen=True)
class SourceWindow:
    source: int
    begin: int
    end: int


@dataclass(frozen=True)
class SourceTransfer:
    owner_begin: int
    owner_end: int
    start: int
    count: int
    windows: tuple[SourceWindow, ...]
    # Each row is an owner, each column a concat operand; None means no read.
    window_indices: tuple[tuple[int | None, ...], ...]


def _digit_bounds(low, high, stride, capacity):
    """Conservative coordinate interval for a row-major flat interval."""
    first, last = low // stride, high // stride
    if first // capacity == last // capacity:
        return first % capacity, last % capacity
    return 0, capacity - 1


def _flat_bounds(program, shape, maps, start, count):
    coords = []
    for axis, capacity in enumerate(shape):
        low, high = _digit_bounds(start, start + count - 1, prod(shape[axis + 1:]), capacity)
        if axis < len(maps):
            active, stages = maps[axis]
            high = min(high, active - 1)
            if low > high:
                return None
            for index, block, shards, local_capacity in stages:
                low, high = (_map_coordinate(coord, index, block, shards, local_capacity) for coord in (low, high))
        coords.append((low, high))
    strides = [prod(program["shape"][axis + 1:]) for axis in range(len(shape))]
    low = sum(bounds[0] * stride for bounds, stride in zip(coords, strides, strict=True))
    high = sum(bounds[1] * stride for bounds, stride in zip(coords, strides, strict=True))
    for step, transform in enumerate(program["transforms"]):
        dims = transform["shape"]
        if step != 0 or tuple(dims) != tuple(program["shape"]):
            coords = [_digit_bounds(low, high, prod(dims[axis + 1:]), capacity)
                      for axis, capacity in enumerate(dims)]
        terms = [(a * stride, b * stride) for (a, b), stride in
                 zip(coords, transform["source_strides"], strict=True)]
        low = sum(min(term) for term in terms)
        high = sum(max(term) for term in terms)
    return low, high


def _operand_bounds(program, shape, maps, start, count):
    bounds = _flat_bounds(program, shape, maps, start, count)
    if bounds is None:
        return [None] * len(program["operands"])
    low, high = bounds
    concat = program["concat_shape"]
    trailing = prod(concat[program["axis"] + 1:])
    row = concat[program["axis"]] * trailing
    prefix = 0
    result = []
    for operand in program["operands"]:
        width = operand["extent"] * trailing
        begin, end = prefix * trailing, prefix * trailing + width
        first_row = low // row + (low % row >= end)
        last_row = high // row - (high % row < begin)
        first = max(low, first_row * row + begin)
        last = min(high, last_row * row + end - 1)
        if first > last:
            result.append(None)
        else:
            source_low = first // row * width + first % row - begin
            source_high = last // row * width + last % row - begin
            if source_low < 0 or source_high >= operand["nbytes"]:
                raise ArtifactError("Readonly source address map exceeds its operand.")
            result.append((operand["source"], operand["offset"] + source_low,
                           operand["offset"] + source_high + 1))
        prefix += operand["extent"]
    return result


def _windows(rows, sources, page):
    for row in rows:
        for value in row:
            if value is not None:
                source, low, high = value
                if not 0 <= low < high <= sources[source]["nbytes"]:
                    raise ArtifactError("Readonly source window is outside its verified file.")
    intervals = sorted({(source, low // page * page,
                         min((high + page - 1) // page * page, sources[source]["nbytes"]))
                        for row in rows for value in row if value is not None
                        for source, low, high in (value,)})
    windows = []
    for source, low, high in intervals:
        if low < 0 or high <= low or high > sources[source]["nbytes"]:
            raise ArtifactError("Readonly source window is outside its verified file.")
        if windows and windows[-1].source == source and low <= windows[-1].end:
            previous = windows.pop()
            windows.append(SourceWindow(source, previous.begin, max(previous.end, high)))
        else:
            windows.append(SourceWindow(source, low, high))
    return tuple(windows)


def source_transfers(program, layouts, sources, *, budget, page=mmap.PAGESIZE):
    """Partition owners/byte intervals until all simultaneously pinned pages fit.

    No source bytes are inspected and no model-sized address tensor is built.
    Bounds may include unused pages. Subdivision terminates at one byte of one
    owner, whose actual read is at most one page. Geometry and the device's
    byte-address function remain unchanged.
    """
    if page <= 0 or budget < page:
        raise ArtifactError("Readonly source pin budget must cover at least one host page.")
    shape = tuple(layouts[0][1])

    def partition(owner_begin, owner_end, start, count):
        rows = [_operand_bounds(program, shape, layout[2], start, count)
                for layout in layouts[owner_begin:owner_end]]
        windows = _windows(rows, sources, page)
        pinned = sum((window.end - window.begin + page - 1) // page * page for window in windows)
        if pinned > budget:
            # Keep cooperating owners together: their disjoint logical
            # components can share source pages. Splitting owners first
            # registers the same physical matrix repeatedly for strided views.
            if count > 1:
                first = count // 2
                yield from partition(owner_begin, owner_end, start, first)
                yield from partition(owner_begin, owner_end, start + first, count - first)
            elif owner_end - owner_begin > 1:
                middle = (owner_begin + owner_end) // 2
                yield from partition(owner_begin, middle, start, count)
                yield from partition(middle, owner_end, start, count)
            else:
                raise ArtifactError("Readonly source single-byte pin geometry exceeds its budget.")
            return
        indices = tuple(tuple(None if value is None else next(
            i for i, window in enumerate(windows)
            if window.source == value[0] and window.begin <= value[1] and window.end >= value[2]
        ) for value in row) for row in rows)
        yield SourceTransfer(owner_begin, owner_end, start, count, windows, indices)

    yield from partition(0, len(layouts), 0, prod(shape))
def _map_coordinate(coord, index, block, shards, capacity):
    if block == 0:
        return coord + index * capacity
    return coord // block * shards * block + index * block + coord % block
