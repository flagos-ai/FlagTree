# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""``_conflicts`` short-circuits on byte overlap without changing its answer.

Planning barriers is quadratic in the number of accesses held in history, so
the hazard predicate runs tens of millions of times on the 45-layer model.
Testing the cheap, highly selective byte-overlap condition first is only valid
if the predicate stays a pure conjunction: these tests enumerate the whole
mode/overlap/partition cross product and compare against the independent
reference formulation.
"""

from itertools import product

import pytest

from triton.flagmega.ir.bufferization import MemorySharingScope
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.memory_effect import MemoryAccessMode, MemoryEffect
from triton.flagmega.passes.tir.bufferize.synchronization import (
    _Access,
    _conflicts,
    _partitions_may_alias,
)


def _access(*, buffer, storage, physical_id, offset, nbytes, effect, partition):
    mode = ("read" if effect.physical_mode is MemoryAccessMode.READ else
            "write" if effect.physical_mode is MemoryAccessMode.WRITE else "read_write")
    return _Access(
        node=f"{buffer}.node",
        buffer=buffer,
        storage=storage,
        physical_id=physical_id,
        offset=offset,
        nbytes=nbytes,
        mode=mode,
        distributed_type=None,
        distributed_storage_kind=DistributedBufferStorageKind.REPLICATED_LOCAL,
        effect=effect,
        access_partition=partition,
        is_reference=False,
        requires_full_chip=False,
        sharing_scope=MemorySharingScope.BLOCK,
    )


def _reference_conflicts(previous, current):
    """The predicate as originally written: effects first, overlap last."""

    producer_reads = bool(previous.effect.physical_mode & MemoryAccessMode.READ)
    producer_writes = bool(previous.effect.physical_mode & MemoryAccessMode.WRITE)
    consumer_reads = bool(current.effect.physical_mode & MemoryAccessMode.READ)
    consumer_writes = bool(current.effect.physical_mode & MemoryAccessMode.WRITE)
    requires_synchronization = (
        (producer_writes and consumer_reads)
        or (producer_reads and consumer_writes)
        or (producer_writes and consumer_writes and previous.buffer != current.buffer))
    if not requires_synchronization or not _partitions_may_alias(
            previous.access_partition, current.access_partition):
        return False
    return (previous.storage == current.storage
            and previous.physical_id == current.physical_id
            and previous.offset < current.offset + current.nbytes
            and current.offset < previous.offset + previous.nbytes)


_EFFECTS = (MemoryEffect.READ, MemoryEffect.WRITE, MemoryEffect.READ_WRITE, MemoryEffect.NONE)
_PLACEMENTS = ((0, 16), (8, 16), (16, 16), (24, 16))
_PARTITIONS = (("whole", None), ("shard", 0), ("shard", 1))


@pytest.mark.parametrize("previous_effect,current_effect", tuple(product(_EFFECTS, _EFFECTS)))
def test_reordered_predicate_matches_the_reference_formulation(previous_effect, current_effect):
    for (previous_offset, previous_nbytes), (current_offset, current_nbytes) in product(
            _PLACEMENTS, _PLACEMENTS):
        for previous_partition, current_partition in product(_PARTITIONS, _PARTITIONS):
            for same_buffer, same_physical, same_storage in product((True, False), repeat=3):
                previous = _access(
                    buffer="a", storage="workspace", physical_id="p0",
                    offset=previous_offset, nbytes=previous_nbytes,
                    effect=previous_effect, partition=previous_partition)
                current = _access(
                    buffer="a" if same_buffer else "b",
                    storage="workspace" if same_storage else "rdata",
                    physical_id="p0" if same_physical else "p1",
                    offset=current_offset, nbytes=current_nbytes,
                    effect=current_effect, partition=current_partition)
                assert _conflicts(previous, current) == _reference_conflicts(previous, current), (
                    previous, current)


def test_disjoint_byte_ranges_never_conflict_whatever_the_effects():
    previous = _access(buffer="a", storage="workspace", physical_id="p0", offset=0, nbytes=8,
                       effect=MemoryEffect.WRITE, partition=("whole", None))
    current = _access(buffer="b", storage="workspace", physical_id="p0", offset=8, nbytes=8,
                      effect=MemoryEffect.READ, partition=("whole", None))
    assert not _conflicts(previous, current)


def test_overlapping_write_then_read_still_conflicts():
    previous = _access(buffer="a", storage="workspace", physical_id="p0", offset=0, nbytes=16,
                       effect=MemoryEffect.WRITE, partition=("whole", None))
    current = _access(buffer="b", storage="workspace", physical_id="p0", offset=8, nbytes=16,
                      effect=MemoryEffect.READ, partition=("whole", None))
    assert _conflicts(previous, current)
