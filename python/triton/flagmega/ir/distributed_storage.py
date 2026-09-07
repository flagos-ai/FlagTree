# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical interpretations of a distributed tensor's logical coordinates."""

from enum import Enum


class DistributedBufferStorageKind(str, Enum):
    """How one distributed logical buffer is represented in physical memory."""

    COMPACT_LOCAL = "compact_local"
    COMPACT_PER_OWNER = "compact_per_owner"
    CANONICAL_GLOBAL = "canonical_global"
    REPLICATED_LOCAL = "replicated_local"

    @property
    def exposes_logical_coordinates(self) -> bool:
        """Whether one pointer spans the complete logical tensor.

        ``CANONICAL_GLOBAL`` is shared by placement owners.  A
        ``REPLICATED_LOCAL`` pointer has the same coordinate ABI, but names a
        complete replica in a block-scoped runtime pool.  Keeping location
        separate from coordinate representation prevents a local replica
        from being mistaken for an inter-block communication boundary.
        """

        return self in {
            DistributedBufferStorageKind.CANONICAL_GLOBAL,
            DistributedBufferStorageKind.REPLICATED_LOCAL,
        }


__all__ = ["DistributedBufferStorageKind"]
