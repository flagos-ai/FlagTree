"""Explicit HCU packet transfers for Gluon kernels.

These operations describe producer and consumer layouts; the HCU compiler
validates their packet layout, bounds, ownership, and synchronization.
"""
from .packet import (
    WholePacketLayout,
    PacketPrefixLayout,
    PacketStorageLayout,
    PacketToken,
    PacketTokenType,
    whole_packet_owner,
    packet_copy,
    packet_read,
    packet_commit,
    packet_wait,
    packet_sched_barrier,
    packet_buffer_store,
)

__all__ = [
    "WholePacketLayout", "PacketPrefixLayout", "PacketStorageLayout", "PacketToken", "PacketTokenType",
    "whole_packet_owner", "packet_copy", "packet_read", "packet_commit",
    "packet_wait", "packet_sched_barrier", "packet_buffer_store",
]

from .register_load import structured_register_load
from .raw_window import (
    raw_window_owner, raw_window_copy, raw_window_read, raw_window_wait,
    raw_warp_group_id,
)
__all__ += [
    "structured_register_load", "raw_window_owner", "raw_window_copy",
    "raw_window_read", "raw_window_wait", "raw_warp_group_id",
]

# Versioned source capability: packet_read accepts distributed Linear charts.
# The lowering still proves the entire physical ownership before any matrix read.
WHOLE_PACKET_LINEAR_READ_VERSION = 1
__all__ += ["WHOLE_PACKET_LINEAR_READ_VERSION"]
