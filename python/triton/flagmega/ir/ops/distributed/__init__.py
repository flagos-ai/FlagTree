# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit distributed op registration."""

from triton.flagmega.ir.ops.distributed.boxing import Boxing
from triton.flagmega.ir.ops.distributed.force_boxing import ForceBoxing
from triton.flagmega.ir.ops.distributed.materialize_local_shards import (
    MaterializeLocalShards,
)
from triton.flagmega.ir.ops.distributed.sharded_view import ShardedView

__all__ = [
    "Boxing",
    "ForceBoxing",
    "MaterializeLocalShards",
    "ShardedView",
]
