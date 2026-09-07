# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small real-IR producers for distributed Boxing kernel tests."""

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler


def partial_sum_module(values=1, partial_axes=(0, 1), *, use_mean=False):
    placement = fm.Placement((8, 16), "yx", "bb")
    preserved_axes = tuple(axis for axis in range(2) if axis not in partial_axes)
    owners = 1
    for axis in partial_axes:
        owners *= placement.hierarchy[axis]
    value_type = fm.tensor_type("float32", (values, owners))
    outer = (
        fm.SBP.split_block_cyclic(preserved_axes, 1)
        if preserved_axes else fm.SBP.broadcast()
    )
    distributed = fm.DistributedType(
        value_type, (outer, fm.SBP.split_contiguous(partial_axes, 1)), placement,
    )

    class PartialSum(fm.Module):
        def forward(self):
            source = self.input("source", value_type)
            local = fm.F.distributed.force_boxing(source, distributed, name="load")
            stats = fm.F.nn.norm_stats(local, axis=1, use_mean=use_mean, name="stats")
            result = fm.F.distributed.force_boxing(
                stats, fm.logical_type(stats.type), name="reduce",
            )
            self.function("main", (source,), (result,))

    module = PartialSum(
        dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}},
    ).build()
    return Compiler().compile(module).module, owners
