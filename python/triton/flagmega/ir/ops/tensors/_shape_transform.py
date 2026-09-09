# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Distribution preservation for shape changes on broadcast axes."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import DistributedType, SBP


def preserve_unchanged_axes(source, output):
    if not isinstance(source, DistributedType):
        return output
    for axis, (before, after) in enumerate(zip(source.tensor.shape, output.shape, strict=True)):
        if before != after and source.axis_policies[axis] != SBP.broadcast():
            raise IRSchemaError("Shape transforms on split axes require explicit resharding.")
    return DistributedType(output, source.axis_policies, source.placement, source.partial)
