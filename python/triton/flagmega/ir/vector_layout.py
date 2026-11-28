# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Vector packet geometry, distinct from logical tensor extents."""

from triton.flagmega.errors import IRSchemaError


def split_vector_lanes(lanes, axes, *, element_bytes, vector_bytes):
    """Split the innermost lane into packets without changing any axis product.

    For a 16-byte packet, eight half elements widened to F32 become (2, 4),
    both on the same logical axis. Outer tensor shape and split units stay fixed.
    The packet width comes from the vectorization/operand contract, not a vendor.
    """
    lanes, axes = tuple(lanes), tuple(axes)
    if len(lanes) != len(axes):
        raise IRSchemaError("Vector lanes require one logical axis per component.")
    while (len(lanes) > 1 and axes[-1] == axes[-2] and lanes[-1] * lanes[-2] * element_bytes <= vector_bytes):
        lanes, axes = (*lanes[:-2], lanes[-2] * lanes[-1]), axes[:-1]
    if not lanes or lanes[-1] * element_bytes <= vector_bytes:
        return lanes, axes
    inner, remainder = divmod(vector_bytes, element_bytes)
    if remainder or inner <= 0 or lanes[-1] % inner:
        raise IRSchemaError("Vector packet cannot preserve an integral lane partition.")
    return (*lanes[:-1], lanes[-1] // inner, inner), (*axes, axes[-1])
