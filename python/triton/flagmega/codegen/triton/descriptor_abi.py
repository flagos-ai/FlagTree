# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Device tensor-map storage ABI, independent of descriptor coordinates."""

from math import prod

from triton.flagmega.errors import CodegenError
from .tensor_descriptor_planner import _DTYPE_ITEM_SIZES


def device_descriptor_request(request, workspaces):
    """Keep the source view intact; encode a map matching the typed Shared tile.

    Passing a tensor map by value makes its parameter-space address an SSA
    value spanning the entry dispatcher loop. Many reusable calls then keep
    all of those addresses live. A device handle is an ordinary pointer kernel
    argument that can be loaded at its call site without cloning the callee.

    `kind` still describes coordinate semantics: single maps use global
    coordinates; owner tables rebase them. `storage` describes only the ABI.
    """
    if request["kind"] != "single":
        return request
    block_shape = tuple(int(value) for value in request["block_shape"])
    item_size = _DTYPE_ITEM_SIZES[str(request["dtype"])]
    encodings = set()
    for workspace in workspaces:
        shared_shape = tuple(int(value) for value in workspace["shape"])
        if shared_shape[1:] != block_shape:
            continue
        width = 0
        if workspace["matrix_compatible"] and prod(shared_shape[:-1]) >= 8:
            contiguous_bytes = shared_shape[-1] * item_size
            width = next((value for value in (128, 64, 32)
                          if contiguous_bytes % value == 0), 0)
        encodings.add(width)
    if len(encodings) != 1:
        raise CodegenError(
            f"Descriptor {request['parameter']!r} must match one typed Shared "
            f"encoding, found {sorted(encodings)} for tile {block_shape}."
        )
    width, = encodings
    # Mirror getTMABlockShape: hardware boxes are at most 256 per dimension;
    # the contiguous box dimension matches the NVMMA swizzle width. TLE emits
    # multiple TMA instructions for a logical tile larger than its box.
    box_shape = tuple(min(value, 256) for value in block_shape)
    if width:
        box_shape = (*box_shape[:-1], width // item_size)
    return {
        **request, "storage": "device", "box_shape": box_shape,
        "swizzle_mode": {0: 0, 32: 1, 64: 2, 128: 3}[width],
    }


__all__ = ["device_descriptor_request"]
