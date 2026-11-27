# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical tensor-descriptor planning shared by Triton kernel renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.errors import CodegenError


def coalesce_dense_weight_groups(
    groups: Mapping[str, Mapping[str, object]],
    roles: Sequence[str],
    *,
    argument: str,
    block_shape: Sequence[int],
) -> dict[str, object]:
    """Coalesce adjacent ``[count, N, K]`` rdata groups into one 2-D view.

    This is a descriptor-only transformation: it neither changes logical IR
    edges nor materializes a weight transform.  The returned row layout keeps
    enough information for a kernel variant to address each original role.
    Coalescing is legal only when the existing buffer plan already proves one
    dense, byte-adjacent region with a common dtype and K extent.
    """

    role_names = tuple(str(role) for role in roles)
    if not role_names:
        raise CodegenError("A coalesced descriptor requires at least one rdata role.")
    try:
        members = tuple(groups[role] for role in role_names)
    except KeyError as error:
        raise CodegenError(
            f"A coalesced descriptor references unknown rdata role {error.args[0]!r}."
        ) from error

    shapes = tuple(tuple(int(value) for value in member["shape"]) for member in members)
    if any(len(shape) != 2 for shape in shapes):
        raise CodegenError("Coalesced descriptor members must all be rank-2 tensors.")
    k_extent = shapes[0][1]
    dtype = str(members[0]["dtype"])
    count = int(members[0]["count"])
    if any(shape[1] != k_extent for shape in shapes):
        raise CodegenError("Coalesced descriptor members must have the same K extent.")
    if any(str(member["dtype"]) != dtype for member in members):
        raise CodegenError("Coalesced descriptor members must have the same dtype.")
    if any(int(member["count"]) != count for member in members):
        raise CodegenError("Coalesced descriptor members must have the same repeat count.")

    first_offset = int(members[0]["offset"])
    cursor = first_offset
    row_cursor = 0
    row_layout: dict[str, dict[str, int]] = {}
    for role, member, shape in zip(role_names, members, shapes):
        member_nbytes = int(member["member_nbytes"])
        stride = int(member["stride"])
        if stride != member_nbytes:
            raise CodegenError(
                f"Coalesced descriptor role {role!r} is not a dense repeated rdata group."
            )
        if int(member["offset"]) != cursor:
            raise CodegenError(
                f"Coalesced descriptor role {role!r} is not byte-adjacent to its predecessor."
            )
        row_layout[role] = {
            "base_row": row_cursor,
            "rows_per_member": shape[0],
        }
        role_nbytes = count * member_nbytes
        cursor += role_nbytes
        row_cursor += count * shape[0]

    requested_block_shape = tuple(int(value) for value in block_shape)
    if (
        len(requested_block_shape) != 2
        or any(value <= 0 for value in requested_block_shape)
        or requested_block_shape[1] > k_extent
        or k_extent % requested_block_shape[1]
    ):
        raise CodegenError(
            "A coalesced descriptor block shape must be positive rank-2 and tile K exactly."
        )
    return {
        "argument": argument,
        "groups": list(role_names),
        "offset": first_offset,
        "nbytes": cursor - first_offset,
        "dtype": dtype,
        "shape": [row_cursor, k_extent],
        "rows_per_layer": 0,
        "block_shape": list(requested_block_shape),
        "row_layout": row_layout,
    }


__all__ = ["coalesce_dense_weight_groups"]
