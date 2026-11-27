# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Block updates on local heads and value tiles, with named state storage."""

from itertools import product
from math import prod

from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent,
    emit_buffer_pointer,
    emit_local_scalar_offset,
    emit_logical_coordinate,
)
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.ops.nn.delta_rule_block_update import DeltaRuleBlockUpdate, state_field_layout, validate_block_tensors


def delta_rule_block_update_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _buffers, _pointer, _distributed_unique_writer_active

    names = ("query", "key", "value", "coefficients", "log_prefix")
    bindings = tuple(_buffer(raw, "inputs", name) for name in names)
    query, key, value, coefficients, prefix = bindings
    types = tuple(_tensor_type(binding["abi"]) for binding in bindings)
    chunk_size = raw["semantic_attrs"].get("chunk_size")
    attrs = DeltaRuleBlockUpdate.normalize_attrs({k: v for k, v in raw["semantic_attrs"].items() if k != "chunk_size"})
    validate_block_tensors(types, attrs)
    result = _buffer(raw, "outputs", "result_0")
    if _tensor_type(result["abi"]) != types[2]:
        raise CodegenError("DeltaRuleBlockUpdate result ABI must preserve the value type.")
    state_fields = tuple(binding for binding in _buffers(raw, "inputs", "state")
                         if binding["formal"] == "state." + attrs["state_field"])
    if len(state_fields) != 1:
        raise CodegenError("DeltaRuleBlockUpdate requires one binding for its named state field.")
    state = state_fields[0]
    from triton.flagmega.ir.distributed_inference import tensor_of
    spec = state_field_layout(_tensor_type(state["abi"]), attrs, tensor_of(types[2]), tensor_of(types[1]))
    qshape = query["abi"]["local_capacity_shape"]
    vshape = value["abi"]["local_capacity_shape"]
    cshape = coefficients["abi"]["local_capacity_shape"]
    if any(not isinstance(size, int) for size in (*qshape, *vshape, *cshape)):
        raise CodegenError("DeltaRuleBlockUpdate implementation requires static local extents.")
    if vshape[1] % qshape[1]:
        raise CodegenError("DeltaRuleBlockUpdate local grouped heads are not aligned.")
    head = emit_logical_coordinate(value["abi"], 1, ("0", "_fm_head", "0"))
    logical = {"layer": "0", "head": head, "value": "_fm_value_row[:, None]", "key": "_fm_key_col[None, :]"}
    coordinates = [logical[axis] for axis in spec.axes]
    remaining = [
        prod(lane
             for position, lane in zip(spec.vector_axes, spec.lanes)
             if position == axis)
        for axis in range(len(spec.axes))
    ]
    lane_terms = []
    for lane_index, (axis, lane) in enumerate(zip(spec.vector_axes, spec.lanes)):
        remaining[axis] //= lane
        lane_terms.append(
            f"((({coordinates[axis]}) // {remaining[axis]}) % {lane}) * {prod(spec.lanes[lane_index + 1:])}")
    packed_coordinates = [
        f"({coordinate}) // {prod(lane for position, lane in zip(spec.vector_axes, spec.lanes) if position == axis)}"
        for axis, coordinate in enumerate(coordinates)
    ]
    state_offset = emit_local_scalar_offset(state["abi"], packed_coordinates,
                                            lane_coordinate=" + ".join(lane_terms) if lane_terms else None)
    qcoordinates = ("_fm_token[:, None]", "_fm_head // " + str(vshape[1] // qshape[1]), "_fm_key_col[None, :]")
    vcoordinates = ("_fm_token[None, :]", "_fm_head", "_fm_value_row[:, None]")
    rendered = {
        **{name: _pointer(binding)
           for name, binding in zip(names, bindings)},
        "state":
        _pointer(state),
        "state_offset":
        state_offset,
        "result_pointers":
        _replicated_result_pointers(result),
        "result_offset":
        emit_local_scalar_offset(result["abi"], vcoordinates),
        "query_offset":
        emit_local_scalar_offset(query["abi"], qcoordinates),
        "key_offset":
        emit_local_scalar_offset(key["abi"], qcoordinates),
        "value_offset":
        emit_local_scalar_offset(value["abi"], vcoordinates),
        "coefficients_offset":
        emit_local_scalar_offset(coefficients["abi"],
                                 ("_fm_block", "_fm_head", "_fm_row[:, None]", "_fm_row[None, :]")),
        "prefix_offset":
        emit_local_scalar_offset(prefix["abi"], ("_fm_block", "_fm_head", "_fm_row")),
        "heads":
        vshape[1],
        "tokens":
        vshape[0],
        "key_dim":
        qshape[2],
        "value_dim":
        vshape[2],
        "key_tile":
        max(16, 1 << (qshape[2] - 1).bit_length()),
        "value_tile":
        int(raw["parameters"]["value_tile"]),
        "blocks":
        cshape[0],
        "block_size":
        cshape[2],
        "scale":
        repr(qshape[2]**-0.5 if attrs["scale"] is None else attrs["scale"]),
        "active_heads":
        emit_active_extent(value["abi"], 1),
        "writer_active":
        _distributed_unique_writer_active(value["abi"]),
    }
    if chunk_size is not None:
        fixed = _buffer(raw, "inputs", "fixed_state")
        rendered.update({
            "fixed_state": _pointer(fixed), "chunk_blocks": chunk_size // cshape[2],
            "chunks": (cshape[0] + chunk_size // cshape[2] - 1) // (chunk_size // cshape[2]),
            "fixed_state_offset": emit_local_scalar_offset(fixed["abi"], (
                "_fm_chunk - 1", "_fm_head",
                "_fm_value_row[:, None]", "_fm_key_col[None, :]")),
        })
    return rendered


def state_matrix_access(raw, attrs, heads_type, *, head, row, column):
    """Resolve an explicitly ordered/packed reference state matrix."""
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffers, _pointer
    from triton.flagmega.ir.distributed_inference import tensor_of
    from triton.flagmega.ir import tensor_type
    fields = tuple(binding for binding in _buffers(raw, "inputs", "state")
                   if binding["formal"] == "state." + attrs["state_field"])
    if len(fields) != 1:
        raise CodegenError("Delta-rule state requires exactly one named field binding.")
    state = fields[0]
    logical = tensor_of(heads_type)
    tensor = tensor_type("bfloat16", (1, logical.shape[1], 128))
    spec = state_field_layout(_tensor_type(state["abi"]), attrs, tensor, tensor)
    coordinate = {"layer": "0", "head": head, "value": row, "key": column}
    values = [coordinate[axis] for axis in spec.axes]
    remaining = [prod(lane for position, lane in zip(spec.vector_axes, spec.lanes) if position == axis)
                 for axis in range(len(spec.axes))]
    lanes = []
    for i, (axis, lane) in enumerate(zip(spec.vector_axes, spec.lanes)):
        remaining[axis] //= lane
        lanes.append(f"((({values[axis]}) // {remaining[axis]}) % {lane}) * {prod(spec.lanes[i + 1:])}")
    packed = [f"({value}) // {prod(lane for position, lane in zip(spec.vector_axes, spec.lanes) if position == axis)}"
              for axis, value in enumerate(values)]
    return _pointer(state), emit_local_scalar_offset(state["abi"], packed, lane_coordinate=" + ".join(lanes) if lanes else None)


def _replicated_result_pointers(binding):
    """An elected state owner publishes each required physical output replica.

    The pooled activation arenas are global pointers even for block-local
    sharing scope. Select the replica's arena/component without changing its
    logical shard coordinates. Canonical results need only one publication.
    """
    abi = binding["abi"]
    distributed = abi.get("distributed_type")
    if not distributed or abi["storage_kind"] == "canonical_global":
        return (emit_buffer_pointer(abi, binding["runtime_argument"]), )
    hierarchy = tuple(distributed["placement"]["hierarchy"])
    used = {
        axis
        for policy in distributed["axis_policies"]
        if policy["kind"] == "split" for stage in policy["stages"] for axis in stage["hierarchy_axes"]
    }
    unused = tuple(axis for axis in range(len(hierarchy)) if axis not in used)
    pointers = []
    for replica in product(*(range(hierarchy[axis]) for axis in unused)):
        fixed = dict(zip(unused, replica))
        # Public mesh coordinates use shard_y/shard_x for rank two.
        from triton.flagmega.codegen.triton.kernel_call_renderers import _mesh_coordinate
        coordinates = [
            str(fixed[axis]) if axis in fixed else _mesh_coordinate(axis, len(hierarchy))
            for axis in range(len(hierarchy))
        ]
        owner = " + ".join(
            f"({coordinate}) * {prod(hierarchy[axis + 1:])}" for axis, coordinate in enumerate(coordinates))
        replica_abi = dict(abi)
        if replica_abi.get("pool_scope_stride_bytes"):
            replica_abi["pool_scope_index"] = owner
        pointers.append(emit_buffer_pointer(replica_abi, binding["runtime_argument"], owner_index=owner))
    return tuple(pointers)
