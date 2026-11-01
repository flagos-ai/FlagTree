# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical addressing for explicit CP transfer and fixup stages."""

from triton.flagmega.codegen.triton.physical_access import emit_active_extent, emit_local_scalar_offset, emit_logical_coordinate
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.codegen.triton.delta_rule_block_renderer import state_matrix_access, _replicated_result_pointers
from triton.flagmega.ir import Node
from triton.flagmega.ir.ops.nn.delta_rule_chunk_transfer import DeltaRuleChunkTransfer, DeltaRuleChunkFixup
from triton.flagmega.errors import CodegenError


def delta_rule_chunk_transfer_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _distributed_unique_writer_active
    names = ("key", "value", "coefficients", "log_prefix")
    bindings = tuple(_buffer(raw, "inputs", name) for name in names)
    result = _buffer(raw, "outputs", "result")
    attrs = DeltaRuleChunkTransfer.normalize_attrs(raw["semantic_attrs"])
    nodes = tuple(Node(name, "builtin.var", (), _tensor_type(b["abi"])) for name, b in zip(names, bindings))
    if DeltaRuleChunkTransfer.infer_type(nodes, attrs) != _tensor_type(result["abi"]):
        raise CodegenError("Chunk-transfer output does not match its grouped-head/state contract.")
    key, value, coefficients, prefix = bindings
    kshape, vshape = (b["abi"]["local_capacity_shape"] for b in (key, value))
    if any(not isinstance(d, int) for d in (*kshape, *vshape)):
        raise CodegenError("Chunk transfer requires static local geometry.")
    return {
        **{name: _pointer(b) for name, b in zip(names, bindings)},
        "result_pointers": _replicated_result_pointers(result),
        "tokens": vshape[0], "heads": vshape[1], "chunk_blocks": attrs["chunk_size"] // 64,
        "chunks": result["abi"]["local_capacity_shape"][0], "matrix_kind": attrs["matrix_kind"],
        "key_offset": emit_local_scalar_offset(key["abi"], ("_fm_token[:, None]", f"_fm_head // {vshape[1] // kshape[1]}", "_fm_dim[None, :]")),
        "value_offset": emit_local_scalar_offset(value["abi"], ("_fm_token[None, :]", "_fm_head", "_fm_dim[:, None]")),
        "coefficients_offset": emit_local_scalar_offset(coefficients["abi"], ("_fm_block", "_fm_head", "_fm_row[:, None]", "_fm_row[None, :]")),
        "prefix_offset": emit_local_scalar_offset(prefix["abi"], ("_fm_block", "_fm_head", "_fm_row")),
        "result_offset": emit_local_scalar_offset(result["abi"], ("_fm_chunk", "_fm_head", "_fm_dim[:, None]", "_fm_dim[None, :]")),
        "active_heads": emit_active_extent(value["abi"], 1),
        "writer_active": _distributed_unique_writer_active(value["abi"]),
    }


def delta_rule_chunk_fixup_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _distributed_unique_writer_active
    transfer, local = (_buffer(raw, "inputs", n) for n in ("transfer", "local_state"))
    result = _buffer(raw, "outputs", "result")
    attrs = DeltaRuleChunkFixup.normalize_attrs(raw["semantic_attrs"])
    shape = result["abi"]["local_capacity_shape"]
    if any(not isinstance(d, int) for d in shape) or tuple(shape[-2:]) != (128, 128):
        raise CodegenError("Chunk fixup requires static 128-wide state matrices.")
    head = emit_logical_coordinate(result["abi"], 1, ("0", "_fm_head", "0", "0"))
    state, offset = state_matrix_access(raw, attrs, _tensor_type(result["abi"]), head=head,
                                        row="_fm_state_row[:, None]", column="_fm_dim[None, :]")
    return {
        "transfer": _pointer(transfer), "local_state": _pointer(local), "state": state,
        "state_offset": offset, "result_pointers": _replicated_result_pointers(result),
        "chunks": shape[0], "heads": shape[1], "active_heads": emit_active_extent(result["abi"], 1),
        "transfer_offset": emit_local_scalar_offset(transfer["abi"], ("_fm_chunk", "_fm_head", "_fm_dim[:, None]", "_fm_dim[None, :]")),
        "local_offset": emit_local_scalar_offset(local["abi"], ("_fm_chunk", "_fm_head", "_fm_state_row[:, None]", "_fm_dim[None, :]")),
        "result_offset": emit_local_scalar_offset(result["abi"], ("_fm_chunk", "_fm_head", "_fm_state_row[:, None]", "_fm_dim[None, :]")),
        "writer_active": _distributed_unique_writer_active(result["abi"]),
    }
