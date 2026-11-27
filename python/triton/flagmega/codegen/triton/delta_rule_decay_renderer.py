# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Block log-decay scan on local heads, with per-buffer storage mapping."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node
from triton.flagmega.ir.ops.nn.delta_rule_log_prefix import DeltaRuleLogPrefix


def delta_rule_log_prefix_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _canonical_writer_active

    alpha = _buffer(raw, "inputs", "alpha")
    result = _buffer(raw, "outputs", "result")
    attrs = DeltaRuleLogPrefix.normalize_attrs(raw["semantic_attrs"])
    source = Node("alpha", "builtin.var", (), _tensor_type(alpha["abi"]))
    if DeltaRuleLogPrefix.infer_type((source, ), attrs) != _tensor_type(result["abi"]):
        raise CodegenError("DeltaRuleLogPrefix result ABI disagrees with its block/head contract.")
    shape = alpha["abi"]["local_capacity_shape"]
    output_shape = result["abi"]["local_capacity_shape"]
    if any(not isinstance(value, int) for value in (*shape, *output_shape)):
        raise CodegenError("DeltaRuleLogPrefix grouped implementation requires static local extents.")
    return {
        "alpha": _pointer(alpha),
        "result": _pointer(result),
        "tokens": shape[0],
        "heads": shape[1],
        "blocks": output_shape[0],
        "block_size": attrs["block_size"],
        "scan_group_size": attrs["scan_group_size"],
        "scan_levels": attrs["scan_group_size"].bit_length() - 1,
        "epsilon": repr(attrs["epsilon"]),
        "fast_log2": attrs["log2_mode"] == "fast",
        "alpha_offset": emit_local_scalar_offset(alpha["abi"], ("_fm_token", "_fm_head")),
        "result_offset": emit_local_scalar_offset(result["abi"], ("_fm_block", "_fm_head", "_fm_row")),
        "writer_active": _canonical_writer_active(result["abi"]),
    }
