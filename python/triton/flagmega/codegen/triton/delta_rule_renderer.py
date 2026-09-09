# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Local head shards with independent operand physical-address mappings."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node
from triton.flagmega.ir.ops.nn.delta_rule_coefficients import DeltaRuleCoefficients


def delta_rule_coefficients_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _canonical_writer_active

    key, beta = (_buffer(raw, "inputs", name) for name in ("key", "beta"))
    result = _buffer(raw, "outputs", "result")
    inputs = tuple(
        Node(name, "builtin.var", (), _tensor_type(value["abi"])) for name, value in (("key", key), ("beta", beta)))
    attrs = DeltaRuleCoefficients.normalize_attrs(raw["semantic_attrs"])
    if DeltaRuleCoefficients.infer_type(inputs, attrs) != _tensor_type(result["abi"]):
        raise CodegenError("DeltaRuleCoefficients result ABI does not match its head/chunk contract.")
    shape = key["abi"]["local_capacity_shape"]
    beta_shape = beta["abi"]["local_capacity_shape"]
    output_shape = result["abi"]["local_capacity_shape"]
    if any(not isinstance(value, int) for value in (*shape, *beta_shape, *output_shape)):
        raise CodegenError("DeltaRuleCoefficients blockwise implementation requires static local extents.")
    tokens, key_heads, dimension = shape
    heads = beta_shape[1]
    if key_heads <= 0 or heads % key_heads:
        raise CodegenError("DeltaRuleCoefficients local grouped head ownership is not aligned.")
    return {
        "key":
        _pointer(key),
        "beta":
        _pointer(beta),
        "result":
        _pointer(result),
        "tokens":
        tokens,
        "heads":
        heads,
        "blocks":
        output_shape[0],
        "block_size":
        attrs["block_size"],
        "key_dim":
        dimension,
        "key_tile":
        max(16, 1 << (dimension - 1).bit_length()),
        "repeats":
        heads // key_heads,
        "key_offset":
        emit_local_scalar_offset(
            key["abi"], ("_fm_tokens[:, None]", "_fm_head // " + str(heads // key_heads), "_fm_feature[None, :]")),
        "beta_offset":
        emit_local_scalar_offset(beta["abi"], ("_fm_tokens", "_fm_head")),
        "result_offset":
        emit_local_scalar_offset(result["abi"], ("_fm_block", "_fm_head", "_fm_row[:, None]", "_fm_col[None, :]")),
        "writer_active":
        _canonical_writer_active(result["abi"]),
    }
