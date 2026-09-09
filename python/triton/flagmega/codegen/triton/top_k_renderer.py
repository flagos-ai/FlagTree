# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Streaming stable TopK without integer-to-float casts or sentinel deletion."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.reduction_domain import local_reduction_domain
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node, TupleType
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.ops.tensors.top_k import TopK


def top_k_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _canonical_writer_active

    source = _buffer(raw, "inputs", "value")
    values = _buffer(raw, "outputs", "result_0")
    indices = _buffer(raw, "outputs", "result_1")
    attrs = raw["semantic_attrs"]
    result_type = TupleType((_tensor_type(values["abi"]), _tensor_type(indices["abi"])))
    if TopK.infer_type((Node("source", "builtin.var", (), _tensor_type(source["abi"])), ), attrs) != result_type:
        raise CodegenError("TopK tuple ABI disagrees with its materialized selection-axis contract.")
    axis = normalize_axis(attrs["axis"], len(source["abi"]["local_capacity_shape"]))
    domain = local_reduction_domain(source["abi"], (axis, ), int(raw["parameters"]["elements_per_program"]))
    floating = source["abi"]["scalar_dtype"] in {"float32", "bfloat16"}
    scalar_type = "tl.float32" if floating else f"tl.{source['abi']['scalar_dtype']}"
    if floating:
        sentinel = '-float("inf")' if attrs["largest"] else 'float("inf")'
    else:
        bits = int(source["abi"]["scalar_itemsize"]) * 8
        sentinel = str(-(2**(bits - 1)) if attrs["largest"] else 2**(bits - 1) - 1)
    winner_coordinates = list(domain["coordinates"])
    winner_coordinates[axis] = "_fm_winner_index"
    output_coordinates = list(domain["coordinates"])
    output_coordinates[axis] = "_fm_selection"
    return {
        **domain,
        "k": attrs["k"],
        "largest": attrs["largest"],
        "scalar_type": scalar_type,
        "sentinel": sentinel,
        "source": _pointer(source),
        "values_pointer": _pointer(values),
        "indices_pointer": _pointer(indices),
        "source_offset": emit_local_scalar_offset(source["abi"], domain["coordinates"]),
        "winner_offset": emit_local_scalar_offset(source["abi"], winner_coordinates),
        "values_offset": emit_local_scalar_offset(values["abi"], output_coordinates),
        "indices_offset": emit_local_scalar_offset(indices["abi"], output_coordinates),
        "values_active": _canonical_writer_active(values["abi"]),
        "indices_active": _canonical_writer_active(indices["abi"]),
    }
