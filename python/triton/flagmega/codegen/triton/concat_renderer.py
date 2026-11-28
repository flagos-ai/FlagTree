# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Variadic local-shard Concat, with explicit offsets for each input."""

from math import prod

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.concat import Concat


def concat_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer,
        _pointer,
        _scalar_local_domain,
        _canonical_writer_active,
    )

    sources = tuple(parameter["buffers"][0] for parameter in raw["inputs"])
    result = _buffer(raw, "outputs", "result")
    result_type = _tensor_type(result["abi"])
    input_types = tuple(_tensor_type(source["abi"]) for source in sources)
    inputs = tuple(Node(str(index), "builtin.var", (), value) for index, value in enumerate(input_types))
    if Concat.infer_type(inputs, raw["semantic_attrs"]) != result_type:
        raise CodegenError("Concat call ABI does not match its inferred local-shard contract.")
    axis = normalize_axis(raw["semantic_attrs"]["axis"], tensor_of(result_type).rank)
    entries = []
    output_start = 0
    for source, value_type in zip(sources, input_types):
        abi = source["abi"]
        extent = tensor_of(value_type).shape[axis].fixed_value
        if prod(abi["local_capacity_shape"]):
            domain = _scalar_local_domain(abi, "_fm_offsets")
            output_coordinates = list(domain["local_coordinates"])
            output_coordinates[axis] = f"(({output_coordinates[axis]}) + {output_start})"
            entries.append({
                "source":
                _pointer(source),
                "capacity":
                domain["capacity"],
                "active":
                domain["active"],
                "input_offset":
                emit_local_scalar_offset(abi, domain["local_coordinates"], lane_coordinate=domain["lane_coordinate"]),
                "output_offset":
                emit_local_scalar_offset(result["abi"], output_coordinates, lane_coordinate=domain["lane_coordinate"]),
            })
        output_start += extent
    return {
        "entries": entries, "result": _pointer(result), "tile": int(raw["parameters"]["elements_per_program"]),
        "writer_active": _canonical_writer_active(result["abi"])
    }
