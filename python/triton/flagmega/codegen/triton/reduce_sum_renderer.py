# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FP32 sum over local axes, retaining distributed partial components."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset, emit_scalar_immediate
from triton.flagmega.codegen.triton.reduction_domain import local_reduction_domain
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.ops.math.reduce_sum import ReduceSum


def reduce_sum_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _canonical_writer_active

    source = _buffer(raw, "inputs", "value")
    result = _buffer(raw, "outputs", "result")
    input_type = _tensor_type(source["abi"])
    attrs = raw["semantic_attrs"]
    if ReduceSum.infer_type((Node("source", "builtin.var", (), input_type), ), attrs) != _tensor_type(result["abi"]):
        raise CodegenError("ReduceSum result ABI disagrees with its local/partial reduction contract.")
    rank = len(source["abi"]["local_capacity_shape"])
    axes = tuple(normalize_axis(axis, rank) for axis in attrs["axes"])
    domain = local_reduction_domain(source["abi"], axes, int(raw["parameters"]["elements_per_program"]))
    output_coordinates = (tuple("0" if axis in axes else coordinate
                                for axis, coordinate in enumerate(domain["coordinates"]))
                          if attrs["keep_dims"] else domain["outer_coordinates"])
    scalar = source["abi"]["storage"] == "scalar"
    scalar_value = None
    if scalar:
        scalar_value = (emit_scalar_immediate(source["abi"], source["runtime_argument"])
                        if source["runtime_value_kind"] == "immediate" else source["runtime_argument"])
    return {
        **domain,
        "identity": not axes,
        "scalar_value": scalar_value,
        "source": None if scalar else _pointer(source),
        "source_offset": emit_local_scalar_offset(source["abi"], domain["coordinates"]),
        "result": _pointer(result),
        "result_offset": emit_local_scalar_offset(result["abi"], output_coordinates),
        "writer_active": _canonical_writer_active(result["abi"]),
    }
