# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stable FP32 softmax over a materialized local axis."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.reduction_domain import local_reduction_domain
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.ops.nn.softmax import Softmax


def softmax_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _canonical_writer_active

    source = _buffer(raw, "inputs", "value")
    result = _buffer(raw, "outputs", "result")
    input_type = _tensor_type(source["abi"])
    from triton.flagmega.codegen.triton.fusion import decode_fusion_attrs, softmax_program
    attrs = decode_fusion_attrs(raw["semantic_attrs"])
    if Softmax.infer_type((Node("source", "builtin.var", (), input_type), ), attrs) != _tensor_type(result["abi"]):
        raise CodegenError("Softmax result ABI disagrees with its materialized axis contract.")
    axis = normalize_axis(attrs["axis"], len(source["abi"]["local_capacity_shape"]))
    domain = local_reduction_domain(source["abi"], (axis, ), int(raw["parameters"]["elements_per_program"]))
    return {
        **softmax_program(raw),
        **domain,
        "source": _pointer(source),
        "source_offset": emit_local_scalar_offset(source["abi"], domain["coordinates"]),
        "result": _pointer(result),
        "result_offset": emit_local_scalar_offset(result["abi"], domain["coordinates"]),
        "writer_active": _canonical_writer_active(result["abi"]),
    }
