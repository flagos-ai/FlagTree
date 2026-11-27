# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.reduction_domain import local_reduction_domain
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.ops.nn.l2_normalization import L2Normalization


def l2_normalization_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _canonical_writer_active

    source = _buffer(raw, "inputs", "value")
    result = _buffer(raw, "outputs", "result")
    attrs = L2Normalization.normalize_attrs(raw["semantic_attrs"])
    value_type = _tensor_type(source["abi"])
    if L2Normalization.infer_type((Node("input", "builtin.var", (), value_type),), attrs) != _tensor_type(result["abi"]):
        raise CodegenError("L2Normalization result ABI disagrees with its reduction contract.")
    axes = tuple(normalize_axis(axis, len(source["abi"]["local_capacity_shape"])) for axis in attrs["axes"])
    domain = local_reduction_domain(source["abi"], axes, int(raw["parameters"]["elements_per_program"]))
    group = attrs.get("reduction_group_size", 1)
    groups = domain["capacity"] // group
    grouped_domain = local_reduction_domain(source["abi"], axes, int(raw["parameters"]["elements_per_program"]),
                                            reduction_index="_fm_group_offsets")
    return {**domain, "source": _pointer(source), "result": _pointer(result),
            "reduction_group_size": group, "group_tile": 1 << (max(groups, 1) - 1).bit_length(),
            "group_active": grouped_domain["active"],
            "group_source_offset": emit_local_scalar_offset(source["abi"], grouped_domain["coordinates"]),
            "source_offset": emit_local_scalar_offset(source["abi"], domain["coordinates"]),
            "result_offset": emit_local_scalar_offset(result["abi"], domain["coordinates"]),
            "writer_active": _canonical_writer_active(result["abi"]),
            "epsilon": repr(attrs["epsilon"]), "epsilon_add": attrs["epsilon_mode"] == "add",
            "accurate_math": attrs.get("math_mode", "fast") == "accurate",
            "reciprocal_multiply": attrs["division_mode"] == "reciprocal_multiply"}
