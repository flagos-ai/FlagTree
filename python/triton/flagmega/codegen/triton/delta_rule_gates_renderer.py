# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node, TupleType
from triton.flagmega.ir.ops.nn.delta_rule_gates import DeltaRuleGates


def delta_rule_gates_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer, _pointer, _canonical_writer_active, _scalar_local_domain)

    inputs = tuple(_buffer(raw, "inputs", parameter.name) for parameter in DeltaRuleGates.input_parameters)
    alpha, beta = tuple(_buffer(raw, "outputs", f"result_{index}") for index in range(2))
    attrs = DeltaRuleGates.normalize_attrs(raw["semantic_attrs"])
    nodes = tuple(Node(parameter.name, "builtin.var", (), _tensor_type(value["abi"]))
                  for parameter, value in zip(DeltaRuleGates.input_parameters, inputs, strict=True))
    expected = TupleType(tuple(_tensor_type(value["abi"]) for value in (alpha, beta)))
    if DeltaRuleGates.infer_type(nodes, attrs) != expected:
        raise CodegenError("DeltaRuleGates result ABI disagrees with its token/head contract.")
    domain = _scalar_local_domain(alpha["abi"], "_fm_offsets")
    coordinates = domain["local_coordinates"]
    result = {**domain, "tile": int(raw["parameters"]["elements_per_program"]),
              "threshold": repr(attrs["softplus_threshold"]), "accurate_exp": attrs["alpha_exp_mode"] == "accurate"}
    for parameter, value in zip(DeltaRuleGates.input_parameters, inputs, strict=True):
        result[parameter.name] = _pointer(value)
        operand_coordinates = coordinates if parameter in (DeltaRuleGates.a, DeltaRuleGates.b) else coordinates[1:]
        result[parameter.name + "_offset"] = emit_local_scalar_offset(value["abi"], operand_coordinates)
    for name, value in (("alpha", alpha), ("beta", beta)):
        result[name] = _pointer(value)
        result[name + "_offset"] = emit_local_scalar_offset(value["abi"], coordinates)
        result[name + "_writer"] = _canonical_writer_active(value["abi"])
    return result
