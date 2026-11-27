# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The optional KDA activation layout must not change recurrent state."""

from math import prod

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.kda import KdaRecurrentCore
from triton.flagmega.ir.ops.nn.kda_gated_recurrent_core import KdaGatedRecurrentCore
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module
from python.test.flagmega.ir.ops.nn.test_kda_gated_recurrent_core import _inputs, _operand_types


def operands(gated):
    types = _operand_types()
    values = _inputs()
    if gated:
        names = ("q", "k", "v", "g", "decay_rate", "gate_bias", "beta", "state")
    else:
        types = (*types[:3], types[4], fm.tensor_type("float32", (1, 2)), types[-1])
        values["decay"] = -5 * torch.sigmoid(values["decay_rate"] * (values["g"].float() + values["gate_bias"]))
        values["beta"] = torch.sigmoid(values["beta"].float())
        names = ("q", "k", "v", "decay", "beta", "state")
    definition = KdaGatedRecurrentCore if gated else KdaRecurrentCore
    arguments = {parameter.name: values[name] for parameter, name in zip(definition.input_parameters, names)}
    return definition, types, arguments, {"lower_bound": -5.} if gated else {}


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("lanes", [(), (4,), (2, 2), (1,)])
def test_kda_packed_activation_preserves_rounding_state_and_serialization(tmp_path, gated, lanes):
    definition, types, values, attrs = operands(gated)
    scalar = primitive_module(definition, types, **attrs)
    packed = primitive_module(definition, types, **attrs, output_lanes=lanes)
    pair = packed.node_map["output"].type
    assert pair.fields[1] == scalar.node_map["output"].type.fields[1]
    assert pair.fields[0] == fm.tensor_type(fm.vector_type("bfloat16", lanes) if lanes else "bfloat16",
                                          (1, 2, 4 // prod(lanes)))
    evaluator = TorchEvaluator(DictWeightResolver({}))
    expected, state = evaluator.run(scalar, values)[0]
    actual, packed_state = evaluator.run(packed, values)[0]
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=0, atol=0)
    torch.testing.assert_close(packed_state, state, rtol=0, atol=0)
    fm.verify_module(packed)
    fm.emit_module(packed, tmp_path / "packed.py")
    assert fm.load_module(tmp_path / "packed.py").semantic_hash == packed.semantic_hash
    assert bool(packed.node_map["output"].attrs.get("output_lanes")) == bool(lanes)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("lanes", [None, 4, (0,), (-1,), (True,), (2.0,)])
def test_kda_rejects_invalid_output_lanes(gated, lanes):
    definition, types, _, attrs = operands(gated)
    with pytest.raises(IRSchemaError, match="output_lanes"):
        primitive_module(definition, types, **attrs, output_lanes=lanes)


@pytest.mark.parametrize("gated", [False, True])
def test_kda_rejects_output_lanes_crossing_a_head_value_boundary(gated):
    definition, types, _, attrs = operands(gated)
    with pytest.raises(IRSchemaError, match="head_v must be divisible"):
        primitive_module(definition, types, **attrs, output_lanes=(8,))
