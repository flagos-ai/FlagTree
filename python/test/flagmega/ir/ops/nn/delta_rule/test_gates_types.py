# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from itertools import product

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.delta_rule_gates import DeltaRuleGates
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


def gate_types(tokens=3, heads=8):
    return (fm.tensor_type("bfloat16", (tokens, heads)), fm.tensor_type("bfloat16", (tokens, heads)),
            fm.tensor_type("float32", (heads,)), fm.tensor_type("bfloat16", (heads,)))


@pytest.mark.parametrize("attrs", ({"softplus_threshold": 0}, {"softplus_threshold": True},
    {"softplus_threshold": float("nan")}, {"alpha_exp_mode": "auto"}))
def test_gate_invalid_attributes(attrs):
    with pytest.raises(IRSchemaError):
        primitive_module(DeltaRuleGates, gate_types(), **attrs)


@pytest.mark.parametrize("index,value", (
    (0, fm.tensor_type("int32", (3, 8))), (1, fm.tensor_type("bfloat16", (2, 8))),
    (2, fm.tensor_type("float32", (4,))), (3, fm.tensor_type("bfloat16", (1, 8))),
))
def test_gate_operand_types_and_extents(index, value):
    types = list(gate_types())
    types[index] = value
    with pytest.raises(IRSchemaError):
        primitive_module(DeltaRuleGates, types)


def test_gate_dynamic_rows_and_fp32_results():
    node = primitive_module(DeltaRuleGates, gate_types("tokens")).node_map["output"]
    result = fm.tensor_type("float32", ("tokens", 8))
    assert node.type == fm.TupleType((result, result))
    assert [parameter.name for parameter in DeltaRuleGates.input_parameters] == ["a", "b", "a_log", "dt_bias"]


def test_gate_relation_join_matches_small_cartesian_product_including_plain_types():
    tensor_types = gate_types()
    placement = fm.Placement((2, 2), "xy", "bb")
    b, h, t = fm.SBP.broadcast(), fm.SBP.split_contiguous((0,)), fm.SBP.split_block_cyclic((1,), 1)
    choices = tuple((tensor, *(fm.DistributedType(tensor, policies, placement) for policies in (
        ((b, b), (b, h), (t, h)) if tensor.rank == 2 else ((b,), (h,))
    ))) for tensor in tensor_types)
    attrs = DeltaRuleGates.normalize_attrs({})

    def accepted(combinations):
        result = set()
        for types in combinations:
            nodes = tuple(fm.Node(str(index), "builtin.var", (), value) for index, value in enumerate(types))
            try:
                DeltaRuleGates.infer_type(nodes, attrs)
                result.add(types)
            except IRSchemaError:
                pass
        return result

    expected = accepted(product(*choices))
    assert len(expected) == 4
    assert accepted(DeltaRuleGates.distributed_input_type_tuples(choices, attrs)) == expected
    missing = (*choices[:2], (choices[2][0], choices[2][1]), choices[3])
    assert accepted(DeltaRuleGates.distributed_input_type_tuples(missing, attrs)) == accepted(product(*missing))
