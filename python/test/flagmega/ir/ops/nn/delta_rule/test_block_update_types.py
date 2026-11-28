# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.delta_rule_block_update import DeltaRuleBlockUpdate
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


def block_types(tokens=65, state=None):
    q = fm.tensor_type("bfloat16", (tokens, 2, 16))
    v = fm.tensor_type("bfloat16", (tokens, 4, 8))
    c = fm.tensor_type("bfloat16", ((tokens + 63) // 64, 4, 64, 64))
    p = fm.tensor_type("float32", ((tokens + 63) // 64, 4, 64))
    state = fm.RefType("state", (("matrix", fm.tensor_type("float32", (4, 8, 16))), )) if state is None else state
    return q, q, v, c, p, state


@pytest.mark.parametrize("tokens", [0, 1, 64, 65, 129])
def test_block_update_output_and_reference_identity(tokens):
    types = block_types(tokens)
    module = fm.verify_module(primitive_module(DeltaRuleBlockUpdate, types))
    result = module.node_map["output"]
    assert result.type == fm.TupleType((types[2], types[-1]))
    assert result.effect == fm.effect("read_write", "delta_rule_state")
    assert DeltaRuleBlockUpdate.state.memory_effect.value == "read_write"


def test_state_layout_accepts_packed_key_and_unit_layer():
    state = fm.RefType("state", (("matrix", fm.tensor_type(fm.VectorType(fm.DType.FLOAT32, (4, )), (1, 4, 8, 4))), ))
    module = primitive_module(DeltaRuleBlockUpdate, block_types(state=state),
                              state_layout=("layer", "head", "value", "key"), state_vector_axes=("key", ))
    assert module.node_map["output"].type.fields[1] == state


@pytest.mark.parametrize("attrs", [{"scale": 0}, {"scale": float("nan")}, {"scale": True}, {"state_field": "missing"},
                                   {"state_field": "a.b"}, {"state_layout":
                                                            ("head", "head", "key")}, {"state_vector_axes": ("key", )}])
def test_reject_bad_state_or_numerical_contract(attrs):
    with pytest.raises(IRSchemaError):
        primitive_module(DeltaRuleBlockUpdate, block_types(), **attrs)


@pytest.mark.parametrize("index,replacement", [(0, fm.tensor_type("float32", (65, 2, 16))),
                                               (1, fm.tensor_type("bfloat16", (65, 2, 8))),
                                               (2, fm.tensor_type("bfloat16", (65, 3, 8))),
                                               (3, fm.tensor_type("bfloat16", (1, 4, 64, 64))),
                                               (4, fm.tensor_type("float32", (2, 4, 32)))])
def test_reject_mismatched_block_operands(index, replacement):
    types = list(block_types())
    types[index] = replacement
    with pytest.raises(IRSchemaError):
        primitive_module(DeltaRuleBlockUpdate, types)
