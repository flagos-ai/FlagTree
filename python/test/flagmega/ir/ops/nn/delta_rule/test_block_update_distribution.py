# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from itertools import product

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.delta_rule_block_update import DeltaRuleBlockUpdate
from python.test.flagmega.ir.ops.nn.delta_rule.test_block_update_types import block_types


@pytest.mark.parametrize("cyclic", (False, True))
def test_head_join_equals_exhaustive_inference_including_invalid_combinations(cyclic):
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    key_head = (fm.SBP.split_block_cyclic((0, ), 1) if cyclic else fm.SBP.split_contiguous((0, ), 1))
    value_head = fm.scale_split_units(key_head, 2, 1)
    tensors = block_types()
    choices = []
    for index, tensor in enumerate(tensors[:-1]):
        head = key_head if index < 2 else value_head
        policies = [b] * tensor.rank
        policies[1] = head
        invalid = [b] * tensor.rank
        invalid[-1] = key_head
        choices.append((
            fm.DistributedType(tensor, (b, ) * tensor.rank, placement),
            fm.DistributedType(tensor, tuple(policies), placement),
            fm.DistributedType(tensor, tuple(invalid), placement),
        ))
    choices.append((tensors[-1], ))
    attrs = DeltaRuleBlockUpdate.normalize_attrs({})

    def accepted(combinations):
        results = set()
        for types in combinations:
            inputs = tuple(
                fm.Node(parameter.name, "builtin.var", (), value)
                for parameter, value in zip(DeltaRuleBlockUpdate.input_parameters, types))
            try:
                output = DeltaRuleBlockUpdate.infer_type(inputs, attrs)
            except IRSchemaError:
                continue
            results.add((types, output))
        return results

    exhaustive = accepted(product(*choices))
    joined = tuple(DeltaRuleBlockUpdate.distributed_input_type_tuples(choices, attrs))
    assert len(exhaustive) == 2
    assert accepted(joined) == exhaustive
    assert len(joined) == len(exhaustive)
    # Removing an available producer contract removes only that relation.
    choices[2] = (choices[2][0], )
    assert accepted(DeltaRuleBlockUpdate.distributed_input_type_tuples(choices, attrs)) == accepted(product(*choices))


def test_block_update_preserves_symbolic_token_block_relation():
    tokens = fm.DimVar("tokens")
    inputs = tuple(
        fm.Node(parameter.name, "builtin.var", (), value)
        for parameter, value in zip(DeltaRuleBlockUpdate.input_parameters, block_types(tokens)))
    output = DeltaRuleBlockUpdate.infer_type(inputs, DeltaRuleBlockUpdate.normalize_attrs({}))
    assert output.fields[0].shape[0] == tokens
