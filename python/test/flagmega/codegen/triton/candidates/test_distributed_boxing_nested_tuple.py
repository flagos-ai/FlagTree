# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A tuple transfer preserves structure but counts tensor leaves, not fields."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates.distributed_boxing import _leaf_transitions
from triton.flagmega.targets import NvidiaSm90Target


def _types():
    tensor = fm.tensor_type("float32", (1, 16))
    placement = fm.Placement((2, 4), "yx", "bb")
    local = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))), placement)
    return tensor, local


@pytest.mark.parametrize("identity_subtree", (False, True))
def test_nested_tuple_proposal_has_one_transition_per_tensor_leaf(identity_subtree):
    tensor, local = _types()
    nested = fm.TupleType((tensor, tensor)) if identity_subtree else fm.TupleType((local, local))
    source_type = fm.TupleType((local, nested))
    target_type = fm.TupleType((tensor, fm.TupleType((tensor, tensor))))
    builder = fm.IRBuilder(dialect="high_level", stage="frozen_constants")
    source = builder.var("source", source_type, id="source")
    result = builder.call("distributed.boxing", (source,), target_type, id="output", attrs={"new_type": target_type})
    builder.function("main", (source,), (result,))
    module = fm.verify_module(builder.build(entry="main"))
    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(point for point in proposed.selection_points if point.id == "tir.output")
    candidate = point.candidates[0]
    assert candidate.id == "tir.distributed_boxing.tensor_store"
    assert candidate.parameters["leaf_transitions"] == (
        "tensor_store", *(('identity',) * 2 if identity_subtree else ('tensor_store',) * 2),
    )
    assert candidate.facts["tuple_field_count"] == 3


def test_identity_tuple_still_has_one_entry_per_leaf():
    tensor, _ = _types()
    value = fm.TupleType((tensor, fm.TupleType((tensor, tensor))))
    assert _leaf_transitions(value, value) == ("identity",) * 3


def test_invalid_nested_field_does_not_get_silently_dropped():
    tensor, local = _types()
    source = fm.TupleType((fm.TupleType((local, local)), tensor))
    wrong_dtype = fm.tensor_type("int32", (1, 16))
    target = fm.TupleType((fm.TupleType((tensor, tensor)), wrong_dtype))
    assert _leaf_transitions(source, target) == ()


def test_same_flattened_leaves_do_not_justify_different_tuple_structure():
    tensor, local = _types()
    source = fm.TupleType((fm.TupleType((local,)), local))
    target = fm.TupleType((tensor, fm.TupleType((tensor,))))
    assert _leaf_transitions(source, target) == ()
