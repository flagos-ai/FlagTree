# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The auto-distributed search must see head-split KDA candidates.

``TypeInferenceCandidateProvider`` owns no layout policy — it only feeds the
producer layouts available at the use edge into the operation's own
``infer_type``, plus whatever ``infer_distributed_input_types`` inverts.  So a
head-split candidate reaches the solver only if the KDA type contract admits
one.  Without it every KDA weight stays replicated on every chip, which is the
4-card capacity problem on GLM-5.3-Flash.
"""

import pytest
from triton.flagmega import ir as fm
from python.test.flagmega.passes.auto_distributed.test_router_primitive_providers import candidates_for

HEADS, DIM, KERNEL = 8, 4, 4
CONV_ATTRS = {"kernel_size": KERNEL, "activation": "silu"}
RECURRENT_ATTRS = {"epsilon": 1e-6, "l2norm_in_kernel": True}
GATED_ATTRS = {"lower_bound": -5.0, "epsilon": 1e-6, "l2norm_in_kernel": True}


def _conv_types():
    projection = fm.tensor_type("bfloat16", (1, HEADS, DIM))
    return (projection, projection, projection,
            fm.tensor_type("bfloat16", (3, HEADS, DIM, KERNEL - 1)),
            fm.tensor_type("bfloat16", (3, HEADS, DIM, KERNEL)))


def _recurrent_types():
    activation = fm.tensor_type("bfloat16", (1, HEADS, DIM))
    wide = fm.tensor_type("float32", (1, HEADS, DIM))
    return (activation, activation, activation, wide,
            fm.tensor_type("float32", (1, HEADS)),
            fm.tensor_type("float32", (1, HEADS, DIM, DIM)))


def _gated_types():
    activation = fm.tensor_type("bfloat16", (1, HEADS, DIM))
    wide = fm.tensor_type("float32", (1, HEADS, DIM))
    return (activation, activation, activation, activation, wide, wide,
            fm.tensor_type("bfloat16", (1, HEADS)),
            fm.tensor_type("float32", (1, HEADS, DIM, DIM)))


CASES = (
    ("nn.kda_convolution", _conv_types, CONV_ATTRS),
    ("nn.kda_recurrent_core", _recurrent_types, RECURRENT_ATTRS),
    ("nn.kda_gated_recurrent_core", _gated_types, GATED_ATTRS),
)
IDS = ("convolution", "recurrent", "gated_recurrent")


def _fields(value):
    return value.fields if isinstance(value, fm.TupleType) else (value,)


@pytest.mark.parametrize("op,types,attrs", CASES, ids=IDS)
@pytest.mark.parametrize("mesh", [(4,), (2, 2)])
def test_the_search_sees_both_a_replicated_and_a_head_split_candidate(op, types, attrs, mesh):
    candidates, _ = candidates_for(op, types(), attrs, mesh=mesh)
    broadcast = fm.SBP.broadcast()

    assert any(isinstance(field.axis_policies[1], fm.SBPSplit)
               for candidate in candidates for field in _fields(candidate.return_type)), \
        "no head-split candidate reached the solver"
    assert any(all(policy == broadcast for policy in field.axis_policies)
               for candidate in candidates for field in _fields(candidate.return_type))


@pytest.mark.parametrize("op,types,attrs", CASES, ids=IDS)
@pytest.mark.parametrize("mesh", [(4,), (2, 2)])
def test_every_candidate_keeps_one_head_owner_and_no_partial(op, types, attrs, mesh):
    candidates, _ = candidates_for(op, types(), attrs, mesh=mesh)
    broadcast = fm.SBP.broadcast()

    for candidate in candidates:
        values = (*candidate.input_types, *_fields(candidate.return_type))
        heads = {value.axis_policies[1] for value in values}
        assert len(heads) == 1, f"{op} candidate mixes head owners: {heads}"
        for value in values:
            assert all(policy == broadcast
                       for axis, policy in enumerate(value.axis_policies) if axis != 1)
            assert value.partial is None and value.exclusive is None


@pytest.mark.parametrize("op,types,attrs", CASES, ids=IDS)
def test_a_head_split_candidate_covers_the_whole_placement(op, types, attrs):
    """A partial split would leave the remaining chips holding a full replica."""

    candidates, placement = candidates_for(op, types(), attrs, mesh=(2, 2))
    splits = [field.axis_policies[1] for candidate in candidates
              for field in _fields(candidate.return_type)
              if isinstance(field.axis_policies[1], fm.SBPSplit)]

    assert splits
    assert any(set(split.hierarchy_axes) == set(range(placement.rank)) for split in splits)


@pytest.mark.parametrize("op,types,attrs", CASES[1:], ids=IDS[1:])
@pytest.mark.parametrize("lanes", [(4,), (2, 2), (1,)])
def test_packed_activation_keeps_the_same_head_owner_as_scalar_state(op, types, attrs, lanes):
    candidates, _ = candidates_for(op, types(), {**attrs, "output_lanes": lanes}, mesh=(2, 2))
    assert candidates
    assert any(isinstance(candidate.return_type.fields[0].axis_policies[1], fm.SBPSplit) for candidate in candidates)
    for candidate in candidates:
        output, state = candidate.return_type.fields
        assert output.tensor.dtype == fm.vector_type("bfloat16", lanes)
        assert state.tensor.dtype == fm.DType.FLOAT32
        assert output.axis_policies[1] == state.axis_policies[1]
        assert all(value.axis_policies[1] == state.axis_policies[1] for value in candidate.input_types)
        inputs = tuple(fm.Node(str(i), "builtin.var", (), kind) for i, kind in enumerate(candidate.input_types))
        assert fm.get_definition(op).infer_type(inputs, {**attrs, "output_lanes": lanes}) == candidate.return_type
