# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The KDA family must accept a head-axis split, not only a full broadcast.

Every KDA operand carries the head count on axis 1 — ``[1, H, D]`` projections,
``[1, H]`` beta, ``[1, H, Dk, Dv]`` recurrent state, ``[3, H, D, K]`` conv
state and weight — and the recurrence is independent per head.  Forcing all
operands to be fully broadcast therefore replicates every KDA weight on every
chip for no algebraic reason.  On GLM-5.3-Flash that is 130.69 MiB per layer
times 34 KDA layers, so a 4-way head split releases 3.25 GiB per chip.

The contract: axis 1 is either broadcast on every operand or carries the *same*
split on every operand, every other axis stays broadcast, and no operand is
partial or exclusive.  Mixed head ownership has no meaning here — a head's
state, gate and projections must live on one owner.
"""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.kda import KdaConvolution, KdaRecurrentCore
from triton.flagmega.ir.ops.nn.kda_gated_recurrent_core import KdaGatedRecurrentCore

HEADS, DIM, KERNEL = 8, 4, 4
CONV_ATTRS = {"kernel_size": KERNEL, "activation": "silu"}
RECURRENT_ATTRS = {"epsilon": 1e-6, "l2norm_in_kernel": True}
GATED_ATTRS = {"lower_bound": -5.0, "epsilon": 1e-6, "l2norm_in_kernel": True}


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _placement():
    return fm.Placement((4,), "x", "b")


def _policies(tensor, head_axis=1, split=None):
    policies = [fm.SBP.broadcast() for _ in tensor.shape]
    if split is not None:
        policies[head_axis] = split
    return tuple(policies)


def _typed(tensor, *, split=None, placement=None, head_axis=1):
    return fm.DistributedType(tensor, _policies(tensor, head_axis, split), placement or _placement())


def _conv_tensors():
    projection = fm.tensor_type("bfloat16", (1, HEADS, DIM))
    return (projection, projection, projection,
            fm.tensor_type("bfloat16", (3, HEADS, DIM, KERNEL - 1)),
            fm.tensor_type("bfloat16", (3, HEADS, DIM, KERNEL)))


def _recurrent_tensors():
    activation = fm.tensor_type("bfloat16", (1, HEADS, DIM))
    wide = fm.tensor_type("float32", (1, HEADS, DIM))
    return (activation, activation, activation, wide,
            fm.tensor_type("float32", (1, HEADS)),
            fm.tensor_type("float32", (1, HEADS, DIM, DIM)))


def _gated_tensors():
    activation = fm.tensor_type("bfloat16", (1, HEADS, DIM))
    wide = fm.tensor_type("float32", (1, HEADS, DIM))
    return (activation, activation, activation, activation, wide, wide,
            fm.tensor_type("bfloat16", (1, HEADS)),
            fm.tensor_type("float32", (1, HEADS, DIM, DIM)))


CASES = (
    ("convolution", KdaConvolution, _conv_tensors, CONV_ATTRS),
    ("recurrent", KdaRecurrentCore, _recurrent_tensors, RECURRENT_ATTRS),
    ("gated_recurrent", KdaGatedRecurrentCore, _gated_tensors, GATED_ATTRS),
)
IDS = tuple(case[0] for case in CASES)


def _infer(definition, tensors, attrs, *, split=None):
    inputs = tuple(_node(str(index), _typed(tensor, split=split))
                   for index, tensor in enumerate(tensors))
    return definition.infer_type(inputs, attrs)


@pytest.mark.parametrize("_name,definition,tensors,attrs", CASES, ids=IDS)
def test_a_common_head_split_survives_to_every_output(_name, definition, tensors, attrs):
    split = fm.SBP.split_contiguous((0,))
    result = _infer(definition, tensors(), attrs, split=split)

    for field in result.fields:
        assert isinstance(field, fm.DistributedType)
        assert field.placement == _placement()
        assert field.axis_policies[1] == split, f"axis 1 of {field.tensor} lost the head split"
        assert all(policy == fm.SBP.broadcast()
                   for index, policy in enumerate(field.axis_policies) if index != 1)
        assert field.partial is None


@pytest.mark.parametrize("_name,definition,tensors,attrs", CASES, ids=IDS)
def test_a_full_broadcast_is_still_accepted(_name, definition, tensors, attrs):
    result = _infer(definition, tensors(), attrs)

    for field in result.fields:
        assert isinstance(field, fm.DistributedType)
        assert all(policy == fm.SBP.broadcast() for policy in field.axis_policies)


@pytest.mark.parametrize("_name,definition,tensors,attrs", CASES, ids=IDS)
def test_mixed_head_ownership_is_rejected(_name, definition, tensors, attrs):
    """One head's state, gate and projections must live on one owner."""

    split = fm.SBP.split_contiguous((0,))
    values = tensors()
    inputs = tuple(_node(str(index), _typed(tensor, split=split if index else None))
                   for index, tensor in enumerate(values))
    with pytest.raises(IRSchemaError, match="head"):
        definition.infer_type(inputs, attrs)


@pytest.mark.parametrize("_name,definition,tensors,attrs", CASES, ids=IDS)
def test_a_split_on_any_other_axis_is_rejected(_name, definition, tensors, attrs):
    """Only the head axis is independent; channels and state columns are not."""

    split = fm.SBP.split_contiguous((0,))
    values = tensors()
    # Axis 2 is the head channel on every operand that has one.
    inputs = tuple(_node(str(index), _typed(tensor, split=split, head_axis=2)
                         if tensor.rank > 2 else _typed(tensor, split=split))
                   for index, tensor in enumerate(values))
    with pytest.raises(IRSchemaError):
        definition.infer_type(inputs, attrs)


@pytest.mark.parametrize("_name,definition,tensors,attrs", CASES, ids=IDS)
def test_the_head_split_relation_is_offered_to_the_search(_name, definition, tensors, attrs):
    """``distributed_input_type_tuples`` must enumerate the head-split contract."""

    split = fm.SBP.split_contiguous((0,))
    choices = tuple((_typed(tensor), _typed(tensor, split=split)) for tensor in tensors())
    offered = tuple(definition.distributed_input_type_tuples(choices, attrs))

    head_split = tuple(_typed(tensor, split=split) for tensor in tensors())
    broadcast = tuple(_typed(tensor) for tensor in tensors())
    assert head_split in offered
    assert broadcast in offered
    # Nothing mixed: every offered tuple must agree on axis 1.
    for candidate in offered:
        assert len({value.axis_policies[1] for value in candidate}) == 1


@pytest.mark.parametrize("_name,definition,tensors,attrs", CASES, ids=IDS)
def test_a_head_split_output_demand_reaches_the_inputs(_name, definition, tensors, attrs):
    """A consumer that wants head-split results must be able to ask for them."""

    split = fm.SBP.split_contiguous((0,))
    values = tensors()
    output = _infer(definition, values, attrs, split=split)
    logical = tuple(value for value in values)
    relation = definition.infer_distributed_input_types(output, logical, attrs)

    assert relation, "the head-split output must invert to head-split inputs"
    for candidate in relation:
        assert all(value.axis_policies[1] == split for value in candidate)
