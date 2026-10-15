# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The gate-fused KDA recurrence: type contract and FP32-interior numerics.

``nn.kda_recurrent_core`` takes an already-formed FP32 log-decay and an already
sigmoided beta, so the published forget gate has to be five graph nodes with a
materialized FP32 ``[1, H, D]`` intermediate.  ``nn.kda_gated_recurrent_core``
takes the raw projections instead and evaluates
``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` and ``sigmoid(beta)``
inside the recurrence, matching ``naive_kda_lowerbound_gate`` plus
``fused_recurrent_kda(use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)``.
"""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.kda_gated_recurrent_core import KdaGatedRecurrentCore

HEADS, DIM = 2, 4
LOWER_BOUND = -5.0


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _broadcast(tensor, placement):
    return fm.DistributedType(tensor, (fm.SBP.broadcast(),) * tensor.rank, placement)


def _operand_types(*, gate_dtype="bfloat16", beta_dtype="bfloat16"):
    activation = fm.tensor_type("bfloat16", (1, HEADS, DIM))
    return (
        activation, activation, activation,
        fm.tensor_type(gate_dtype, (1, HEADS, DIM)),
        fm.tensor_type("float32", (1, HEADS, DIM)),
        fm.tensor_type("float32", (1, HEADS, DIM)),
        fm.tensor_type(beta_dtype, (1, HEADS)),
        fm.tensor_type("float32", (1, HEADS, DIM, DIM)),
    )


def _attrs(**overrides):
    return {"lower_bound": LOWER_BOUND, "epsilon": 1e-6, "l2norm_in_kernel": True, **overrides}


def _module(*, l2norm=True, gate_dtype="bfloat16", beta_dtype="bfloat16"):
    names = ("q", "k", "v", "g", "decay_rate", "gate_bias", "beta", "state")
    types = _operand_types(gate_dtype=gate_dtype, beta_dtype=beta_dtype)

    class Graph(fm.Module):
        def forward(self):
            operands = tuple(self.input(name, value_type, id=name)
                             for name, value_type in zip(names, types))
            pair = fm.F.nn.kda_gated_recurrent_core(
                *operands, lower_bound=LOWER_BOUND, epsilon=1e-6, l2norm_in_kernel=l2norm)
            output, new_state = fm.F.tensors.get_items(pair, 0, 1, name_prefix="recurrent")
            self.function("main", operands, (output, new_state))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _inputs(seed=11, *, gate_dtype=torch.bfloat16, beta_dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape):
        return torch.randn(shape, generator=generator)

    return {
        "q": normal(1, HEADS, DIM).bfloat16(),
        "k": normal(1, HEADS, DIM).bfloat16(),
        "v": normal(1, HEADS, DIM).bfloat16(),
        "g": normal(1, HEADS, DIM).to(gate_dtype),
        # exp(A_log) broadcast to the channel axis by the constant island.
        "decay_rate": normal(1, HEADS, 1).exp().expand(1, HEADS, DIM).contiguous(),
        "gate_bias": normal(1, HEADS, DIM),
        "beta": normal(1, HEADS).to(beta_dtype),
        "state": normal(1, HEADS, DIM, DIM),
    }


def _reference(values, *, l2norm=True):
    """naive_kda_lowerbound_gate + naive_recurrent_kda, all FP32 interior."""
    gate = values["g"].float() + values["gate_bias"].float()
    decay = LOWER_BOUND * torch.sigmoid(values["decay_rate"].float() * gate)
    beta = torch.sigmoid(values["beta"].float())
    query, key = values["q"].float(), values["k"].float()
    if l2norm:
        query = query / torch.sqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = key / torch.sqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    state = values["state"] * decay.exp().unsqueeze(-1)
    delta = (values["v"].float() - (state * key.unsqueeze(-1)).sum(-2)) * beta.unsqueeze(-1)
    state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    return (state * query.unsqueeze(-1)).sum(-2).bfloat16(), state


def test_evaluator_matches_the_upstream_fused_gate_recurrence():
    values = _inputs()
    actual, actual_state = TorchEvaluator(DictWeightResolver({})).run(_module(), values)
    expected, expected_state = _reference(values)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


def test_the_graph_expansion_of_the_gate_is_observably_different():
    """Guard against a vacuous bit-exact pass.

    The rejected form materializes the gate through BF16-typed ``math`` nodes,
    which rounds the sum and the sigmoid.  If this fixture ever stopped
    separating the two policies, the assertion above would prove nothing.
    """
    values = _inputs()
    expected, _ = _reference(values)
    # nn.kda_recurrent_core's operand contract forces an FP32 decay, so the
    # expansion's rounding lands on the intermediate add/mul instead.
    rounded_gate = (values["g"].float() + values["gate_bias"].float()).bfloat16().float()
    rounded_decay = (LOWER_BOUND * torch.sigmoid(
        (values["decay_rate"].float() * rounded_gate).bfloat16().float()))
    beta = torch.sigmoid(values["beta"].float().bfloat16().float())
    query, key = values["q"].float(), values["k"].float()
    query = query / torch.sqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    key = key / torch.sqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    state = values["state"] * rounded_decay.exp().unsqueeze(-1)
    delta = (values["v"].float() - (state * key.unsqueeze(-1)).sum(-2)) * beta.unsqueeze(-1)
    state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    expanded = (state * query.unsqueeze(-1)).sum(-2).bfloat16()

    assert torch.count_nonzero(expected != expanded), (
        "This fixture no longer separates the FP32 in-kernel gate from the "
        "rounded graph expansion, so the bit-exact test above is vacuous."
    )


def test_evaluator_honors_disabled_l2_normalization():
    values = _inputs(seed=13)
    actual, actual_state = TorchEvaluator(DictWeightResolver({})).run(
        _module(l2norm=False), values)
    expected, expected_state = _reference(values, l2norm=False)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("gate_dtype", ["bfloat16", "float32"])
def test_the_raw_gate_may_arrive_in_either_activation_dtype(gate_dtype):
    """Upstream keeps ``g`` in the activation dtype; FP32 is also admissible."""
    torch_dtype = torch.bfloat16 if gate_dtype == "bfloat16" else torch.float32
    values = _inputs(seed=17, gate_dtype=torch_dtype)
    actual, _ = TorchEvaluator(DictWeightResolver({})).run(
        _module(gate_dtype=gate_dtype), values)
    expected, _ = _reference(values)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bound", [0.0, 5.0])
def test_a_non_negative_lower_bound_is_rejected(bound):
    """The softplus gate is a different contract, not a bound of zero."""
    with pytest.raises(IRSchemaError, match="lower_bound must be a negative"):
        KdaGatedRecurrentCore.normalize_attrs(_attrs(lower_bound=bound))


def test_weight_maps_must_arrive_in_head_layout():
    """The exp/broadcast belongs to the constant island, not the runtime graph."""
    types = list(_operand_types())
    types[4] = fm.tensor_type("float32", (HEADS,))
    inputs = tuple(_node(str(index), value) for index, value in enumerate(types))
    with pytest.raises(IRSchemaError, match="decay_rate/gate_bias maps must be"):
        KdaGatedRecurrentCore.infer_type(inputs, _attrs())


def test_preserves_one_common_broadcast_placement():
    placement = fm.Placement((4,), "x", "b")
    inputs = tuple(_node(str(index), _broadcast(value, placement))
                   for index, value in enumerate(_operand_types()))
    result = KdaGatedRecurrentCore.infer_type(inputs, _attrs())

    assert all(isinstance(field, fm.DistributedType) for field in result.fields)
    assert all(field.placement == placement for field in result.fields)


def test_rejects_mixed_or_mismatched_distributed_inputs():
    placement = fm.Placement((4,), "x", "b")
    other = fm.Placement((2,), "x", "b")
    plain = _operand_types()
    mixed = (*plain[:1], _broadcast(plain[1], placement), *plain[2:])
    with pytest.raises(IRSchemaError, match="all plain or all distributed"):
        KdaGatedRecurrentCore.infer_type(
            tuple(_node(str(index), value) for index, value in enumerate(mixed)), _attrs())

    mismatched = tuple(_broadcast(value, other if index == 1 else placement)
                       for index, value in enumerate(plain))
    with pytest.raises(IRSchemaError, match="one placement"):
        KdaGatedRecurrentCore.infer_type(
            tuple(_node(str(index), value) for index, value in enumerate(mismatched)), _attrs())
