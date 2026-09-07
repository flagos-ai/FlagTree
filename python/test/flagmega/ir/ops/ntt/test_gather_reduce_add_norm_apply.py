# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.ntt.gather_reduce_add_norm_apply import (
    GatherReduceAddNormApply,
)


def _types(*, reduce_op=fm.ReduceOp.SUM):
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    value = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 8)),
        (broadcast, broadcast),
        placement,
    )
    partial = fm.DistributedType(
        value.tensor,
        value.axis_policies,
        placement,
        fm.SBP.partial((0, 1), reduce_op),
    )
    parameter = fm.DistributedType(
        fm.tensor_type("bfloat16", (8,)), (broadcast,), placement
    )
    return partial, value, parameter


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def test_type_is_value_and_normalized_tuple_with_named_alias_contract():
    partial, value, parameter = _types()
    result = GatherReduceAddNormApply.infer_type(
        (
            _node("partial", partial),
            _node("addend", value),
            _node("scale", parameter),
            _node("bias", parameter),
        ),
        {"axis": -1, "epsilon": 1e-6, "use_mean": False, "has_bias": True},
    )

    assert result == fm.TupleType((value, value))
    assert GatherReduceAddNormApply.inplace_output_parameters == (
        GatherReduceAddNormApply.addend,
        None,
    )
    assert (
        GatherReduceAddNormApply.input.memory_effect.owner_access.value
        == "partial_group"
    )


def test_collective_materialization_declares_chip_visible_results_and_operands():
    assert GatherReduceAddNormApply.result_memory_effects == (
        fm.MemoryEffect.CHIP_WRITE, fm.MemoryEffect.CHIP_WRITE,
    )
    for parameter in (GatherReduceAddNormApply.addend, GatherReduceAddNormApply.scale, GatherReduceAddNormApply.bias):
        assert parameter.memory_effect == fm.MemoryEffect.CHIP_READ


@pytest.mark.parametrize("reduce_op", [fm.ReduceOp.MAX, fm.ReduceOp.MIN])
def test_type_rejects_non_additive_partial_inputs(reduce_op):
    partial, value, parameter = _types(reduce_op=reduce_op)
    with pytest.raises(IRSchemaError, match="Sum-partial"):
        GatherReduceAddNormApply.infer_type(
            (
                _node("partial", partial),
                _node("addend", value),
                _node("scale", parameter),
                _node("bias", parameter),
            ),
            {"axis": -1, "epsilon": 1e-6, "use_mean": False, "has_bias": True},
        )


@pytest.mark.parametrize("use_mean", [False, True])
def test_evaluator_preserves_rounded_residual_and_applies_private_stats(use_mean):
    partial_type, value_type, parameter_type = _types()

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            partial = self.input("partial", partial_type, id="partial")
            addend = self.input("addend", value_type, id="addend")
            scale = self.input("scale", parameter_type, id="scale")
            bias = self.input("bias", parameter_type, id="bias")
            result = fm.F.ntt.gather_reduce_add_norm_apply(
                partial,
                addend,
                scale,
                bias,
                axis=-1,
                epsilon=1e-6,
                use_mean=use_mean,
                name="result",
            )
            value, normalized = fm.F.tensors.get_items(
                result, 0, 1, name_prefix="result"
            )
            self.function(
                "main", (partial, addend, scale, bias), (value, normalized)
            )

    torch.manual_seed(23)
    partial = torch.randn((1, 8), dtype=torch.bfloat16)
    addend = torch.randn((1, 8), dtype=torch.bfloat16)
    scale = torch.randn((8,), dtype=torch.bfloat16)
    bias = torch.randn((8,), dtype=torch.bfloat16)
    actual_value, actual_norm = TorchEvaluator(DictWeightResolver({})).run(
        Graph().build(),
        {"partial": partial, "addend": addend, "scale": scale, "bias": bias},
    )
    expected_value = (partial + addend).to(torch.bfloat16)
    if use_mean:
        mean = expected_value.float().mean(-1, keepdim=True)
        variance = expected_value.float().square().mean(-1, keepdim=True) - mean.square()
        centered = expected_value.float() - mean
    else:
        variance = expected_value.float().square().mean(-1, keepdim=True)
        centered = expected_value.float()
    expected_norm = (
        centered * torch.rsqrt(variance.clamp_min(0) + 1e-6)
        * scale.float() + bias.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual_value, expected_value)
    torch.testing.assert_close(actual_norm, expected_norm)
