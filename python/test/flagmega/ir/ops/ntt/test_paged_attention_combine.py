# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


LAYOUT = ("seq", "head", "dim")


def _typed(name: str, value_type: fm.IRType) -> fm.Node:
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _infer(
    max_type,
    sum_type,
    acc_type,
    output_type,
    *,
    hidden_size=32,
    output_dtype="bfloat16",
):
    return fm.get_definition("ntt.paged_attention_combine").infer_type(
        (
            _typed("max", max_type),
            _typed("sum", sum_type),
            _typed("acc", acc_type),
        ),
        {
            "layout": LAYOUT,
            "hidden_size": hidden_size,
            "output_data_type": output_dtype,
            "output_type": output_type,
            "split_hierarchy_axis": 0,
            "split_count": 8,
        },
    )


def test_logical_combine_reconstructs_the_exact_output_contract():
    stats = fm.tensor_type("float32", (1, 4, 1))
    acc = fm.tensor_type("float32", (1, 4, 8))
    output = fm.tensor_type("bfloat16", (1, 4, 8))

    assert _infer(stats, stats, acc, output) == output


def test_combine_inputs_read_across_the_declared_partial_owner_group():
    definition = fm.get_definition("ntt.paged_attention_combine")

    assert tuple(
        parameter.memory_effect.owner_access
        for parameter in definition.input_parameters
    ) == (fm.MemoryOwnerAccess.PARTIAL_GROUP,) * 3


def test_vector_output_repacks_scalar_accumulator_dim():
    stats = fm.tensor_type("float32", (1, 4, 1))
    acc = fm.tensor_type("float32", (1, 4, 8))
    dtype = fm.vector_type("bfloat16", (2, 2))
    output = fm.tensor_type(dtype, (1, 4, 2))

    assert _infer(stats, stats, acc, output, output_dtype=dtype) == output


def test_distributed_combine_discharges_partial_and_preserves_other_axes():
    placement = fm.Placement((8, 16), "yx", "bb")
    policies = (
        fm.SBP.broadcast(),
        fm.SBP.split_contiguous((1,), 1),
        fm.SBP.broadcast(),
    )
    stats_tensor = fm.tensor_type("float32", (1, 16, 1))
    acc_tensor = fm.tensor_type("float32", (1, 16, 8))
    max_state = fm.DistributedType(
        stats_tensor, policies, placement, fm.SBP.partial((0,), fm.ReduceOp.MAX)
    )
    sum_state = fm.DistributedType(
        stats_tensor, policies, placement, fm.SBP.partial((0,), fm.ReduceOp.SUM)
    )
    acc_state = fm.DistributedType(
        acc_tensor, policies, placement, fm.SBP.partial((0,), fm.ReduceOp.SUM)
    )
    output = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 8)), policies, placement
    )

    assert _infer(
        max_state, sum_state, acc_state, output, hidden_size=128
    ) == output


def test_distributed_combine_may_turn_released_partial_axis_into_output_split():
    placement = fm.Placement((8, 16), "yx", "bb")
    input_policies = (
        fm.SBP.broadcast(),
        fm.SBP.split_contiguous((1,), 1),
        fm.SBP.broadcast(),
    )
    stats_tensor = fm.tensor_type("float32", (1, 16, 1))
    acc_tensor = fm.tensor_type("float32", (1, 16, 8))
    states = (
        fm.DistributedType(
            stats_tensor,
            input_policies,
            placement,
            fm.SBP.partial((0,), fm.ReduceOp.MAX),
        ),
        fm.DistributedType(
            stats_tensor,
            input_policies,
            placement,
            fm.SBP.partial((0,), fm.ReduceOp.SUM),
        ),
        fm.DistributedType(
            acc_tensor,
            input_policies,
            placement,
            fm.SBP.partial((0,), fm.ReduceOp.SUM),
        ),
    )
    output = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 8)),
        (
            fm.SBP.broadcast(),
            fm.SBP.split_contiguous((1,), 1),
            fm.SBP.split_contiguous((0,), 1),
        ),
        placement,
    )

    assert _infer(*states, output, hidden_size=128) == output


def test_combine_rejects_wrong_partial_operator_or_mismatched_policies():
    placement = fm.Placement((8, 16), "yx", "bb")
    broadcast = (fm.SBP.broadcast(),) * 3
    head_split = (
        fm.SBP.broadcast(),
        fm.SBP.split_contiguous((1,), 1),
        fm.SBP.broadcast(),
    )
    stats = fm.tensor_type("float32", (1, 16, 1))
    acc = fm.tensor_type("float32", (1, 16, 8))
    output = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 8)), broadcast, placement
    )
    max_state = fm.DistributedType(
        stats, broadcast, placement, fm.SBP.partial((0,), fm.ReduceOp.SUM)
    )
    sum_state = fm.DistributedType(
        stats, broadcast, placement, fm.SBP.partial((0,), fm.ReduceOp.SUM)
    )
    acc_state = fm.DistributedType(
        acc, head_split, placement, fm.SBP.partial((0,), fm.ReduceOp.SUM)
    )

    with pytest.raises(IRSchemaError, match=r"P\(Max\)/P\(Sum\)/P\(Sum\)"):
        _infer(max_state, sum_state, acc_state, output, hidden_size=128)


class _CombineGraph(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        stats = fm.tensor_type("float32", (1, 2, 1))
        acc_type = fm.tensor_type("float32", (1, 2, 4))
        max_state = self.input("max_state", stats)
        sum_state = self.input("sum_state", stats)
        acc_state = self.input("acc_state", acc_type)
        output = fm.F.ntt.paged_attention_combine(
            max_state,
            sum_state,
            acc_state,
            layout=LAYOUT,
            hidden_size=8,
            output_data_type="bfloat16",
            output_type=fm.tensor_type("bfloat16", (1, 2, 4)),
            split_hierarchy_axis=0,
            split_count=8,
            name="output",
        )
        self.function("main", (max_state, sum_state, acc_state), (output,))


def test_combine_evaluator_divides_accumulator_by_sum_and_casts_output():
    module = _CombineGraph().build()
    acc = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    denominator = torch.tensor([[[2.0]], [[4.0]]], dtype=torch.float32).reshape(1, 2, 1)

    output = TorchEvaluator(DictWeightResolver({})).run(
        module,
        {
            "max_state": torch.zeros((1, 2, 1), dtype=torch.float32),
            "sum_state": denominator,
            "acc_state": acc,
        },
    )[0]

    torch.testing.assert_close(output, (acc / denominator).to(torch.bfloat16))
