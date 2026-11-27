# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch
from math import prod

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.ntt.add_norm_stats import (
    AddNormStats,
    can_materialize_sum_partial,
)


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


@pytest.mark.parametrize("lanes", [1, 8])
@pytest.mark.parametrize("layout", ["broadcast", "reduce-scatter", "hybrid", "gather-hybrid"])
def test_cost_counts_all_partial_owners_for_each_materialized_output(lanes, layout):
    placement = fm.Placement((3, 5), "ab", "bb")
    dtype = "bfloat16" if lanes == 1 else fm.vector_type("bfloat16", (lanes,))
    tensor = fm.tensor_type(dtype, (1, 240))
    broadcast = (fm.SBP.broadcast(), fm.SBP.broadcast())
    hybrid = (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,)))
    source_sbp = hybrid if "hybrid" in layout else broadcast
    target_sbp = hybrid if layout == "hybrid" else broadcast
    if layout == "reduce-scatter":
        target_sbp = (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1)))
    axes = (0,) if "hybrid" in layout else (0, 1)
    source = fm.DistributedType(tensor, source_sbp, placement, partial=fm.SBP.partial(axes))
    target = fm.DistributedType(tensor, target_sbp, placement)
    inputs = (_node("partial", source), _node("residual", target))
    attrs = {"axis": -1, "use_mean": False}
    result = AddNormStats.infer_type(inputs, attrs)
    factors = AddNormStats.cost_factors(inputs, attrs, result)
    local = fm.local_tensor_type(target)
    elements = prod(d.fixed_value for d in local.shape) * lanes
    fan_in = prod(placement.hierarchy[axis] for axis in axes)
    # Accumulator fan-in, residual read, and the materialized value for stats.
    assert factors.block_local_memory_load_bytes == elements * 2 * (fan_in + 2)
    assert factors.elementwise_operations == elements * (fan_in - 1)
    assert factors.grid_synchronizations == 1


def test_parameter_info_and_dense_type_are_explicit_value_stats_tuple():
    value_type = fm.tensor_type("bfloat16", (2, 16))
    result = AddNormStats.infer_type(
        (_node("partial", value_type), _node("residual", value_type)),
        {"axis": -1, "use_mean": False},
    )

    assert AddNormStats.input.name == "input"
    assert AddNormStats.addend.name == "addend"
    assert result == fm.TupleType((
        value_type,
        fm.tensor_type("float32", (1, 2, 1)),
    ))


def test_sum_partial_can_materialize_to_broadcast_on_any_mesh_shape():
    placement = fm.Placement((3, 5), "ab", "bb")
    tensor = fm.tensor_type("bfloat16", (15, 16))
    partial = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0, 1)),
    )
    materialized = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )

    assert can_materialize_sum_partial(partial, materialized)
    result = AddNormStats.infer_type(
        (_node("partial", partial), _node("residual", materialized)),
        {"axis": 1, "use_mean": False},
    )
    assert result == fm.TupleType((
        materialized,
        fm.DistributedType(
            fm.tensor_type("float32", (1, 15, 1)),
            (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
        ),
    ))


def test_non_sum_or_incompatible_nonpartial_reshard_is_rejected():
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("bfloat16", (8, 16))
    target = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    max_partial = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0,), fm.ReduceOp.MAX),
    )
    wrong_split = fm.DistributedType(
        tensor,
        (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()),
        placement,
    )

    assert not can_materialize_sum_partial(max_partial, target)
    assert not can_materialize_sum_partial(wrong_split, target)
    with pytest.raises(IRSchemaError, match="cannot materialize"):
        AddNormStats.infer_type(
            (_node("input", max_partial), _node("addend", target)),
            {"axis": 1, "use_mean": False},
        )


class _CombineModule(fm.Module):
    def __init__(self, use_mean):
        super().__init__(dialect="ntt", stage="packed", entry="main")
        self.use_mean = use_mean

    def forward(self):
        value_type = fm.tensor_type("float32", (2, 8))
        partial = self.input("partial", value_type)
        residual = self.input("residual", value_type)
        combine = fm.F.ntt.add_norm_stats(
            partial,
            residual,
            axis=-1,
            use_mean=self.use_mean,
            name="combine",
        )
        value, stats = fm.F.tensors.get_items(
            combine, 0, 1, name_prefix="combine_result")
        self.function("main", (partial, residual), (value, stats))


@pytest.mark.parametrize("use_mean", (False, True))
def test_evaluator_returns_materialized_value_and_its_additive_statistics(use_mean):
    module = _CombineModule(use_mean).build()
    partial = torch.randn(2, 8)
    residual = torch.randn(2, 8)
    value, stats = TorchEvaluator(DictWeightResolver({})).run(
        module, {"partial": partial, "residual": residual})

    expected = partial + residual
    torch.testing.assert_close(value, expected)
    expected_fields = [expected.sum(dim=-1, keepdim=True)] if use_mean else []
    expected_fields.append(expected.square().sum(dim=-1, keepdim=True))
    torch.testing.assert_close(stats, torch.stack(expected_fields))


@pytest.mark.parametrize("use_mean", (False, True))
def test_evaluator_reduces_vector_lanes_as_part_of_the_logical_axis(use_mean):
    class VectorCombine(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="packed", entry="main")

        def forward(self):
            value_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 2))
            partial = self.input("partial", value_type)
            residual = self.input("residual", value_type)
            result = fm.F.ntt.add_norm_stats(
                partial,
                residual,
                axis=1,
                use_mean=use_mean,
                name="result",
            )
            value, stats = fm.F.tensors.get_items(
                result, 0, 1, name_prefix="result_field")
            self.function("main", (partial, residual), (value, stats))

    module = VectorCombine().build()
    partial = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    residual = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    value, stats = TorchEvaluator(DictWeightResolver({})).run(
        module, {"partial": partial, "residual": residual})

    expected = partial + residual
    logical = expected.reshape(1, 16).float()
    expected_fields = [logical.sum(dim=-1, keepdim=True)] if use_mean else []
    expected_fields.append(logical.square().sum(dim=-1, keepdim=True))
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(stats, torch.stack(expected_fields))
