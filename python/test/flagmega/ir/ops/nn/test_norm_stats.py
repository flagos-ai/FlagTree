# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.errors import IRSchemaError


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def test_norm_stats_parameter_info_and_dense_type_match_nncase_contract():
    value = _node("value", fm.tensor_type("bfloat16", [3, 5, 7]))
    result = NormStats.infer_type((value,), {"axis": 1, "use_mean": True})

    assert NormStats.value.name == "input"
    assert NormStats.value.type_pattern.reason == "is_tensor"
    assert result == fm.tensor_type("float32", [2, 3, 1, 1])


def test_norm_stats_hidden_split_becomes_additive_partial_on_two_dimensional_mesh():
    placement = fm.Placement((2, 4), "yx", "bb")
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", [8, 32]),
        (fm.SBP.split_contiguous((0,)), fm.SBP.split_contiguous((1,))),
        placement,
    )
    result = NormStats.infer_type(
        (_node("value", value_type),), {"axis": 1, "use_mean": False})

    assert isinstance(result, fm.DistributedType)
    assert result.tensor == fm.tensor_type("float32", [1, 8, 1])
    assert result.axis_policies == (
        fm.SBP.broadcast(), fm.SBP.split_contiguous((0,)), fm.SBP.broadcast())
    assert result.partial == fm.SBP.partial((1,))


class _StatsModule(fm.Module):
    def __init__(self, use_mean):
        super().__init__(dialect="nn", stage="imported", entry="main")
        self.use_mean = use_mean

    def forward(self):
        value = self.input("value", fm.tensor_type("float32", [2, 3, 4]))
        stats = fm.F.nn.norm_stats(
            value, axis=1, use_mean=self.use_mean, name="stats")
        self.function("main", (value,), (stats,))


def test_norm_stats_evaluator_returns_sum_then_sum_of_squares():
    module = _StatsModule(True).build()
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 8
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {"value": value})[0]

    expected = torch.cat((
        value.sum(dim=(1, 2), keepdim=True).unsqueeze(0),
        value.square().sum(dim=(1, 2), keepdim=True).unsqueeze(0),
    ), dim=0)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("group", [0, -2, 3, True])
def test_norm_stats_rejects_invalid_ordered_group(group):
    with pytest.raises(IRSchemaError, match="positive power of two"):
        NormStats.normalize_attrs({"axis": -1, "use_mean": False, "reduction_group_size": group})


def test_norm_stats_default_group_preserves_canonical_attrs():
    assert NormStats.normalize_attrs({"axis": -1, "use_mean": False}) == {
        "axis": -1, "use_mean": False}


@pytest.mark.parametrize("seed,order", [(-1, "group_fma"), (8, "group_fma"),
                                         (True, "group_fma"), (1, "tree")])
def test_norm_stats_rejects_invalid_group_fma_seed(seed, order):
    with pytest.raises(IRSchemaError, match="reduction_group_seed_index"):
        NormStats.normalize_attrs({"axis": -1, "use_mean": False,
            "reduction_group_size": 8, "reduction_order": order,
            "reduction_group_seed_index": seed})


def test_norm_stats_ordered_groups_include_vector_lanes():
    vector = fm.VectorType(fm.DType.BFLOAT16, (32,))
    value = _node("value", fm.tensor_type(vector, [1, 8]))
    result = NormStats.infer_type((value,), {
        "axis": -1, "use_mean": False, "reduction_group_size": 2})
    assert result == fm.tensor_type("float32", [1, 1, 1])


def test_norm_stats_ordered_groups_require_complete_reduction_owners():
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", [1, 256]),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))),
        fm.Placement((2,), "x", "b"),
    )
    with pytest.raises(IRSchemaError, match="broadcast reduction axes"):
        NormStats.infer_type((_node("value", value_type),), {
            "axis": -1, "use_mean": False, "reduction_group_size": 2})


def test_norm_stats_ordered_groups_reject_indivisible_capacity():
    value = _node("value", fm.tensor_type("bfloat16", [1, 7]))
    with pytest.raises(IRSchemaError, match="fixed scalar reduction capacity"):
        NormStats.infer_type((value,), {
            "axis": -1, "use_mean": False, "reduction_group_size": 2})


@pytest.mark.parametrize("attrs", [
    {"reduction_order": "invalid"},
    {"reduction_order": "tile_fma"},
    {"reduction_block_size": 128},
    {"reduction_order": "tile_fma", "reduction_block_size": 3},
    {"reduction_order": "tile_fma", "reduction_group_size": 8, "reduction_block_size": 4},
    {"reduction_order": "tree", "reduction_block_traversal": "reverse"},
    {"reduction_order": "tree", "reduction_block_traversal": []},
    {"reduction_order": "tree", "reduction_tree_group_size": 3},
    {"reduction_tree_group_size": 32},
])
def test_norm_stats_rejects_incomplete_tiled_contract(attrs):
    with pytest.raises(IRSchemaError, match="reduction_order|reduction_block_size|reduction_block_traversal|reduction_tree_group_size"):
        NormStats.normalize_attrs({"axis": -1, "use_mean": False, **attrs})
