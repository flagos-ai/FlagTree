# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


LAYOUT = ("seq", "head", "dim")


def _typed(name: str, value_type: fm.IRType) -> fm.Node:
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _infer(
    query_type: fm.IRType,
    *,
    hidden_size=32,
    split_axis=0,
    split_count=8,
):
    layer_type = fm.tensor_type("int32", ())
    if isinstance(query_type, fm.DistributedType):
        layer_type = fm.DistributedType(layer_type, (), query_type.placement)
    return fm.get_definition("ntt.paged_attention_partial").infer_type(
        (
            _typed("query", query_type),
            _typed("state", fm.RefType("paged_attention_kv_cache")),
            _typed("layer", layer_type),
        ),
        {
            "scale": 8**-0.5,
            "layout": LAYOUT,
            "hidden_size": hidden_size,
            "split_hierarchy_axis": split_axis,
            "split_count": split_count,
        },
    )


def test_logical_partial_uses_fp32_max_sum_and_scalar_accumulator_states():
    result = _infer(fm.tensor_type("bfloat16", (1, 4, 8)))

    assert result == fm.TupleType((
        fm.tensor_type("float32", (1, 4, 1)),
        fm.tensor_type("float32", (1, 4, 1)),
        fm.tensor_type("float32", (1, 4, 8)),
    ))


def test_vector_query_is_scalarized_along_the_semantic_dim_axis():
    result = _infer(
        fm.tensor_type(fm.vector_type("bfloat16", (2, 2)), (1, 4, 2))
    )

    assert result.fields[0] == fm.tensor_type("float32", (1, 4, 1))
    assert result.fields[2] == fm.tensor_type("float32", (1, 4, 8))


def test_distributed_partial_uses_unused_physical_block_axis_with_max_sum_sum():
    placement = fm.Placement((8, 16), "yx", "bb")
    query = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 8)),
        (
            fm.SBP.broadcast(),
            fm.SBP.split_contiguous((1,), 1),
            fm.SBP.broadcast(),
        ),
        placement,
    )

    result = _infer(query, hidden_size=128)

    assert isinstance(result, fm.TupleType)
    assert tuple(field.axis_policies for field in result.fields) == (
        query.axis_policies,
        query.axis_policies,
        query.axis_policies,
    )
    assert tuple(field.partial for field in result.fields) == (
        fm.SBP.partial((0,), fm.ReduceOp.MAX),
        fm.SBP.partial((0,), fm.ReduceOp.SUM),
        fm.SBP.partial((0,), fm.ReduceOp.SUM),
    )


def test_partial_rejects_a_split_axis_already_used_by_query_layout():
    placement = fm.Placement((8, 16), "yx", "bb")
    query = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 8)),
        (
            fm.SBP.broadcast(),
            fm.SBP.split_contiguous((0,), 1),
            fm.SBP.broadcast(),
        ),
        placement,
    )

    with pytest.raises(IRSchemaError, match="unused physical block hierarchy axis"):
        _infer(query, hidden_size=128)


@pytest.mark.parametrize(
    ("placement", "split_count"),
    (
        (fm.Placement((8, 16), "yx", "cb"), 8),
        (fm.Placement((8, 16), "yx", "bb"), 4),
    ),
)
def test_partial_rejects_non_block_or_extent_mismatched_split_axis(
    placement, split_count
):
    query = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 8)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )

    with pytest.raises(IRSchemaError, match="physical block hierarchy axis"):
        _infer(query, hidden_size=128, split_count=split_count)
