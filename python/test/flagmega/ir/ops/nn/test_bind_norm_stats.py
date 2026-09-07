# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.bind_norm_stats import BindNormStats
from triton.flagmega.ir.ops.nn.norm_stats import NormStats


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def test_bind_norm_stats_accepts_materialized_form_of_partial_statistics():
    placement = fm.Placement((8, 16), "yx", "bb")
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", [1, 2048]),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        placement,
    )
    value = _node("value", value_type)
    partial = NormStats.infer_type((value,), {"axis": -1, "use_mean": False})
    assert isinstance(partial, fm.DistributedType) and partial.partial is not None
    materialized = replace(partial, partial=None)

    result = BindNormStats.infer_type(
        (value, _node("stats", materialized)),
        {"axis": -1, "use_mean": False},
    )
    assert result == materialized
    assert BindNormStats.cost(_node("bind", materialized)).is_complete


def test_bind_norm_stats_rejects_unmaterialized_partial():
    placement = fm.Placement((8,), "x", "b")
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", [1, 64]),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))),
        placement,
    )
    value = _node("value", value_type)
    partial = NormStats.infer_type((value,), {"axis": -1, "use_mean": False})
    with pytest.raises(IRSchemaError, match="materialized non-partial"):
        BindNormStats.infer_type(
            (value, _node("stats", partial)),
            {"axis": -1, "use_mean": False},
        )
