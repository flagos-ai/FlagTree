# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear_combine import (
    PackedQKVParallelLinearCombine,
    can_materialize_packed_qkv,
)


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _qkv_types(*, partial_axes=(0,)):
    placement = fm.Placement((8, 16), "yx", "bb")
    tensors = (
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256)),
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 128)),
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 128)),
    )
    output = fm.TupleType(tuple(
        fm.DistributedType(
            tensor,
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)),
            placement,
        )
        for tensor in tensors
    ))
    partial = fm.TupleType(tuple(
        fm.DistributedType(
            tensor,
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)),
            placement,
            partial=fm.SBP.partial(partial_axes),
        )
        for tensor in tensors
    ))
    return partial, output


def test_combine_materializes_three_coupled_sum_partials():
    partial, output = _qkv_types()

    assert PackedQKVParallelLinearCombine.qkv.name == "qkv"
    assert PackedQKVParallelLinearCombine.output_type.name == "output_type"
    assert can_materialize_packed_qkv(partial, output)
    assert PackedQKVParallelLinearCombine.infer_type(
        (_node("qkv", partial),), {"output_type": output}
    ) == output


def test_combine_rejects_different_partial_axes_and_logical_tensors():
    partial, output = _qkv_types()
    fields = list(partial.fields)
    fields[2] = fm.DistributedType(
        fields[2].tensor,
        fields[2].axis_policies,
        fields[2].placement,
        partial=fm.SBP.partial((1,)),
    )
    mismatched_partial = fm.TupleType(tuple(fields))

    assert not can_materialize_packed_qkv(mismatched_partial, output)
    with pytest.raises(IRSchemaError, match="cannot materialize"):
        PackedQKVParallelLinearCombine.infer_type(
            (_node("qkv", mismatched_partial),), {"output_type": output}
        )


def test_combine_accepts_exact_materialized_identity_without_distribution():
    output = fm.TupleType((
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 32)),
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 16)),
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 16)),
    ))

    assert can_materialize_packed_qkv(output, output)
    assert PackedQKVParallelLinearCombine.infer_type(
        (_node("qkv", output),), {"output_type": output}
    ) == output
