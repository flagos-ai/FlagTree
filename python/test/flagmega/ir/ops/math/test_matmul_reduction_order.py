# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""An ordered projection cannot turn into independently summed K shards."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.math.matmul import MatMul
from triton.flagmega.ir.ops.ntt.packed_matmul import PackedMatMul
from triton.flagmega.rules.ntt.vectorize.matmul import VectorizeMatMul


def operands():
    lhs = fm.Node("x", "builtin.var", (), fm.tensor_type("bfloat16", (32, 128)))
    rhs = fm.Node("w", "builtin.var", (), fm.tensor_type("bfloat16", (64, 128)))
    return lhs, rhs


def test_default_reduction_order_keeps_existing_ir_canonical():
    assert "reduction_order" not in MatMul.normalize_attrs({})
    assert "reduction_order" not in MatMul.normalize_attrs({"reduction_order": "relaxed"})


def test_ordered_reduction_survives_every_vectorization():
    lhs, rhs = operands()
    prepared = MatMul.prepare((lhs, rhs), {"transpose_b": True, "reduction_order": "mma_k16"})
    projection = fm.Node("projection", MatMul.op_name, (lhs.id, rhs.id), prepared.result_type,
                         prepared.effect, prepared.attrs)
    module = fm.IRModule(dialect="hl", stage="imported", nodes=(lhs, rhs, projection), functions=(
        fm.Function("main", (lhs.id, rhs.id), (projection.id,)),), entry="main")
    rule = VectorizeMatMul()
    candidates = rule.candidates(projection, module)
    assert {c.parameters["kind"] for c in candidates} == {"n", "mn", "mkn"}
    for candidate in candidates:
        result = rule.rewrite(projection, module, candidate)
        compute = next(n for n in result.prefix_nodes if n.op == "math.vectorized_matmul")
        assert compute.attrs["reduction_order"] == "mma_k16"


@pytest.mark.parametrize("packed", [False, True])
def test_ordered_reduction_rejects_cross_owner_k_reassociation(packed):
    placement = fm.Placement((2, 2), "yx", "bb")
    b, split = fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 1)
    lhs, rhs = operands()
    lhs = fm.Node("x", "builtin.var", (), fm.DistributedType(lhs.type, (b, split), placement))
    if packed:
        rhs_type = fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (8, 8))
        rhs = fm.Node("w", "builtin.var", (), fm.DistributedType(rhs_type, (split, b), placement))
        none = fm.Node("none", "builtin.none", (), fm.NoneType())
        definition, inputs = PackedMatMul, (lhs, rhs, none, none)
        attrs = definition.normalize_attrs({"reduction_order": "mma_k16"})
    else:
        rhs = fm.Node("w", "builtin.var", (), fm.DistributedType(rhs.type, (b, split), placement))
        definition, inputs = MatMul, (lhs, rhs)
        attrs = definition.normalize_attrs({"transpose_b": True, "reduction_order": "mma_k16"})
    with pytest.raises(IRSchemaError, match="ordered.*reduction"):
        definition.infer_type(inputs, attrs)


@pytest.mark.parametrize("order", [None, "simt", "mma", 16, True])
def test_unknown_reduction_order_is_rejected(order):
    with pytest.raises(IRSchemaError, match="reduction_order"):
        MatMul.normalize_attrs({"reduction_order": order})
