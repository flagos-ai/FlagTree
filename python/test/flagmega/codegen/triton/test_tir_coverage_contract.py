# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.lowering import TritonTirLoweringPolicy
from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.targets import NvidiaSm90Target


def _unary_module(op: str) -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="distributed")
    value_type = fm.tensor_type("float32", (16,))
    value = builder.var("value", value_type, id="value")
    if op == "tensors.cast":
        result = builder.call(op, (value,), value_type, id="result", attrs={"dtype": "float32"})
    else:
        result = builder.call(op, (value,), value_type, id="result")
    builder.function("main", (value,), (result,))
    return builder.build(entry="main")


@pytest.mark.parametrize("op,variant", [
    ("math.silu", "silu"),
    ("tensors.cast", "cast"),
])
def test_portable_elementwise_candidates_have_real_generic_templates(op, variant):
    proposed = NvidiaSm90Target().propose_tir(_unary_module(op))
    point = next(point for point in proposed.selection_points if point.id == "tir.result")
    candidate = point.candidates[0]

    assert candidate.id == f"tir.elementwise.{variant}.scalar"
    assert candidate.facts["portable_triton"] is True
    assert "requires" not in candidate.facts
    assert TritonTemplateRegistry().resolve(
        KernelTemplateSpec("elementwise", variant, "nvidia", "sm90")
    ) == f"kernels/elementwise/{variant}.py.jinja"


def test_lowering_rejects_unreviewed_op_instead_of_fabricating_generic_kernel():
    builder = fm.IRBuilder(dialect="high_level", stage="frozen_constants")
    value_type = fm.tensor_type("float32", (16,))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    result = builder.call("tensors.concat", (lhs, rhs), fm.tensor_type("float32", (32,)), id="result", attrs={"axis": 0})
    builder.function("main", (lhs, rhs), (result,))
    module = builder.build(entry="main")

    with pytest.raises(IRVerificationError, match="No reviewed Triton TIR candidate.*tensors.concat"):
        TritonTirLoweringPolicy().lower(module, NvidiaSm90Target())


def test_packed_dense_shape_without_a_real_template_is_not_advertised():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(
                dialect="high_level",
                stage="frozen_constants",
                entry="main",
            )

        def forward(self):
            lhs = self.input(
                "lhs", fm.tensor_type("bfloat16", (1, 1024)), id="lhs"
            )
            weight = self.input(
                "weight",
                fm.tensor_type("bfloat16", (64, 1, 2, 64)),
                id="weight",
            )
            result = fm.F.math.packed_dense_matmul(lhs, weight, name="result")
            self.function("main", (lhs, weight), (result,))

    target = NvidiaSm90Target()
    proposed = target.propose_tir(Graph().build())

    assert "tir.result" not in {point.id for point in proposed.selection_points}
    with pytest.raises(
        IRVerificationError,
        match="No reviewed Triton TIR candidate.*math.packed_dense_matmul",
    ):
        TritonTirLoweringPolicy().lower(proposed, target)
