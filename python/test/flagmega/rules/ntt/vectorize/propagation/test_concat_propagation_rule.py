# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.vectorize.propagation import concat_propagation_rules


def _rule(name):
    return next(rule for rule in concat_propagation_rules() if rule.name == name)


def _assert_equivalent(original, rewritten, inputs):
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, inputs)[0],
        evaluator.run(original, inputs)[0],
    )


def test_vectorize_concat_packs_every_operand(materialize_rewrite):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (2, 16)), id="lhs")
            rhs = self.input("rhs", fm.tensor_type("bfloat16", (3, 16)), id="rhs")
            concat = fm.F.tensors.concat(lhs, rhs, axis=0, name="concat")
            root = fm.F.tensors.pack(concat, (8,), axes=(1,), name="root")
            self.function("main", (lhs, rhs), (root,))

    module = Graph().build()
    result = _rule("VectorizeConcatPropagation").apply(
        module.node_map["root"], module
    )

    assert result is not None
    assert [value.op for value in result.prefix_nodes] == [
        "tensors.pack", "tensors.pack",
    ]
    assert result.replacement.op == "tensors.concat"
    rewritten = materialize_rewrite(module, result)
    _assert_equivalent(
        module,
        rewritten,
        {
            "lhs": torch.randn((2, 16), dtype=torch.bfloat16),
            "rhs": torch.randn((3, 16), dtype=torch.bfloat16),
        },
    )


def test_concat_devectorize_packs_only_scalar_operands(materialize_rewrite):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            vector = self.input(
                "vector",
                fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 2)),
                id="vector",
            )
            scalar = self.input(
                "scalar", fm.tensor_type("bfloat16", (3, 16)), id="scalar"
            )
            unpacked = fm.F.tensors.unpack(vector, axes=(1,), name="unpacked")
            root = fm.F.tensors.concat(unpacked, scalar, axis=0, name="root")
            self.function("main", (vector, scalar), (root,))

    module = Graph().build()
    result = _rule("ConcatDevectorizePropagation").apply(
        module.node_map["root"], module
    )

    assert result is not None
    assert [value.op for value in result.prefix_nodes] == [
        "tensors.pack", "tensors.concat",
    ]
    assert result.replacement.op == "tensors.unpack"
    rewritten = materialize_rewrite(module, result)
    _assert_equivalent(
        module,
        rewritten,
        {
            "vector": torch.randn((2, 2, 8), dtype=torch.bfloat16),
            "scalar": torch.randn((3, 16), dtype=torch.bfloat16),
        },
    )


def test_vectorize_concat_rejects_operands_that_are_not_individually_packable():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (2, 7)), id="lhs")
            rhs = self.input("rhs", fm.tensor_type("bfloat16", (2, 9)), id="rhs")
            concat = fm.F.tensors.concat(lhs, rhs, axis=1, name="concat")
            root = fm.F.tensors.pack(concat, (8,), axes=(1,), name="root")
            self.function("main", (lhs, rhs), (root,))

    module = Graph().build()

    assert _rule("VectorizeConcatPropagation").apply(
        module.node_map["root"], module
    ) is None


def test_concat_devectorize_rejects_different_lane_contracts():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            lhs = self.input(
                "lhs", fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 2)),
                id="lhs",
            )
            rhs = self.input(
                "rhs", fm.tensor_type(fm.vector_type("bfloat16", (4,)), (2, 4)),
                id="rhs",
            )
            lhs_scalar = fm.F.tensors.unpack(lhs, axes=(1,), name="lhs_scalar")
            rhs_scalar = fm.F.tensors.unpack(rhs, axes=(1,), name="rhs_scalar")
            root = fm.F.tensors.concat(
                lhs_scalar, rhs_scalar, axis=0, name="root"
            )
            self.function("main", (lhs, rhs), (root,))

    module = Graph().build()

    assert _rule("ConcatDevectorizePropagation").apply(
        module.node_map["root"], module
    ) is None
