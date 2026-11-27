# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.vectorize.propagation import binary_propagation_rules


def _rule(name):
    return next(rule for rule in binary_propagation_rules() if rule.name == name)


def _pack_binary_module(op="add"):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value_type = fm.tensor_type("bfloat16", (2, 16))
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            scalar = getattr(fm.F.math, op)(lhs, rhs, name="scalar")
            root = fm.F.tensors.pack(scalar, (8,), axes=(1,), name="root")
            self.function("main", (lhs, rhs), (root,))

    return Graph().build()


@pytest.mark.parametrize("op", ["add", "mul"])
def test_vectorize_binary_propagation_matches_and_rewrites_one_rule(materialize_rewrite, op):
    module = _pack_binary_module(op)
    result = _rule("VectorizeBinaryPropagation").apply(module.node_map["root"], module)
    assert result is not None
    assert [node.op for node in result.prefix_nodes] == ["tensors.pack", "tensors.pack"]
    assert result.replacement.op == "math.vectorized_binary"
    assert result.replacement.attrs["binary_op"] == op
    rewritten = materialize_rewrite(module, result)

    lhs = torch.randn(2, 16, dtype=torch.bfloat16)
    rhs = torch.randn(2, 16, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"lhs": lhs, "rhs": rhs})[0],
        evaluator.run(module, {"lhs": lhs, "rhs": rhs})[0],
    )


@pytest.mark.parametrize(
    "side,rule_name",
    [("lhs", "BinaryDevectorizeLhsPropagation"), ("rhs", "BinaryDevectorizeRhsPropagation")],
)
def test_one_sided_devectorize_propagation_packs_the_other_operand(
    materialize_rewrite, side, rule_name,
):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            vector = self.input(
                "vector", fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 2)), id="vector",
            )
            scalar = self.input("scalar", fm.tensor_type("bfloat16", (2, 16)), id="scalar")
            unpacked = fm.F.tensors.unpack(vector, axes=(1,), name="unpacked")
            operands = (unpacked, scalar) if side == "lhs" else (scalar, unpacked)
            root = fm.F.math.add(*operands, name="root")
            self.function("main", (vector, scalar), (root,))

    module = Graph().build()
    result = _rule(rule_name).apply(module.node_map["root"], module)
    assert result is not None
    assert [node.op for node in result.prefix_nodes] == ["tensors.pack", "math.vectorized_binary"]
    assert result.replacement.op == "tensors.unpack"
    packed = result.prefix_nodes[0]
    assert packed.inputs == ("scalar",)
    rewritten = materialize_rewrite(module, result)

    vector = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    scalar = torch.randn(2, 16, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"vector": vector, "scalar": scalar})[0],
        evaluator.run(module, {"vector": vector, "scalar": scalar})[0],
    )


def test_binary_rules_do_not_match_unrelated_roots():
    module = _pack_binary_module()
    scalar = module.node_map["scalar"]
    assert _rule("VectorizeBinaryPropagation").apply(scalar, module) is None
    assert _rule("BinaryDevectorizeLhsPropagation").apply(scalar, module) is None
    assert _rule("BinaryDevectorizeRhsPropagation").apply(scalar, module) is None


def test_devectorize_rule_rejects_scalar_operand_without_unpack():
    module = _pack_binary_module()
    assert _rule("BinaryDevectorizeLhsPropagation").apply(module.node_map["scalar"], module) is None
