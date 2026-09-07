# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def _module():
    class Projection(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, 64)), id="lhs")
            rhs = self.weight(
                "rhs",
                fm.tensor_type("bfloat16", (64, 128)),
                source="memory",
                key="rhs",
                id="rhs",
            )
            output = fm.F.math.matmul(lhs, rhs, name="output")
            self.function("main", (lhs,), (output,))

    return Projection().build()


def test_auto_packing_matches_vectorized_owner_and_keeps_stable_root_selection_id():
    proposed = Compiler().compile(_module(), stop_after="propose-packing").module
    compute = proposed.node_map["output.vectorized.compute"]
    point = next(point for point in proposed.selection_points if point.id == "packing.output")

    assert compute.op == "math.vectorized_matmul"
    assert point.owner == compute.id
    assert point.default_candidate == "packing.k_major_n8_k16"
    assert next(
        candidate for candidate in point.candidates
        if candidate.id == point.default_candidate
    ).facts["preserves_vectorized_result"]


def test_k_major_rule_rewrites_exact_nncase_recipe_and_preserves_vector_result():
    original = _module()
    vectorized = Compiler().compile(original, stop_after="apply-vectorization").module
    packed = Compiler().compile(original, stop_after="apply-packing").module
    compute = packed.node_map["output.vectorized.compute"]

    assert compute.op == "ntt.packed_matmul"
    assert compute.type == vectorized.node_map[compute.id].type
    assert isinstance(fm.logical_type(compute.type), fm.TensorType)
    packed_rhs = packed.node_map[compute.inputs[1]]
    assert packed_rhs.type == fm.tensor_type(
        fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (4, 16)
    )
    assert [
        node.op for node in packed.nodes
        if node.id.startswith("output.vectorized.compute.rhs_k_major.")
    ] == [
        "tensors.unpack",
        "tensors.pack",
        "tensors.pack",
        "tensors.pack",
        "tensors.permute",
        "tensors.permute",
    ]

    generator = torch.Generator().manual_seed(20260904)
    lhs = torch.randn((1, 64), generator=generator, dtype=torch.bfloat16)
    rhs = torch.randn((64, 128), generator=generator, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({"rhs": rhs}))
    torch.testing.assert_close(
        evaluator.run(packed, {"lhs": lhs})[0],
        evaluator.run(original, {"lhs": lhs})[0],
        rtol=0,
        atol=0,
    )
