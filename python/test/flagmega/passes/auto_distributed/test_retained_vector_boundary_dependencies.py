# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts


def test_retained_result_boundary_keeps_its_physical_slice_dependency():

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type(fm.vector_type("bfloat16", (8, )), (2, 4)), id="value")
            compute = fm.F.math.vectorized_unary(
                value, unary_op="sigmoid", name="compute", metadata={
                    "vectorization_root": "boundary", "vectorization_semantic_id": "sigmoid", "vectorized_from":
                    "math.sigmoid", "vectorization_internal": True, "vectorization_candidate":
                    "vectorization.last_axis", "vector_axes": (1, ), "vector_lanes": (8, )
                })
            sliced = fm.F.tensors.slice(
                value, starts=(1, ), ends=(3, ), axes=(1, ), name="slice", metadata={
                    "vectorization_root": "old_crop", "vectorization_semantic_id": "scalar_crop", "vectorized_from":
                    "tensors.slice", "vectorization_internal": True, "vectorization_candidate":
                    "vectorization.propagated", "vector_axes": (1, ), "vector_lanes": (8, ), "vectorization_inputs":
                    ("value", ), "vectorization_attrs":
                    {"starts": (8, ), "ends": (24, ), "axes": (1, ), "steps": (1, )}
                })
            output = fm.F.tensors.concat(
                compute, sliced, axis=1, name="concat", metadata={
                    "vectorization_internal": True, "vectorized_from": "tensors.concat", "vectorization_semantic_id":
                    "boundary"
                })
            self.function("main", (value, ), (output, ))

    original = Graph(dialect="ntt", stage="distributed", entry="main").build()
    result = fm.verify_module(lower_vectorization_contracts(original))
    assert result.node_map["slice"] == original.node_map["slice"]
    assert result.node_map["concat"].inputs == ("compute", "slice")
