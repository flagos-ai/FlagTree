# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts


@pytest.mark.parametrize("columns", [1, 5, 9])
def test_padded_projection_preserves_physical_packed_function_parameter(columns):
    groups = (columns + 7) // 8
    metadata = {
        "vectorization_candidate": "vectorization.matmul.n",
        "vectorization_root": "projection",
        "vector_axes": (1, ),
        "vector_lanes": (8, ),
        "vectorized_from": "math.matmul",
        "vectorization_attrs": {"transpose_a": False, "transpose_b": True},
    }

    class Graph(fm.Module):

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (2, 32)), id="lhs")
            rhs = self.input("rhs", fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (2, groups)), id="rhs")
            none = fm.F.builtin.none()
            compute = fm.F.ntt.packed_matmul(
                lhs, rhs, none, none, output_data_type="bfloat16", name="compute", metadata={
                    **metadata,
                    "vectorization_internal": True,
                    "vectorization_role": "compute",
                })
            unpacked = fm.F.tensors.unpack(
                compute, axes=(1, ), name="unpack", metadata={
                    "vectorization_root": "projection",
                    "vectorization_internal": True,
                    "vectorization_role": "unpack",
                })
            result = fm.F.tensors.slice_to_shape(unpacked, (2, columns), name="projection", metadata=metadata)
            self.function("main", (lhs, rhs), (result, ))

    before = Graph(dialect="ntt", stage="matmul_norm_stats_lowered", entry="main").build()
    fm.verify_module(before)
    after = lower_vectorization_contracts(before)
    fm.verify_module(after)
    assert after.node_map["compute"].op == "ntt.packed_matmul"
    assert after.node_map["compute"].inputs[:2] == ("lhs", "rhs")
    assert after.node_map["projection"].op == "tensors.slice_to_shape"
    generator = torch.Generator().manual_seed(317)
    lhs = torch.randn(2, 32, generator=generator).bfloat16()
    logical_rhs = torch.zeros(32, groups * 8, dtype=torch.bfloat16)
    logical_rhs[:, :columns] = torch.randn(32, columns, generator=generator).bfloat16()
    rhs = logical_rhs.reshape(2, 2, 8, groups, 8).permute(0, 3, 4, 1, 2).contiguous()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    actual = evaluator.run(after, {"lhs": lhs, "rhs": rhs})[0]
    torch.testing.assert_close(actual, lhs @ logical_rhs[:, :columns], rtol=0, atol=0)
    torch.testing.assert_close(actual, evaluator.run(before, {"lhs": lhs, "rhs": rhs})[0], rtol=0, atol=0)
