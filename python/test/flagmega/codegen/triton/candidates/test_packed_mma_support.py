# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The BF16 MMA variant has an explicit dtype and target contract."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.iluvatar_bi_v150 import IluvatarBiV150Target
from triton.flagmega.targets.nvidia_sm120 import NvidiaSm120Target


@pytest.mark.parametrize("source,output", [("bfloat16", "bfloat16"), ("bfloat16", "float32"),
                                         ("float32", "bfloat16")])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("reduction", [64, 1024])
def test_packed_projection_only_accepts_its_declared_contract(source, output, partitioned, reduction):
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    k = fm.SBP.split_contiguous((0,), 32) if partitioned else b
    kw = fm.SBP.split_contiguous((0,), 2) if partitioned else b

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.DistributedType(fm.tensor_type(source, (1, reduction)), (b, k), placement))
            w = self.input("w", fm.DistributedType(fm.tensor_type(fm.vector_type(source, (8, 2, 8)),
                                                                 (reduction // 16, 4)),
                                                  (kw, b), placement))
            none = fm.F.builtin.none()
            y = fm.F.ntt.packed_matmul(x, w, none, none, output_data_type=output, fused_reduce=partitioned,
                                      name="projection", metadata={
                "selected_vectorization": "vectorization.matmul.n", "selected_vector_axes": (1,),
                "selected_vector_lanes": (8,),
            })
            self.function("main", (x, w), (y,))

    module = Graph(dialect="ntt", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(p for p in proposed.selection_points if p.owner == "projection")
    identities = {c.id for c in point.candidates}
    assert ("tir.dense_matmul.packed_k_major_mma" in identities) == (source == output == "bfloat16"
                                                                and not partitioned)
    assert ("tir.dense_matmul.packed_k_major_threaded_fma" in identities) == (
        source == "bfloat16" and output == "float32" and reduction % 1024 == 0 and not partitioned
    )


def test_packed_mma_does_not_leak_into_corex_or_unsupported_nvidia_capabilities():
    identity = "tir.dense_matmul.packed_k_major_mma"
    assert identity not in {i.id for i in IluvatarBiV150Target().triton_implementation_model.implementations}
    implementation = NvidiaSm90Target().triton_implementation_model.implementation(identity)
    assert not NvidiaSm90Target().capability.missing(implementation.requires)
    assert NvidiaSm120Target().capability.missing(implementation.requires) == ("mma_v3",)
    assert "tir.dense_matmul.packed_k_major_threaded_fma" not in {
        i.id for i in IluvatarBiV150Target().triton_implementation_model.implementations
    }
