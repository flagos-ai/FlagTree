# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


def projection(k, n, *, output_split=False):
    placement = fm.Placement((2, 2), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    k_axes = (0, ) if output_split or k % 64 else (0, 1)
    k_policy = fm.SBP.split_contiguous(k_axes, (k // 16) // (2**len(k_axes)))
    n_policy = fm.SBP.split_block_cyclic((1, ), 1) if output_split else broadcast

    class Graph(fm.Module):

        def forward(self):
            lhs = self.input(
                "lhs",
                fm.DistributedType(fm.tensor_type("bfloat16", (1, k)),
                                   (broadcast, fm.scale_split_units(k_policy, 16, 1)), placement))
            rhs = self.input(
                "rhs",
                fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (k // 16, n // 8)),
                                   (k_policy, n_policy), placement))
            none = fm.F.builtin.none()
            partial = fm.F.ntt.packed_matmul(
                lhs, rhs, none, none, output_data_type="bfloat16", name="projection", metadata={
                    "selected_vectorization": "vectorization.matmul.n",
                    "selected_vector_axes": (1, ),
                    "selected_vector_lanes": (8, ),
                })
            result = fm.F.tensors.unpack(fm.F.distributed.boxing(partial, partial.type.tensor), axes=(1, ))
            self.function("main", (lhs, rhs), (result, ))

    return Graph(dialect="ntt", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": placement.to_data()}}).build()


@pytest.mark.parametrize("k,n,output_split", [(64, 8, False), (96, 24, False), (96, 48, True)])
def test_packed_partial_projection_accepts_actual_masked_simt_capability(k, n, output_split):
    proposed = NvidiaSm90Target().propose_tir(projection(k, n, output_split=output_split))
    variant = "split_k_n_packed_k_major_gemv" if output_split else "split_k_packed_k_major_gemv"
    point = next(point for point in proposed.selection_points if point.owner == "projection")
    assert f"tir.dense_matmul.{variant}" in {candidate.id for candidate in point.candidates}


def test_packed_partial_tail_requires_masked_capability_and_does_not_relax_descriptors():
    model = sm90_triton_implementation_model()
    model = replace(
        model, implementations=tuple(
            replace(value, contract={**value.contract, "supports_masked_tiles": False}) if value.variant in
            {"split_k_packed_k_major_gemv", "split_k_n_packed_k_major_gemv"} else value
            for value in model.implementations))
    proposed = NvidiaSm90Target(triton_implementation_model=model).propose_tir(projection(64, 8))
    assert not any(point.owner == "projection" for point in proposed.selection_points)
