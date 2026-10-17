# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""K partitions must retain the accumulator until the global result rounds."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.selection import override_plan


def matmul_module(wide):
    mesh = fm.Placement((8, 16), "yx", "bb")
    left = fm.tensor_type("bfloat16", (1, 1024))
    right = fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (64, 128))
    lhs_type = fm.DistributedType(left, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 128)), mesh)
    rhs_type = fm.DistributedType(right, (fm.SBP.split_contiguous((0,), 8), fm.SBP.split_contiguous((1,), 8)), mesh)

    class Graph(fm.Module):
        def forward(self):
            lhs = self.input("lhs", left, id="lhs")
            rhs = self.input("rhs", right, id="rhs")
            a = fm.F.distributed.force_boxing(lhs, lhs_type, name="load_a")
            b = fm.F.distributed.force_boxing(rhs, rhs_type, name="load_b")
            none = fm.F.builtin.none()
            partial = fm.F.ntt.packed_matmul(a, b, none, none,
                                            output_data_type="float32" if wide else "bfloat16", name="dot",
                                            metadata={"selected_vectorization": "vectorization.matmul.n",
                                                      "selected_vector_axes": (1,),
                                                      "selected_vector_lanes": (8,)})
            complete = fm.F.distributed.force_boxing(partial, fm.logical_type(partial.type), name="reduce")
            dense = fm.F.tensors.unpack(complete, axes=(1,) * len(fm.logical_type(complete.type).dtype.lanes))
            result = fm.F.tensors.cast(dense, "bfloat16", name="round") if wide else dense
            self.function("main", (lhs, rhs), (result,))

    return Graph(dialect="distributed", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("wide", [False, True])
def test_partition_rounding_does_not_erase_cancellation_residue(tmp_path, wide):
    proposed = Compiler().compile(matmul_module(wide), stop_after="propose-tir").module
    selected = Compiler().run_stage(proposed, "lower-tir", plan=override_plan(proposed, (
        ("tir.dot", "tir.dense_matmul.split_k_n_packed_k_major_gemv"),
    ))).module
    compiled = Compiler().compile(selected).module
    artifact = write_artifact(compiled, tmp_path / "dot", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    lhs = torch.ones((1, 1024), dtype=torch.bfloat16, device="cuda")
    rhs = torch.zeros((1024, 1024), dtype=torch.bfloat16, device="cuda")
    rhs[:, 0] = 1
    rhs[:, 1] = 2 ** -8
    rhs[:, 128] = -1
    packed = rhs.T.reshape(64, 2, 8, 128, 8).permute(0, 3, 4, 1, 2).contiguous()
    output = torch.empty((1, 1024), dtype=torch.bfloat16, device="cuda")
    try:
        runtime.prepare(lhs, packed, output=output)
        runtime.run_into(output, lhs, packed)
        torch.testing.assert_close(output, torch.full_like(output, 2 ** -8), rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
