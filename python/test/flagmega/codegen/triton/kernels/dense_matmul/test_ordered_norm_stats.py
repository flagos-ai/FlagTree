# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ordered projections retain BF16 stores inside multirow residual fusion."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("rows,split_rows,residual_dtype", [(1, False, "bfloat16"), (32, False, "bfloat16"),
                                                           (32, True, "bfloat16"), (32, True, "float32")])
def test_ordered_norm_stats_preserves_each_row(tmp_path, rows, split_rows, residual_dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    k, n = 128, 256
    placement = fm.Placement((2, 2), "yx", "bb")
    b, m = fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), (rows + 1) // 2) if split_rows else fm.SBP.broadcast()
    ns = fm.SBP.split_contiguous((1,), n // 16)
    x_type = fm.tensor_type("bfloat16", (rows, k))
    w_type = fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (k // 16, n // 8))
    r_type = fm.tensor_type(fm.VectorType(fm.DType(residual_dtype), (8,)), (rows, n // 8))

    class Graph(fm.Module):
        def forward(self):
            x, w, r = (self.input(name, t, id=name) for name, t in (("x", x_type), ("w", w_type), ("r", r_type)))
            local_x = fm.F.distributed.force_boxing(x, fm.DistributedType(x_type, (m, b), placement))
            local_w = fm.F.distributed.force_boxing(w, fm.DistributedType(w_type, (b, ns), placement))
            local_r = fm.F.distributed.force_boxing(r, fm.DistributedType(r_type, (m, ns), placement))
            result = fm.F.ntt.matmul_norm_stats(local_x, local_w, local_r, rhs_layout="k_major", axis=1,
                                                use_mean=False, output_data_type="bfloat16",
                                                reduction_order="mma_k16", name="projection",
                                                metadata={"selected_vectorization": "vectorization.matmul.n",
                                                          "selected_vector_axes": (1,),
                                                          "selected_vector_lanes": (8,)})
            value, stats = fm.F.tensors.get_item(result, 0), fm.F.tensors.get_item(result, 1)
            value = fm.F.distributed.force_boxing(value, value.type.tensor)
            stats = fm.F.distributed.force_boxing(stats, stats.type.tensor)
            self.function("main", (x, w, r), (value, stats))

    source = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    compiled = Compiler().compile(source).module
    selection = compiled.selection_map["tir.projection"]
    assert selection.candidate_id == "tir.dense_matmul.packed_k_major_mma_norm_stats"
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(917)
    x = (torch.randint(-4, 5, (rows, k), generator=generator).float() / 32).bfloat16().cuda()
    w = (torch.randint(-4, 5, (k, n), generator=generator).float() / 32).bfloat16().cuda()
    packed = w.reshape(k // 16, 2, 8, n // 8, 8).permute(0, 3, 4, 1, 2).contiguous()
    residual = (torch.randint(-4, 5, (rows, n), generator=generator).float() / 32).to(getattr(torch, residual_dtype)).cuda()
    output = torch.full((rows, n // 8, 8), float("nan"), dtype=getattr(torch, residual_dtype), device="cuda")
    stats = torch.full((1, rows, 1), float("nan"), device="cuda")
    expected = (x @ w).float() + residual.float()
    expected = expected.to(getattr(torch, residual_dtype))
    try:
        arguments = (x, packed, residual.reshape(rows, n // 8, 8), output, stats)
        runtime.prepare(*arguments)
        runtime.run_into(*arguments)
        torch.cuda.synchronize()
        torch.testing.assert_close(output.reshape(rows, n), expected, rtol=0, atol=0)
        torch.testing.assert_close(stats, expected.float().square().sum(-1, keepdim=True)[None], rtol=1e-6, atol=1e-6)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
