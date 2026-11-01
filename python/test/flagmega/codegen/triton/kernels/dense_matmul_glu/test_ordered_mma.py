# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The fused GLU retains both ordered BF16 projection stores."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("rows,packed", [(1, False), (32, False), (32, True)])
def test_ordered_mma_glu_preserves_rows(tmp_path, rows, packed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    k, n = 128, 128
    mesh = fm.Placement((2, 2), "yx", "bb")
    b, ns = fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), n // (16 if packed else 2))
    weight = fm.tensor_type("bfloat16", (k // 16, n // 8, 2, 64) if packed else (n, k))
    policies = (b, ns, b, b) if packed else (ns, b)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (rows, k)), id="x")
            g, u = (self.input(name, weight, id=name) for name in ("g", "u"))
            local_x = fm.F.distributed.force_boxing(x, fm.DistributedType(x.type, (b, b), mesh))
            local_g, local_u = (fm.F.distributed.force_boxing(value, fm.DistributedType(weight, policies, mesh)) for value in (g, u))
            operation = fm.F.nn.packed_dense_matmul_glu if packed else fm.F.nn.dense_matmul_glu
            result = operation(local_x, local_g, local_u, round_activation=False, reduction_order="mma_k16", name="glu")
            result = fm.F.distributed.force_boxing(result, result.type.tensor)
            self.function("main", (x, g, u), (result,))

    source = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()
    compiled = Compiler().compile(source).module
    assert compiled.selection_map["tir.glu"].candidate_id == "tir.dense_matmul_glu.ordered_mma_" + ("packed" if packed else "logical")
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    rng = torch.Generator().manual_seed(917)
    x = (torch.randint(-4, 5, (rows, k), generator=rng).float() / 32).bfloat16().cuda()
    g, u = ((torch.randint(-4, 5, (n, k), generator=rng).float() / 32).bfloat16().cuda() for _ in range(2))
    expected = (torch.nn.functional.silu((x @ g.T).float()) * (x @ u.T).float()).bfloat16()
    output = torch.full_like(expected, float("nan"))
    def storage(w):
        return w.reshape(n // 8, 8, k // 16, 16).permute(2, 0, 1, 3).contiguous().reshape(k // 16, n // 8, 2, 64) if packed else w
    args = (x, storage(g), storage(u))
    try:
        runtime.prepare(*args, output=output)
        runtime.run_into(output, *args)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
