# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FP8 GLU must address compact weight rows relative to the selected owner."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.selection import apply_plan, override_plan


def _module(*, canonical_up=False, gate_limit=None, up_limit=None):
    mesh = fm.Placement((2, 4), "yx", "bb")
    b = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous((0, 1), 24)
    value = fm.tensor_type("bfloat16", (1, 128))
    weight = fm.tensor_type("float8_e4m3fn", (192, 128))
    scale = fm.tensor_type("float32", (12, 1))

    class Graph(fm.Module):
        def forward(self):
            x = self.input("value", value, id="value")
            gate = self.input("gate", weight, id="gate")
            up = self.input("up", weight, id="up")
            scales = self.input("scales", scale, id="scales")
            px = fm.F.distributed.force_boxing(x, fm.DistributedType(value, (b, b), mesh))
            pg = fm.F.distributed.force_boxing(gate, fm.DistributedType(weight, (split, b), mesh))
            up_type = fm.DistributedType(weight, (split, b), mesh)
            pu = (fm.F.distributed.sharded_view(up, up_type) if canonical_up
                  else fm.F.distributed.force_boxing(up, up_type))
            ps = fm.F.distributed.force_boxing(scales, fm.DistributedType(scale, (b, b), mesh))
            result = fm.F.nn.matmul_glu(px, pg, pu, ps, ps, activation="silu",
                weight_block_n=16, weight_block_k=128,
                gate_clamp_max=gate_limit, up_clamp=up_limit, name="glu")
            output = fm.F.distributed.force_boxing(result, fm.logical_type(result.type))
            self.function("main", (x, gate, up, scales), (output,))

    return Graph(dialect="distributed", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()


@pytest.mark.parametrize("variant", ("tir.matmul_glu.simt", "tir.matmul_glu.mma"))
@pytest.mark.parametrize("canonical_up", (False, True))
@pytest.mark.parametrize("limits", ((None, None), (.75, .5)))
def test_output_shards_use_compact_fp8_weight_rows(tmp_path, variant, canonical_up, limits):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    compiler = Compiler()
    proposed = compiler.compile(_module(canonical_up=canonical_up, gate_limit=limits[0], up_limit=limits[1]),
                                stop_after="propose-tir").module
    choices = [(point.id, variant) for point in proposed.selection_points
               if any(candidate.id == variant for candidate in point.candidates)]
    assert len(choices) == 1
    selected = apply_plan(proposed, override_plan(proposed, choices))
    artifact = write_artifact(compiler.compile(selected).module, tmp_path / "glu",
                              target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    try:
        x = torch.ones((1, 128), device="cuda", dtype=torch.bfloat16)
        gate = ((torch.arange(192, device="cuda") % 3 + 1) / 256).reshape(192, 1).expand(192, 128)
        up_rows = ((torch.arange(192, device="cuda") % 2) * 2 - 1).reshape(192, 1) / 128
        up = up_rows.expand(192, 128).to(torch.float8_e4m3fn)
        gate = gate.to(torch.float8_e4m3fn)
        scales = torch.ones((12, 1), device="cuda")
        expected_gate = gate.float().reshape(192, 128).sum(1).bfloat16()
        expected_up = up.float().sum(1).bfloat16()
        if limits[0] is not None:
            expected_gate = expected_gate.clamp(max=limits[0])
        if limits[1] is not None:
            expected_up = expected_up.clamp(min=-limits[1], max=limits[1])
        expected = (torch.nn.functional.silu(expected_gate.float()).bfloat16() * expected_up).reshape(1, 192)
        output = torch.empty_like(expected)
        runtime.prepare(x, gate, up, scales, output=output)
        runtime.run_into(output, x, gate, up, scales)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
