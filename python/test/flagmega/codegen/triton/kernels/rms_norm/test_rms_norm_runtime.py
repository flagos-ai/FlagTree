# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


class _ConstantOnlyCheckpoint:
    """Resolver used only to materialize compiler-owned splat recipes."""

    def load_tensor(self, key, *, device="cpu"):
        raise AssertionError(
            f"RMSNorm fixture unexpectedly requested checkpoint tensor {key!r}"
        )


def _module():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 3, 5)))
            weight = self.input("weight", fm.tensor_type("bfloat16", (5,)))
            output = fm.F.nn.rms_norm(
                value, weight, epsilon=1e-6, weight_bias=0.25, name="output"
            )
            self.function("main", (value, weight), (output,))

    return Graph().build()


def test_generic_local_rms_norm_source_has_no_mesh_axis_reduction(tmp_path):
    module = Compiler().compile(_module()).module
    artifact = write_artifact(
        module,
        tmp_path / "local-rms",
        target="nvidia-sm90",
        checkpoint=_ConstantOnlyCheckpoint(),
        emit_executable=True,
    )
    source = (artifact / "generated_kernels.py").read_text("utf-8")

    assert "# flagmega-kernel: rms_norm/local platform=generic" in source
    assert "for rms_outer_index in tl.range" in source
    assert "rms_reduction_offsets" in source
    assert "partial_stats" not in source
    assert "stats_mesh_axis" not in source


def test_generic_local_rms_norm_matches_torch_on_h800(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = Compiler().compile(_module()).module
    artifact = write_artifact(
        module,
        tmp_path / "local-rms-runtime",
        target="nvidia-sm90",
        checkpoint=_ConstantOnlyCheckpoint(),
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = torch.randn((2, 3, 5), dtype=torch.bfloat16, device="cuda:0")
    weight = torch.randn((5,), dtype=torch.bfloat16, device="cuda:0")
    output = torch.empty_like(value)

    runtime.prepare(value, weight, output=output)
    runtime.run_into(output, value, weight)
    torch.cuda.synchronize()

    variance = value.float().square().mean(dim=-1, keepdim=True)
    expected = (
        value.float() * (variance + 1e-6).rsqrt()
        * (weight.float() + 0.25)
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
