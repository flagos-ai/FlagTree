# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""RoPE casts tables and rounds both products before adding them."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_rope_preserves_individual_tensor_operations(tmp_path, dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type(dtype, (2, 2, 64)))
            cosine = self.input("cosine", fm.tensor_type("float32", (2, 1, 64)))
            sine = self.input("sine", cosine.type)
            result = fm.F.nn.rope(value, cosine, sine, name="rotated")
            self.function("main", (value, cosine, sine), (result,))

    source = Graph(dialect="high_level", stage="imported", entry="main").build()
    compiled = Compiler().compile(source).module
    artifact = write_artifact(compiled, tmp_path / "rope", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(75)
    value = torch.randn((2, 2, 64), generator=generator).to(device="cuda", dtype=getattr(torch, dtype))
    cosine = torch.randn((2, 1, 64), generator=generator).cuda()
    sine = torch.randn((2, 1, 64), generator=generator).cuda()
    rotated = torch.cat((-value[..., 32:], value[..., :32]), dim=-1)
    expected = value * cosine.to(value.dtype) + rotated * sine.to(value.dtype)
    output = torch.empty_like(value)
    runtime.prepare(value, cosine, sine, output=output)
    runtime.run_into(output, value, cosine, sine)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
