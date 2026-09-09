# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Parallel tiles preserve scalar-element math, casts, empty tensors and tails."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


# Rank-zero values use the separate by-value scalar ABI, not tensor pointers.
@pytest.mark.parametrize("shape", ((0,), (1,), (3, 17), (257,), (4096,)))
@pytest.mark.parametrize("variant", ("add", "mul", "silu", "cast"))
@pytest.mark.parametrize("dtype", ("bfloat16", "float32"))
def test_unpacked_tensor_tiles_execute_exact_stores(tmp_path, shape, variant, dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    output_dtype = "float32" if dtype == "bfloat16" else "bfloat16"

    class Graph(fm.Module):
        def forward(self):
            value_type = fm.tensor_type(dtype, shape)
            lhs = self.input("lhs", value_type, id="lhs")
            inputs = (lhs,)
            if variant in ("add", "mul"):
                rhs = self.input("rhs", value_type, id="rhs")
                inputs += (rhs,)
                output = getattr(fm.F.math, variant)(lhs, rhs, name="output")
            elif variant == "silu":
                output = fm.F.math.silu(lhs, name="output")
            else:
                output = fm.F.tensors.cast(lhs, fm.DType(output_dtype), name="output")
            self.function("main", inputs, (output,))

    # Resume after vectorization, preserving the scalar IR representation.
    module = Graph(dialect="nn", stage="frozen_constants", entry="main").build()
    final = Compiler().compile(module).module
    artifact = write_artifact(final, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator(device="cuda:0").manual_seed(3827)
    lhs = torch.randn(shape, dtype=getattr(torch, dtype), device="cuda:0", generator=generator)
    inputs = (lhs,)
    if variant in ("add", "mul"):
        rhs = torch.randn(shape, dtype=lhs.dtype, device=lhs.device, generator=generator)
        inputs += (rhs,)
        expected = lhs + rhs if variant == "add" else lhs * rhs
    elif variant == "silu":
        expected = (lhs.float() * torch.sigmoid(lhs.float())).to(lhs.dtype)
    else:
        expected = lhs.to(getattr(torch, output_dtype))
    runtime.prepare(*inputs)
    actual = runtime.run(*inputs)
    torch.cuda.synchronize()
    tolerance = 1e-6 if variant == "silu" and dtype == "float32" else 0
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
