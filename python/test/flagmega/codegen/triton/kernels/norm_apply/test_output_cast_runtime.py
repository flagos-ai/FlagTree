# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execute local and owner-reduced NormApply final dtype conversions."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("source_dtype,output_dtype", [("float32", "bfloat16"), ("bfloat16", "float32")])
def test_output_cast_preserves_input_rounding_and_uses_distinct_storage(tmp_path, collective, source_dtype, output_dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    mesh = fm.Placement((8, 16), "yx", "bb")
    x_type = fm.tensor_type(source_dtype, (1, 1024))
    p_type = fm.tensor_type("bfloat16", (1024,))
    out_type = fm.tensor_type(output_dtype, (1, 1024))
    b, split = fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 2)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", x_type)
            scale = self.input("scale", p_type)
            bias = self.input("bias", p_type)
            px = fm.F.distributed.force_boxing(fm.F.tensors.pack(x, lanes=(4,), axes=(1,)),
                fm.DistributedType(fm.tensor_type(fm.vector_type(source_dtype, 4), (1, 256)), (b, split), mesh))
            parameters = tuple(fm.F.distributed.force_boxing(fm.F.tensors.pack(value, lanes=(4,), axes=(0,)),
                fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", 4), (256,)), (split,), mesh))
                for value in (scale, bias))
            if collective:
                stats = fm.F.nn.norm_stats(px, axis=1, use_mean=False)
                materialized_type = replace(stats.type, partial=None)
                output = fm.F.ntt.gather_reduce_norm_apply(stats, px, *parameters,
                    materialized_stats_type=materialized_type, axis=1, epsilon=1e-8, use_mean=False,
                    output_dtype=output_dtype, name="apply")
            else:
                external_stats = self.input("stats", fm.tensor_type("float32", (1, 1, 1)))
                materialized_type = fm.DistributedType(external_stats.type, (b, b, b), mesh)
                output = fm.F.nn.norm_apply(px, fm.F.distributed.force_boxing(external_stats, materialized_type),
                    *parameters, axis=1, epsilon=1e-8, use_mean=False, output_dtype=output_dtype, name="apply")
            logical = fm.F.tensors.unpack(output, axes=(1,))
            result = fm.F.distributed.force_boxing(logical, out_type)
            self.function("main", (x, scale, bias) if collective else (x, scale, bias, external_stats), (result,))

    module = Compiler().compile(Graph(dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()).module
    artifact = write_artifact(module, tmp_path / "norm", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    x = torch.ones((1, 1024), device="cuda", dtype=getattr(torch, source_dtype))
    scale = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    bias = torch.full_like(scale, 1 / 256)
    # Exact BF16 halfway case: F32 result must still contain the BF16
    # normalization output rounding, not the unrounded FP32 sum.
    expected = (x.float() * scale.float() + bias.float()).to(x.dtype).to(getattr(torch, output_dtype))
    output = torch.empty_like(expected)
    inputs = (x, scale, bias) if collective else (x, scale, bias, torch.full((1, 1, 1), 1024.0, device="cuda"))
    runtime.prepare(*inputs, output=output)
    runtime.run_into(output, *inputs)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
