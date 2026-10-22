# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Real partial producer plus one collective; no model/importer dependency."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load


_PARTITIONS = [
    (3, (0, 1), False), (129, (0, 1), True),
    (25, (0,), True), (257, (1,), False),
]
_AMD_TARGETS = ("amd-gfx1100", "amd-gfx1201")


def _build_collective(values, partial_axes, use_mean, output_dtype):
    placement = fm.Placement((8, 16), "yx", "bb")
    preserved = tuple(axis for axis in range(2) if axis not in partial_axes)
    owners = 1
    for axis in partial_axes:
        owners *= placement.hierarchy[axis]
    source_type = fm.tensor_type("float32", (values, owners))
    value_type = fm.tensor_type("float32", (1, values, 1))
    normalized_type = fm.tensor_type(output_dtype, (1, values, 1))
    parameter_type = fm.tensor_type("float32", (values, 1))
    b = fm.SBP.broadcast()
    source_distributed = fm.DistributedType(
        source_type,
        (fm.SBP.split_block_cyclic(preserved, 1) if preserved else b,
         fm.SBP.split_contiguous(partial_axes, 1)), placement,
    )
    value_distributed = fm.DistributedType(value_type, (b, b, b), placement)
    parameter_distributed = fm.DistributedType(parameter_type, (b, b), placement)

    class Graph(fm.Module):
        def forward(self):
            source = self.input("source", source_type)
            residual = self.input("residual", value_type)
            scale = self.input("scale", parameter_type)
            bias = self.input("bias", parameter_type)
            local = fm.F.distributed.force_boxing(source, source_distributed)
            partial = fm.F.nn.norm_stats(local, axis=1, use_mean=False)
            output = fm.F.ntt.gather_reduce_add_norm_apply(
                partial,
                fm.F.distributed.force_boxing(residual, value_distributed),
                fm.F.distributed.force_boxing(scale, parameter_distributed),
                fm.F.distributed.force_boxing(bias, parameter_distributed),
                axis=1, epsilon=1e-5, use_mean=use_mean, output_dtype=output_dtype, name="collective",
            )
            value, normalized = fm.F.tensors.get_items(output, 0, 1)
            self.function("main", (source, residual, scale, bias), (
                fm.F.distributed.force_boxing(value, value_type),
                fm.F.distributed.force_boxing(normalized, normalized_type),
            ))

    built = Graph(
        dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}},
    ).build()
    return built, owners


@pytest.mark.parametrize("values,partial_axes,use_mean", _PARTITIONS)
@pytest.mark.parametrize("output_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("target_name", _AMD_TARGETS)
def test_collective_partition_lowers_on_amd(
    values, partial_axes, use_mean, output_dtype, target_name,
):
    # The SM90 runtime test below skips on AMD, so this compile-level check is the
    # only guard on RDNA that these distributed shapes -- suffix normalization
    # over (1, values, 1) with axis=1, and the partial/empty-owner layouts -- lower
    # end to end through the batched fused residual-norm epilogue. Runs on any
    # backend (no GPU needed) rather than silently skipping.
    built, _ = _build_collective(values, partial_axes, use_mean, output_dtype)
    compiled = Compiler(CompileOptions(target=target_name)).compile(built).module
    assert describe_tir_package(compiled)


@pytest.mark.parametrize("values,partial_axes,use_mean", _PARTITIONS)
@pytest.mark.parametrize("output_dtype", ["float32", "bfloat16"])
def test_collective_partition_preserves_values_stats_and_empty_owners(
    tmp_path, values, partial_axes, use_mean, output_dtype,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    built, owners = _build_collective(values, partial_axes, use_mean, output_dtype)
    module = Compiler().compile(built).module
    artifact = write_artifact(module, tmp_path / "collective", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    source = (torch.arange(values * owners, device="cuda").reshape(values, owners) % 13 - 6).float() / 8
    residual = torch.linspace(-1, 1, values, device="cuda").reshape(1, values, 1)
    scale = torch.linspace(.5, 1.5, values, device="cuda").reshape(values, 1)
    bias = torch.linspace(-.1, .1, values, device="cuda").reshape(values, 1)
    expected_value = source.square().sum(-1, keepdim=True).unsqueeze(0) + residual
    mean = expected_value.mean() if use_mean else 0.0
    variance = expected_value.square().mean() - mean * mean
    expected_norm = (expected_value - mean) * torch.rsqrt(variance.clamp_min(0) + 1e-5) * scale + bias
    expected_norm = expected_norm.to(getattr(torch, output_dtype))
    outputs = (torch.empty_like(expected_value), torch.empty_like(expected_norm))
    runtime.prepare(source, residual, scale, bias, *outputs)
    for _ in range(3):
        runtime.run_into(source, residual, scale, bias, *outputs)
        torch.testing.assert_close(outputs[0], expected_value, rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], expected_norm, rtol=2e-5, atol=2e-5)
    assert runtime.resource_report["spill_bytes"] == 0
