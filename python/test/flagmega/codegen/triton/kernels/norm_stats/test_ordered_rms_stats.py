# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A selected RMS sum preserves tile FMA and lane-local addition order."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.devices import is_nvidia_sm90


def _reference(value):
    # The pinned TileLang CUDA contract owns 16 positions per logical lane.
    # Widening the fixed finite fixture computes each tile FMA independently
    # of the device implementation, with one FP32 rounding per accumulation.
    rows, width = value.shape
    capacity = ((width + 1023) // 1024) * 1024
    padded = torch.zeros((rows, capacity))
    padded[:, :width] = value
    tiles = padded.reshape(rows, -1, 2, 64, 8).permute(1, 0, 3, 2, 4).reshape(-1, rows, 64, 16)
    squares = torch.zeros((rows, 64, 16))
    for tile in tiles:
        squares = (tile.double().square() + squares.double()).float()
    subtotal = torch.zeros((rows, 64))
    for element in range(8):
        subtotal = subtotal + squares[:, :, element]
        subtotal = subtotal + squares[:, :, element + 8]
    # AllReduce<SumOp,64> performs XOR32 first, then XOR16 down to XOR1.
    indices = torch.arange(64)
    for offset in (32, 16, 8, 4, 2, 1):
        subtotal = subtotal + subtotal[:, indices ^ offset]
    return subtotal[:, 0].reshape(1, rows, 1)


def _module(shape=(1, 4096), *, use_mean=False):
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type("float32", shape), id="value")
            result = fm.F.nn.norm_stats(value, axis=-1, use_mean=use_mean)
            self.function("main", (value,), (result,))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _compile(module, algorithm="fma_1024", warps=8):
    base = NvidiaSm90Target().triton_implementation_model
    catalog = replace(
        base,
        implementations=tuple(
            replace(item, parameters={**item.parameters, "compute_num_warps": warps, "reduction_algorithm": algorithm})
            if item.family == "norm_stats"
            else item
            for item in base.implementations
        ),
    )
    compiler = Compiler()
    compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    return compiler.compile(module).module


@pytest.mark.parametrize("algorithm", ["unknown", True, 1, []])
def test_invalid_rms_reduction_algorithm_is_rejected(tmp_path, algorithm):
    with pytest.raises((CodegenError, IRVerificationError), match="reduction_algorithm"):
        module = _compile(_module(), algorithm)
        write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


def test_ordered_rms_rejects_mean_statistics(tmp_path):
    with pytest.raises(CodegenError, match="RMS statistics only"):
        module = _compile(_module(use_mean=True))
        write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


def test_ordered_rms_rejects_single_warp(tmp_path):
    # Normal SM90 launch planning currently starts at four warps. A resumed
    # executable contract can still request one and must be rejected here.
    module = _compile(_module(), warps=1)
    package = module.metadata["codegen_package_plan"]
    module = replace(
        module,
        metadata={
            **module.metadata,
            "launch_contract": {**module.metadata["launch_contract"], "num_warps": 1},
            "codegen_package_plan": {**package, "launch": {**package["launch"], "num_warps": 1}},
        },
    )
    with pytest.raises(CodegenError, match="at least two compute warps"):
        write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


def test_resumed_ordered_rms_requires_sm90(tmp_path, monkeypatch):
    import triton.flagmega.codegen.triton.tir_package as package

    monkeypatch.setattr(package, "module_template_target", lambda module: ("iluvatar", "bi-v150"))
    with pytest.raises(CodegenError, match="requires the NVIDIA SM90 target"):
        module = _compile(_module())
        write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


@pytest.mark.parametrize("shape", [(1, 4096), (3, 1025), (2, 128)], ids=["single_full", "multi_tail", "short"])
def test_ordered_fma_rms_sum_matches_independent_reference(tmp_path, shape):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    generator = torch.Generator().manual_seed(0)
    host = torch.randn(shape, generator=generator)
    value = host.cuda()
    expected = _reference(host).cuda()
    module = _compile(_module(shape))
    restored = fm.load_module(fm.emit_module(module, tmp_path / "selected.py"))
    assert restored.semantic_hash == module.semantic_hash
    artifact = write_artifact(restored, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    try:
        output = torch.empty_like(expected)
        runtime.prepare(value, output=output)
        runtime.run_into(output, value)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
