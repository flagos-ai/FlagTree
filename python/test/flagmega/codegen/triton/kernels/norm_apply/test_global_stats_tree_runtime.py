# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Cross-chip statistics retain one reduction over globally ordered lanes."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.errors import CodegenError, IRVerificationError

from python.test.flagmega.codegen.triton.kernels.norm_apply.test_cross_chip_stats_runtime import cross_chip_norm_module
from python.test.flagmega.devices import has_nvidia_sm90_devices


@triton.jit
def _reference(Input, Output, USE_MEAN: tl.constexpr):
    offsets = tl.arange(0, 512)
    layout: tl.constexpr = tle.gpu.BlockEncoding([8], [32], [4], [0])
    value = tle.gpu.set_layout(tl.load(Input + offsets), layout)
    mean = tl.sum(value, 0) / 512 if USE_MEAN else 0.0
    variance = tl.sum(value * value, 0) / 512 - mean * mean
    inverse = tl.rsqrt(tl.maximum(variance, 0.0) + 1e-5)
    tl.store(Output + offsets, (value - mean) * inverse)


def _selected_module(use_mean):
    module = cross_chip_norm_module(lanes=8, use_mean=use_mean, dtype="float32")
    definitions = []
    for definition in module.kernel_definitions:
        dispatch = definition.dispatch
        parameters = dict(dispatch.microkernel.parameters)
        if dispatch.microkernel.family == "norm_stats":
            parameters.update(block_size=2048, reduction_layout="contiguous_8")
        elif dispatch.microkernel.family == "gather_reduce_norm_apply":
            parameters.update(partial_reduction="global_tree")
        definitions.append(
            replace(
                definition, dispatch=replace(dispatch, microkernel=replace(dispatch.microkernel, parameters=parameters))
            )
        )
    return fm.verify_module(replace(module, kernel_definitions=tuple(definitions)))


def _worker(artifact, use_mean):
    from triton.flagmega.runtime import load

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    source = torch.randn((1, 512), generator=torch.Generator().manual_seed(0)).to(device)
    scale = torch.ones(512, device=device)
    bias = torch.zeros_like(scale)
    output = torch.empty_like(source)
    expected = torch.empty_like(source)
    runtime = load(artifact, device=device)
    try:
        runtime.prepare(source, scale, bias, output=output)
        for step in range(3):
            value = source * (step + 1)
            _reference[(1,)](value, expected, use_mean, num_warps=4)
            runtime.run_into(output, value, scale, bias)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["unknown", 1, True, []])
def test_invalid_global_statistics_mode_is_rejected(tmp_path, mode):
    module = _selected_module(False)
    with pytest.raises((CodegenError, IRVerificationError), match="partial_reduction"):
        definitions = tuple(
            replace(
                item,
                dispatch=replace(
                    item.dispatch,
                    microkernel=replace(
                        item.dispatch.microkernel,
                        parameters={**item.dispatch.microkernel.parameters, "partial_reduction": mode},
                    ),
                ),
            )
            if item.dispatch.microkernel.family == "gather_reduce_norm_apply"
            else item
            for item in module.kernel_definitions
        )
        write_artifact(
            replace(module, kernel_definitions=definitions),
            tmp_path / "invalid",
            target="nvidia-sm90",
            emit_executable=True,
        )


def test_resumed_global_statistics_tree_requires_sm90(tmp_path, monkeypatch):
    import triton.flagmega.codegen.triton.tir_package as package

    monkeypatch.setattr(package, "module_template_target", lambda module: ("iluvatar", "bi-v150"))
    with pytest.raises(CodegenError, match="requires the NVIDIA SM90 target"):
        write_artifact(_selected_module(False), tmp_path / "invalid", target="nvidia-sm90", emit_executable=True)


@pytest.mark.parametrize("use_mean", [False, True])
def test_global_statistics_tree_matches_complete_reduction(tmp_path, use_mean):
    if not has_nvidia_sm90_devices(4):
        pytest.skip("requires four SM90 devices")
    artifact = write_artifact(_selected_module(use_mean), tmp_path / "norm", target="nvidia-sm90", emit_executable=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=4",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=127.0.0.1:0",
            "--local_addr=127.0.0.1",
            str(Path(__file__).resolve()),
            str(artifact),
            str(int(use_mean)),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]


if __name__ == "__main__":
    _worker(sys.argv[1], bool(int(sys.argv[2])))
