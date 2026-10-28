# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""RMS statistics split across chips must retain their peer reduction."""

import os
import subprocess
import sys
from math import prod
from pathlib import Path

import pytest
from python.test.flagmega.devices import has_nvidia_sm90_devices


def cross_chip_norm_module(*, lanes=1, use_mean=False, split_axes=(0, 2), gather_output=True, dtype="bfloat16"):
    from triton.flagmega import ir as fm
    from triton.flagmega.compiler import Compiler

    mesh = fm.Placement((4, 8, 16), "cyx", "cbb")
    value = fm.tensor_type(dtype, (1, 512))
    parameter = fm.tensor_type("float32", (512,))
    granularity = 512 // prod(mesh.hierarchy[axis] for axis in split_axes) // lanes
    split = fm.SBP.split_contiguous(split_axes, granularity)
    packed_value = value if lanes == 1 else fm.tensor_type(fm.vector_type(dtype, lanes), (1, 512 // lanes))
    packed_parameter = parameter if lanes == 1 else fm.tensor_type(fm.vector_type("float32", lanes), (512 // lanes,))
    value_type = fm.DistributedType(packed_value, (fm.SBP.broadcast(), split), mesh)
    parameter_type = fm.DistributedType(packed_parameter, (split,), mesh)

    class Graph(fm.Module):
        def forward(self):
            source = self.input("source", value, id="source")
            scale = self.input("scale", parameter, id="scale")
            bias = self.input("bias", parameter, id="bias")
            packed = (source, scale, bias) if lanes == 1 else (
                fm.F.tensors.pack(source, lanes=(lanes,), axes=(1,)),
                fm.F.tensors.pack(scale, lanes=(lanes,), axes=(0,)),
                fm.F.tensors.pack(bias, lanes=(lanes,), axes=(0,)),
            )
            x = fm.F.distributed.boxing(packed[0], value_type, name="load")
            w = fm.F.distributed.boxing(packed[1], parameter_type, name="load_scale")
            b = fm.F.distributed.boxing(packed[2], parameter_type, name="load_bias")
            partial = fm.F.nn.norm_stats(x, axis=1, use_mean=use_mean, name="stats")
            stats = fm.F.distributed.boxing(partial, fm.DistributedType(
                fm.logical_type(partial.type), (fm.SBP.broadcast(),) * 3, mesh), name="sum_stats")
            normalized = fm.F.nn.norm_apply(x, stats, w, b, axis=1, epsilon=1e-5,
                                             use_mean=use_mean, name="normalize")
            if gather_output:
                scalar = normalized if lanes == 1 else fm.F.tensors.unpack(normalized, axes=(1,))
                result = fm.F.distributed.boxing(scalar, value, name="gather")
            else:
                result = normalized
            self.function("main", (source, scale, bias), (result,))

    return Compiler().compile(Graph(dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()).module


def _worker(artifact, use_mean):
    import torch
    from triton.flagmega.runtime import load

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    runtime = load(artifact, device=device)
    source = torch.tensor([.5, 1, 2, 4], device=device, dtype=torch.bfloat16).repeat_interleave(128).reshape(1, 512)
    scale = torch.ones(512, device=device, dtype=torch.float32)
    bias = torch.zeros_like(scale)
    output = torch.empty_like(source)
    try:
        runtime.prepare(source, scale, bias, output=output)
        for step in range(4):
            scaled = source * (step + 1)
            value = scaled.float()
            mean = value.mean(-1, keepdim=True) if use_mean else 0.0
            variance = (value.square().mean(-1, keepdim=True) - mean * mean).clamp_min(0)
            expected = ((value - mean) * torch.rsqrt(variance + 1e-5)).bfloat16()
            output.fill_(float("nan"))
            runtime.run_into(output, scaled, scale, bias)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("lanes,use_mean,split_axes", [
    (1, False, (0, 2)), (8, False, (0, 2)), (8, True, (0, 2)), (1, True, (0,)),
])
def test_cross_chip_statistics_are_reduced_before_rms_apply(tmp_path, lanes, use_mean, split_axes):
    pytest.importorskip("torch")
    if not has_nvidia_sm90_devices(4):
        pytest.skip("Four SM90 CUDA devices are required")
    from triton.flagmega.artifacts import write_artifact

    artifact = write_artifact(cross_chip_norm_module(lanes=lanes, use_mean=use_mean, split_axes=split_axes),
                              tmp_path / "norm", target="nvidia-sm90", emit_executable=True)
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=4",
        "--rdzv_backend=c10d", "--rdzv_endpoint=127.0.0.1:0", "--local_addr=127.0.0.1",
        str(Path(__file__).resolve()), str(artifact), str(int(use_mean)),
    ], capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]


def test_fused_remote_statistics_register_window_without_a_boxing_leaf(tmp_path):
    from triton.flagmega.codegen.triton import render_triton_package

    package = render_triton_package(cross_chip_norm_module(gather_output=False), tmp_path)
    assert package["device_window_storage"] == "workspace"
    source = (tmp_path / "generated_kernels.py").read_text()
    assert "for norm_partial_peer in tl.static_range(4)" in source
    assert "norm_partial_byte_offset" in source


if __name__ == "__main__":
    _worker(sys.argv[1], bool(int(sys.argv[2])))
