# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""An exclusive norm must gather an internally produced chip-split row."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def exclusive_norm_module(lanes):
    from triton.flagmega import ir as fm
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.passes.auto_distributed import (
        DistributedReshardRealization as Realization,
    )
    from triton.flagmega.passes.auto_distributed import (
        DistributedReshardRealizationContext as Context,
    )
    from triton.flagmega.passes.auto_distributed import (
        DistributedReshardSourceKind as Source,
    )
    from triton.flagmega.passes.auto_distributed import (
        DistributedReshardUsageKind as Usage,
    )
    from triton.flagmega.passes.auto_distributed import (
        PyNttDistributedReshardRealizationPolicy,
    )

    mesh = fm.Placement((2, 2, 2), "cyx", "cbb")
    x_type = fm.tensor_type("bfloat16", (1, 128))
    w_type = (fm.tensor_type("bfloat16", (128, 1536)) if lanes == 1
              else fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (8, 192)))
    scale_type = fm.tensor_type("float32" if lanes == 1 else fm.vector_type("float32", (8,)),
                                (1536 // lanes,))
    split = fm.SBPSplit(stages=(*fm.SBP.split_contiguous((0,)).stages,
                               *fm.SBP.split_block_cyclic((1, 2), 1).stages))

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", x_type, id="x")
            w = self.input("w", w_type, id="w")
            scale = self.input("scale", scale_type, id="scale")
            bias = self.input("bias", scale_type, id="bias")
            a = fm.F.distributed.force_boxing(x, fm.DistributedType(
                x_type, (fm.SBP.broadcast(),) * 2, mesh), name="load_x")
            b = fm.F.distributed.force_boxing(w, fm.DistributedType(
                w_type, (fm.SBP.broadcast(), split), mesh), name="load_w")
            if lanes == 1:
                projected = fm.F.math.matmul(a, b, name="project")
            else:
                none = fm.F.builtin.none()
                projected = fm.F.ntt.packed_matmul(a, b, none, none,
                    output_data_type="bfloat16", name="project",
                    metadata={"selected_vectorization": "vectorization.matmul.n",
                              "selected_vector_axes": (1,), "selected_vector_lanes": (8,)})
            target = fm.DistributedType(fm.logical_type(projected.type),
                (fm.SBP.broadcast(),) * 2, mesh, exclusive=fm.SBP.exclusive((1, 2)))
            realization = PyNttDistributedReshardRealizationPolicy().classify(
                Context(projected.type, target, Source.INTERNAL, Usage.INTERNAL))
            adapter = (fm.F.distributed.sharded_view if realization is Realization.SHARDED_VIEW
                       else fm.F.distributed.force_boxing)
            complete = adapter(projected, target, name="exclusive_input")
            parameter_type = fm.DistributedType(scale_type, (fm.SBP.broadcast(),), mesh,
                                               exclusive=fm.SBP.exclusive((1, 2)))
            gamma = fm.F.distributed.force_boxing(scale, parameter_type, name="load_scale")
            beta = fm.F.distributed.force_boxing(bias, parameter_type, name="load_bias")
            stats = fm.F.nn.norm_stats(complete, axis=1, use_mean=False, name="stats")
            normalized = fm.F.nn.norm_apply(complete, stats, gamma, beta, axis=1,
                epsilon=1e-5, use_mean=False, name="normalize")
            scalar = normalized if lanes == 1 else fm.F.tensors.unpack(normalized, axes=(1,))
            result = fm.F.distributed.force_boxing(scalar, fm.logical_type(scalar.type), name="output")
            self.function("main", (x, w, scale, bias), (result,))

    return Compiler().compile(Graph(dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()).module


def _worker(artifact, lanes):
    import torch
    from triton.flagmega.runtime import load

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    runtime = load(artifact, device=device)
    x = torch.ones((1, 128), device=device, dtype=torch.bfloat16)
    row = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.bfloat16).repeat_interleave(384)
    w = (row / 128).expand(128, -1).contiguous()
    if lanes > 1:
        w = w.reshape(8, 2, 8, 192, 8).permute(0, 3, 4, 1, 2).contiguous()
    scale = torch.ones((1536,) if lanes == 1 else (192, 8), device=device, dtype=torch.float32)
    bias = torch.zeros_like(scale)
    output = torch.empty((1, 1536), device=device, dtype=torch.bfloat16)
    try:
        runtime.prepare(x, w, scale, bias, output=output)
        for step in range(2):
            # Internal producers must overwrite their valid regions. Missing
            # peer shards must never be read through an alias of this poison.
            runtime._pool_values["workspace"].view(torch.float32).fill_(float("nan"))
            source = x * (step + 1)
            value = row.float().reshape(1, -1) * (step + 1)
            expected = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)).bfloat16()
            runtime.run_into(output, source, w, scale, bias)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("lanes", (1, 8))
def test_exclusive_norm_gathers_peer_shards_before_reading_poisoned_workspace(tmp_path, lanes):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2 or any(torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(2)):
        pytest.skip("Two SM90 CUDA devices are required")
    from triton.flagmega.artifacts import write_artifact

    artifact = write_artifact(exclusive_norm_module(lanes), tmp_path / "norm", target="nvidia-sm90",
                              emit_executable=True)
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(Path(__file__).resolve()), str(artifact), str(lanes),
    ], capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]


if __name__ == "__main__":
    _worker(sys.argv[1], int(sys.argv[2]))
