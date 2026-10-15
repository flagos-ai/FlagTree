# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One AddNormStats artifact across independent per-process workspace replicas."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _worker(artifact, elements):
    import torch
    from triton.flagmega.runtime import load

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    runtime = load(artifact, device=device)
    try:
        source = ((torch.arange(elements, device=device) % 17 - 8) / 4).reshape(1, -1).bfloat16()
        residual = ((torch.arange(elements, device=device) % 13 - 6) / 4).reshape(1, -1).bfloat16()
        expected = source + residual
        expected_stats = expected.float().square().sum(-1, keepdim=True).unsqueeze(0)
        expected_token = expected.argmax(-1).int()
        expected = torch.nn.functional.silu(expected)
        value, stats = torch.empty_like(expected), torch.empty_like(expected_stats)
        token = torch.empty_like(expected_token)
        runtime.prepare(source, residual, value, stats, token)
        for _ in range(3):
            value.fill_(float("nan"))
            stats.fill_(float("nan"))
            token.fill_(-9)
            runtime.run_into(source, residual, value, stats, token)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
            torch.testing.assert_close(token, expected_token, rtol=0, atol=0)
    finally:
        runtime.close()


@pytest.mark.parametrize("variant", ["rms", "persistent_rms"])
@pytest.mark.parametrize("elements", [16, 2176])
@pytest.mark.parametrize("hierarchy,levels", [((4, 2, 4), "cbb"), ((2, 4, 4), "bcb")])
def test_replicated_add_norm_stats_four_ranks(tmp_path, variant, elements, hierarchy, levels):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 4 or any(
        torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(4)
    ):
        pytest.skip("Four SM90 CUDA devices are required")
    from triton.flagmega import ir as fm
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler

    placement = fm.Placement(hierarchy, "abc", levels)
    block_axes = tuple(axis for axis, level in enumerate(levels) if level == "b")
    tensor = fm.tensor_type("bfloat16", (1, elements))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, placement)

    class Graph(fm.Module):
        def forward(self):
            source = self.input("source", tensor)
            residual = self.input("residual", tensor)
            value = fm.F.distributed.sharded_view(source, broadcast)
            addend = fm.F.distributed.sharded_view(residual, broadcast)
            combined = fm.F.ntt.add_norm_stats(value, addend, axis=-1, use_mean=False, name="combine")
            value = fm.F.tensors.get_item(combined, 0)
            stats = fm.F.tensors.get_item(combined, 1)
            split = fm.DistributedType(tensor, (
                fm.SBP.broadcast(), fm.SBP.split_contiguous(block_axes, elements // 8),
            ), placement)
            local = fm.F.distributed.sharded_view(value, split)
            token = fm.F.nn.greedy_sample(local)
            token = fm.F.distributed.force_boxing(token, fm.tensor_type("int32", (1,)))
            value = fm.F.math.silu(local)
            value = fm.F.distributed.force_boxing(value, tensor)
            stats = fm.F.distributed.force_boxing(stats, stats.type.tensor)
            self.function("main", (source, residual), (value, stats, token))

    compiler = Compiler()
    model = compiler.target.triton_implementation_model
    compiler.target.triton_implementation_model = replace(model, preferences={
        **model.preferences, "add_norm_stats": (f"tir.add_norm_stats.{variant}",),
    })
    module = compiler.compile(Graph(
        dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}},
    ).build()).module
    assert fm.kernel_dispatch_for_call(module, module.node_map["combine"]).microkernel.variant == variant
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
        str(Path(__file__).resolve()), str(artifact), str(elements),
    ], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _worker(sys.argv[1], int(sys.argv[2]))
