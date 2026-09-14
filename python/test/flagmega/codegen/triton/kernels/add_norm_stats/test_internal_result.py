# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("private_residual", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("variant", ["persistent_rms", "rms"])
def test_stats_publishes_internal_results(tmp_path, private_residual, dtype, variant):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type(dtype, (1, 32))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, placement)
    split = fm.DistributedType(tensor, (
        fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 4),
    ), placement)

    class Graph(fm.Module):
        def forward(self):
            source = self.input("source", tensor)
            residual = self.input("residual", tensor)
            value = fm.F.distributed.sharded_view(source, broadcast)
            addend = fm.F.distributed.sharded_view(residual, broadcast)
            if private_residual:
                addend = fm.F.math.silu(addend, name="private_residual")
            combined = fm.F.ntt.add_norm_stats(
                value, addend, axis=-1, use_mean=False, name="combine",
            )
            value = fm.F.tensors.get_item(combined, 0)
            stats = fm.F.tensors.get_item(combined, 1)
            value = fm.F.math.silu(fm.F.distributed.sharded_view(value, split))
            value = fm.F.distributed.force_boxing(value, tensor)
            stats = fm.F.math.add(stats, stats)
            stats = fm.F.distributed.force_boxing(stats, stats.type.tensor)
            self.function("main", (source, residual), (value, stats))

    compiler = Compiler()
    model = compiler.target.triton_implementation_model
    identity = f"tir.add_norm_stats.{variant}"
    compiler.target.triton_implementation_model = replace(model, preferences={
        **model.preferences, "add_norm_stats": (identity,),
    })
    module = compiler.compile(Graph(
        dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}},
    ).build()).module
    dispatch = fm.kernel_dispatch_for_call(module, module.node_map["combine"])
    assert dispatch.microkernel.variant == variant
    plan = fm.verify_buffer_plan(module)
    for name in ("combine.0", "combine.1"):
        assert plan.buffer_map[name].distributed_storage_kind == (
            fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
        )
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    source = ((torch.arange(32, device="cuda") % 17 - 8).reshape(1, 32) / 4).to(getattr(torch, dtype))
    residual = ((torch.arange(32, device="cuda") % 13 - 6).reshape(1, 32) / 4).to(getattr(torch, dtype))
    expected = source + (torch.nn.functional.silu(residual) if private_residual else residual)
    expected_stats = expected.float().square().sum(-1, keepdim=True).unsqueeze(0) * 2
    expected = torch.nn.functional.silu(expected)
    value, stats = torch.empty_like(expected), torch.empty_like(expected_stats)
    runtime.prepare(source, residual, value, stats)
    for _ in range(3):
        value.fill_(float("nan"))
        stats.fill_(float("nan"))
        runtime.run_into(source, residual, value, stats)
        torch.testing.assert_close(value, expected)
        torch.testing.assert_close(stats, expected_stats)
