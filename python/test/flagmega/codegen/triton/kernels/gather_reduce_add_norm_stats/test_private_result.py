# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("private_result", [True, False])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_hybrid_partial_combine_reads_private_broadcast_residual_and_publishes_complete_results(
        tmp_path, private_result, dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    placement = fm.Placement((2, 4), "yx", "bb")
    source_type = fm.tensor_type(dtype, (1, 32, 8))
    value_type = fm.tensor_type(dtype, (1, 32))
    distributed = fm.DistributedType(source_type, (
        fm.SBP.broadcast(),
        fm.SBP.split_contiguous((0, ), 16),
        fm.SBP.split_contiguous((1, ), 2),
    ), placement)
    broadcast = fm.DistributedType(value_type, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)

    class Graph(fm.Module):

        def forward(self):
            source = self.input("source", source_type)
            residual = self.input("residual", value_type)
            local = fm.F.distributed.force_boxing(source, distributed)
            partial = fm.F.math.reduce_sum(local, axes=(2, ), keep_dims=False)
            addend = fm.F.distributed.force_boxing(residual, broadcast, name="private_residual")
            combined = fm.F.ntt.matmul_norm_stats_combine(partial, addend, axis=-1, use_mean=False, name="combine")
            value = fm.F.tensors.get_item(combined, 0)
            stats = fm.F.tensors.get_item(combined, 1)
            if private_result:
                # Downstream owners consume distinct parts of their private
                # copies, so writing only owner zero cannot pass this test.
                consumer_type = fm.DistributedType(value_type, (
                    fm.SBP.broadcast(),
                    fm.SBP.split_contiguous((0, 1), 4),
                ), placement)
                value = fm.F.distributed.sharded_view(value, consumer_type)
                value = fm.F.math.silu(value)
                value = fm.F.distributed.force_boxing(value, value.type.tensor)
                stats = fm.F.math.add(stats, stats)
                stats = fm.F.distributed.force_boxing(stats, stats.type.tensor)
            self.function("main", (source, residual), (value, stats))

    module = Compiler().compile(
        Graph(
            dialect="distributed",
            stage="frozen_constants",
            entry="main",
            metadata={"auto_distribution": {"placement": placement.to_data()}},
        ).build()).module
    plan = fm.verify_buffer_plan(module)
    if private_result:
        assert plan.buffer_map[
            "private_residual"].distributed_storage_kind == fm.DistributedBufferStorageKind.COMPACT_LOCAL
    expected_storage = (fm.DistributedBufferStorageKind.COMPACT_LOCAL
                        if private_result else fm.DistributedBufferStorageKind.CANONICAL_GLOBAL)
    assert plan.buffer_map["combine.0"].distributed_storage_kind == expected_storage
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    source = ((torch.arange(256, device="cuda").reshape(1, 32, 8) % 17 - 8).float() / 4).to(getattr(torch, dtype))
    residual = ((torch.arange(32, device="cuda").reshape(1, 32) % 13 - 6).float() / 4).to(getattr(torch, dtype))
    expected = source.sum(-1) + residual
    expected_stats = expected.float().square().sum(-1, keepdim=True).unsqueeze(0)
    if private_result:
        expected = torch.nn.functional.silu(expected)
        expected_stats = expected_stats * 2
    value = torch.empty_like(expected)
    stats = torch.empty_like(expected_stats)
    runtime.prepare(source, residual, value, stats)
    for _ in range(3):
        value.fill_(float("nan"))
        stats.fill_(float("nan"))
        runtime.run_into(source, residual, value, stats)
        torch.testing.assert_close(value.cpu(), expected.cpu(), rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(stats.cpu(), expected_stats.cpu(), rtol=0, atol=0)
