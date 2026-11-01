# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A second owner view must read every CTA's populated private replica."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
def test_private_concat_reshape_repartition_publishes_exact_bytes(tmp_path, lanes):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

    mesh = fm.Placement((8, 16), "yx", "bb")
    dtype = fm.vector_type("bfloat16", lanes) if lanes else "bfloat16"
    tensor = fm.tensor_type(dtype, (1, 64))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, mesh)

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", tensor, id="value")
            local = fm.F.distributed.sharded_view(value, broadcast)
            pair = fm.F.tensors.concat(local, local, axis=1, name="pair")
            first = fm.F.distributed.sharded_view(pair, replace(pair.type,
                axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16))))
            reshaped = fm.F.tensors.reshape(first, (1, 1, 128))
            final = fm.F.distributed.sharded_view(reshaped, replace(reshaped.type,
                axis_policies=(fm.SBP.broadcast(), fm.SBP.broadcast(),
                               fm.SBP.split_contiguous((0, 1), 1))))
            output = fm.F.distributed.force_boxing(final, final.type.tensor)
            self.function("main", (value,), (output,))

    graph = Graph(dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()
    module = Compiler().compile(graph).module
    source = fm.verify_buffer_plan(module).buffer_map["pair"]
    assert source.storage == "block_local_data"
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    try:
        value = torch.empty((1, 64, *lanes), dtype=torch.bfloat16, device="cuda")
        count = 128 * (8 if lanes else 1)
        guarded = torch.full((count + 64,), -123., dtype=torch.bfloat16, device="cuda")
        output = guarded[32:-32].reshape((1, 1, 128, *lanes))
        arguments = tuple(output if spec["role"] == "result" else value for spec in runtime.external_arguments)
        Runtime.prepare(runtime, *arguments)
        indices = torch.arange(value.numel(), device="cuda").reshape(value.shape)
        for mutation in range(3):
            value.copy_(((indices * 3 + mutation * 11) % 113 - 56).float() / 8)
            output.fill_(float("nan"))
            Runtime.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            expected = torch.cat((value, value), dim=1).reshape(output.shape)
            assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            assert bool((guarded[:32] == -123.).all() & (guarded[-32:] == -123.).all())
        assert runtime.resource_report["spill_bytes"] == 0
        assert runtime.resource_report["grid"] == [128, 1, 1]
        assert source.distributed_storage_kind is fm.DistributedBufferStorageKind.COMPACT_LOCAL
    finally:
        runtime.close()
