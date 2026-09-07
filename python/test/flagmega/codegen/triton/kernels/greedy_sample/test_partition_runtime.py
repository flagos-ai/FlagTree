# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One collective kernel, explicit SBP, and torch.argmax as the oracle."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("hierarchy,policy,vocab,batch,batch_policy", [
    ((2, 4), fm.SBP.broadcast(), 37, 1, fm.SBP.broadcast()),
    ((2, 4), fm.SBP.split_contiguous((0, 1), 8), 64, 1, fm.SBP.broadcast()),
    ((2, 4), fm.SBP.split_block_cyclic((0, 1), 3), 37, 1, fm.SBP.broadcast()),
    ((2, 2, 2), fm.SBP.split_block_cyclic((0, 2), 4), 3, 1, fm.SBP.broadcast()),
    ((2, 4), fm.SBP.split_block_cyclic((0, 1), 8), 8195, 1, fm.SBP.broadcast()),
    ((2, 4), fm.SBP.split_block_cyclic((0, 1), 3), 37, 3, fm.SBP.broadcast()),
    ((2, 4), fm.SBP.split_contiguous((1,), 8), 32, 4, fm.SBP.split_contiguous((0,), 2)),
    ((2, 2, 2), fm.SBP.split_block_cyclic((0, 2), 3), 37, 3, fm.SBP.split_block_cyclic((1,), 2)),
])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_argmax_partition_tail_ties_infinities_and_nan(tmp_path, hierarchy, policy, vocab, batch, batch_policy, dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required by the current executable target")
    placement = fm.Placement(hierarchy, "abc"[:len(hierarchy)], "b" * len(hierarchy))
    logical = fm.tensor_type(dtype, (batch, vocab))
    distributed = fm.DistributedType(logical, (batch_policy, policy), placement)

    class Graph(fm.Module):
        def forward(self):
            value = self.input("logits", logical)
            local = fm.F.distributed.force_boxing(value, distributed)
            token = fm.F.nn.greedy_sample(local, name="sample")
            self.function("main", (value,), (
                fm.F.distributed.force_boxing(token, fm.tensor_type("int32", (batch,))),
            ))

    module = Compiler().compile(Graph(
        dialect="distributed", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}},
    ).build()).module
    artifact = write_artifact(module, tmp_path / "argmax", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.empty((batch, vocab), dtype=getattr(torch, dtype), device="cuda")
    output = runtime.create_outputs()
    runtime.prepare(value, output=output)
    cases = [
        torch.arange(batch * vocab, device="cuda").remainder(11).to(value.dtype).reshape_as(value),
        torch.full_like(value, -float("inf")),
        torch.full_like(value, float("inf")),
        torch.full_like(value, float("nan")),
    ]
    for special in [17., float("inf"), float("nan")]:
        sample = torch.zeros_like(value)
        for row in range(batch):
            sample[row, [row % vocab, vocab - 1]] = special
        cases.append(sample)
    for sample in cases:
        value.copy_(sample)
        output.fill_(-9)
        runtime.run_into(output, value)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, value.argmax(-1).int(), rtol=0, atol=0)
