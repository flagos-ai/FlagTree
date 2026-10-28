# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A split gate must be read under the value's own policies, per shard.

The gate is elementwise over the whole value, so a head-split value forces a
head-split gate.  Rendering the gate in the result domain is what makes the
per-shard offsets agree; reading it as a replicated tensor would silently
gate every shard with head 0.
"""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from python.test.flagmega.devices import is_nvidia_sm90

TOKENS, HEADS, DIM = 1, 8, 16
EPS = 1e-5
MESH = (2, 4)


def _module():
    placement = fm.Placement(MESH, "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous((0, 1))
    value_tensor = fm.tensor_type("bfloat16", (TOKENS, HEADS, DIM))
    parameter_tensor = fm.tensor_type("float32", (DIM,))
    # Head-split value and gate; the normalized suffix stays replicated.
    value_type = fm.DistributedType(value_tensor, (broadcast, split, broadcast), placement)
    parameter_type = fm.DistributedType(parameter_tensor, (broadcast,), placement)

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", value_type, id="value")
            gate = self.input("gate", value_type, id="gate")
            scale = self.input("scale", parameter_type, id="scale")
            bias = self.input("bias", parameter_type, id="bias")
            stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False, name="stats")
            output = fm.F.nn.gated_norm_apply(
                value, stats, scale, bias, gate, axis=-1, epsilon=EPS, use_mean=False,
                activation="sigmoid", name="output")
            local = fm.F.distributed.force_boxing(output, output.type.tensor)
            self.function("main", (value, gate, scale, bias), (local,))

    return Graph(dialect="distributed", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": placement.to_data()}}).build()


def test_the_head_split_gate_is_applied_per_shard(tmp_path):
    torch = pytest.importorskip("torch")
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    module = Compiler().compile(_module()).module
    artifact = write_artifact(
        module, tmp_path / "gated-shards", target="nvidia-sm90", emit_executable=True
    )
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(31)
    value = torch.randn((TOKENS, HEADS, DIM), generator=generator).to("cuda", torch.bfloat16)
    # A per-head gate ramp: mixing up the shard offsets is observable.
    gate = (
        torch.arange(HEADS, dtype=torch.float32).reshape(1, HEADS, 1).expand(TOKENS, HEADS, DIM)
        - HEADS / 2
    ).contiguous().to("cuda", torch.bfloat16)
    scale = torch.randn((DIM,), generator=generator).to("cuda", torch.float32)
    bias = torch.zeros_like(scale)
    wide = value.float()
    normalized = (
        wide * torch.rsqrt(wide.pow(2).mean(dim=-1, keepdim=True) + EPS) * scale + bias
    )
    expected = (normalized * torch.sigmoid(gate.float())).to(torch.bfloat16)
    output = torch.empty_like(value)

    runtime.prepare(value, gate, scale, bias, output=output)
    runtime.run_into(output, value, gate, scale, bias)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    # A shard-offset mix-up would gate every head with head 0's value.
    wrong = (normalized * torch.sigmoid(gate[:, :1].float())).to(torch.bfloat16)
    assert torch.count_nonzero(expected != wrong), "The gate ramp no longer varies by head."
