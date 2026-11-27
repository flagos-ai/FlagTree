# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pair-local scalar/vector RoPE on distributed storage."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("lanes", [1, 8])
@pytest.mark.parametrize("rotary", [64, 128])
def test_rope_preserves_head_dim_shards_without_cross_owner_reads(tmp_path, lanes, rotary):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    mesh = fm.Placement((2, 4), "yx", "bb")
    broad = fm.SBP.broadcast()
    dtype = fm.DType.BFLOAT16 if lanes == 1 else fm.vector_type("bfloat16", (lanes,))
    table_dtype = fm.DType.FLOAT32 if lanes == 1 else fm.vector_type("float32", (2, lanes))
    value_type = fm.tensor_type(dtype, (1, 3, 128 // lanes))
    table_type = fm.tensor_type(table_dtype, (1, 1, rotary // (1 if lanes == 1 else 2 * lanes)))
    distributed = fm.DistributedType(value_type, (broad, fm.SBP.split_block_cyclic((0,), 1),
        fm.SBP.split_block_cyclic((1,), 8 // lanes)), mesh)

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", value_type)
            cos = self.input("cos", table_type)
            sin = self.input("sin", table_type)
            shard = fm.F.distributed.boxing(value, distributed)
            # The producer forces an ordinary owner-local temporary.
            shard = fm.F.math.silu(shard) if lanes == 1 else fm.F.math.vectorized_unary(
                shard, unary_op="silu", metadata={"selected_vectorization": "vectorization.last_axis",
                    "selected_vector_axes": (2,), "selected_vector_lanes": (lanes,)})
            tables = tuple(fm.F.distributed.boxing(table, fm.DistributedType(table_type, (broad,) * 3, mesh))
                           for table in (cos, sin))
            op = fm.F.nn.rope if lanes == 1 else fm.F.ntt.vectorized_rope
            output = op(shard, *tables, rotary_dim=rotary, name="rope")
            assert output.type == distributed
            self.function("main", (value, cos, sin), (fm.F.distributed.boxing(output, value_type),))

    module = Graph(dialect="high_level", stage="distributed", entry="main",
                   metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator(device="cuda").manual_seed(281)
    value = torch.randn((1, 3, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
    cos, sin = (torch.randn((1, 1, rotary), generator=generator, device="cuda") for _ in range(2))
    source = torch.nn.functional.silu(value.float()).bfloat16().float()
    prefix = source[..., :rotary]
    partner = torch.cat((-prefix[..., rotary // 2:], prefix[..., :rotary // 2]), -1)
    expected = torch.cat((prefix * cos + partner * sin, source[..., rotary:]), -1).bfloat16()
    output = torch.empty_like(value)
    arguments = (value, cos, sin) if lanes == 1 else (
        value.reshape(1, 3, 128 // lanes, lanes),
        cos.reshape(1, 1, rotary // (2 * lanes), 2, lanes),
        sin.reshape(1, 1, rotary // (2 * lanes), 2, lanes),
    )
    result = output if lanes == 1 else output.reshape(1, 3, 128 // lanes, lanes)
    runtime.prepare(*arguments, output=result)
    runtime.run_into(result, *arguments)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
