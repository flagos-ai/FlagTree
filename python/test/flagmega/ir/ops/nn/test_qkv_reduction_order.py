# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""QKV projections retain one ordered accumulator across the complete K."""

import pytest
import torch
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.runtime import load


def module(rows=1):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (rows, 2048)), id="x")
            weights = tuple(self.weight(n, fm.tensor_type("bfloat16", (2048, 512)),
                                        id=n, source="memory", key=n)
                            for n in ("q", "k", "v"))
            none = fm.F.builtin.none()
            qkv = fm.F.nn.qkv_parallel_linear(
                x, *weights, *([none] * 9), num_heads=8, num_kv_heads=8,
                output_data_type="bfloat16", reduction_order="mma_k16")
            self.function("main", (x,), fm.F.tensors.get_items(qkv, 0, 1, 2))
    return Graph(dialect="high_level", stage="imported", entry="main").build()


def test_qkv_ordered_contract_survives_python_resume(tmp_path):
    graph = module()
    assert fm.load_module(fm.emit_module(graph, tmp_path / "qkv.py")) == graph


def test_qkv_ordered_contract_rejects_independent_k_shards():
    graph = module()
    call = next(n for n in graph.nodes if n.op == "nn.qkv_parallel_linear")
    inputs = tuple(graph.node_map[n] for n in call.inputs)
    source = replace(inputs[0], type=fm.DistributedType(
        inputs[0].type, (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 16)),
        fm.Placement((8,), "x", "b")))
    with pytest.raises(IRSchemaError, match="independently accumulated K shards"):
        fm.get_definition(call.op).prepare((source, *inputs[1:]), call.attrs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", (1, 32))
def test_ordered_qkv_matches_independent_bf16_projection(tmp_path, rows):
    generator = torch.Generator(device="cuda").manual_seed(509)
    x = torch.randn((rows, 2048), device="cuda", generator=generator).bfloat16()
    weights = tuple(torch.randn((2048, 512), device="cuda", generator=generator).bfloat16() for _ in range(3))
    checkpoint = MemoryCheckpoint(
        {}, {n: TensorInfo(n, fm.DType.BFLOAT16, (2048, 512), "memory") for n in ("q", "k", "v")},
        {n: w.cpu() for n, w in zip(("q", "k", "v"), weights)})
    graph = module(rows)
    artifact = write_artifact(Compiler().compile(graph).module, tmp_path / "artifact",
                              target="nvidia-sm90", checkpoint=checkpoint, emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    expected = tuple(x @ weight for weight in weights)
    outputs = tuple(torch.empty_like(value) for value in expected)
    try:
        runtime.prepare(x, *outputs)
        runtime.run_into(x, *outputs)
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        reports = runtime.resource_report
        if not isinstance(reports, tuple):
            reports = (reports,)
        assert all(report["spill_bytes"] == 0 for report in reports)
    finally:
        runtime.close()
