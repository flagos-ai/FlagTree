# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""mHC consumes a packet value directly, preserving every arithmetic boundary."""

from math import prod

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


CASES = [
    ((8,), "bfloat16", "float32", "float32", "value_fma_first", 1, 4, 4096),
    ((2, 4), "float32", "bfloat16", "bfloat16", "residual_then_value", 2, 3, 56),
    ((1,), "bfloat16", "bfloat16", "float32", "value_fma_first", 1, 1, 24),
    ((2, 4), "bfloat16", "float32", "float32", "value_fma_first", 2, 3, 40),
]


def graph(lanes, value_dtype, residual_dtype, output_dtype, mode, tokens, hc, hidden):
    mesh = fm.Placement((8, 16), "yx", "bb")

    class Graph(fm.Module):
        def forward(self):
            kinds = [
                fm.tensor_type(
                    fm.vector_type(value_dtype, lanes) if lanes else value_dtype, (tokens, hidden // prod(lanes))
                ),
                fm.tensor_type(residual_dtype, (tokens, hc, hidden)),
                fm.tensor_type("float32", (tokens, hc)),
                fm.tensor_type("float32", (tokens, hc, hc)),
            ]
            values = tuple(
                self.input(name, kind, id=name)
                for name, kind in zip(("value", "residual", "post", "comb"), kinds, strict=True)
            )
            result = fm.F.nn.mhc_post(
                *values,
                output_dtype=output_dtype,
                residual_input_dtype="bfloat16",
                accumulation_mode=mode,
                name="output",
            )
            self.function("main", values, (result,))

    return Graph(
        dialect="distributed",
        stage="frozen_constants",
        entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}},
    ).build()


def operands(value_dtype, residual_dtype, tokens, hc, hidden):
    generator = torch.Generator().manual_seed(20261018)
    return {
        "value": torch.randn((tokens, hidden), generator=generator).to(getattr(torch, value_dtype)),
        "residual": torch.randn((tokens, hc, hidden), generator=generator).to(getattr(torch, residual_dtype)),
        "post": torch.randn((tokens, hc), generator=generator),
        "comb": torch.randn((tokens, hc, hc), generator=generator),
    }


@pytest.mark.parametrize("case", CASES)
def test_mhc_post_packet_evaluation_and_roundtrip(tmp_path, case):
    lanes, value_dtype, residual_dtype, output_dtype, mode, tokens, hc, hidden = case
    original = graph((), *case[1:])
    packed = graph(*case)
    assert packed.node_map["output"].type == original.node_map["output"].type
    assert fm.load_module(fm.emit_module(packed, tmp_path / "packed.py")).semantic_hash == packed.semantic_hash
    values = operands(value_dtype, residual_dtype, tokens, hc, hidden)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    (expected,) = evaluator.run(original, values)
    values["value"] = values["value"].reshape(tokens, hidden // prod(lanes), *lanes)
    (actual,) = evaluator.run(packed, values)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("case", CASES)
def test_packed_value_reaches_generated_kernel_without_an_adapter(tmp_path, case):
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler

    compiled = Compiler().compile(graph(*case)).module
    assert not any(
        node.op in {"tensors.pack", "tensors.unpack", "tensors.cast", "ntt.vectorized_cast"} for node in compiled.nodes
    )
    fm.verify_buffer_plan(compiled)
    write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


@pytest.mark.parametrize("case", CASES)
def test_device_mhc_post_reads_packed_value_without_materializing_it(tmp_path, case):
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    lanes, value_dtype, residual_dtype, output_dtype, mode, tokens, hc, hidden = case
    runtimes = []
    outputs, guards = [], []
    try:
        for label, source in (("scalar", graph((), *case[1:])), ("packet", graph(*case))):
            compiled = Compiler().compile(source).module
            runtime = load(
                write_artifact(compiled, tmp_path / label, target="nvidia-sm90", emit_executable=True), device="cuda:0"
            )
            runtimes.append(runtime)
            storage = torch.full(
                (tokens * hc * hidden + 128,), -123.0, dtype=getattr(torch, output_dtype), device="cuda"
            )
            outputs.append(storage[64:-64].reshape(tokens, hc, hidden))
            guards.extend((storage[:64], storage[-64:]))
        values = operands(value_dtype, residual_dtype, tokens, hc, hidden)
        for mutation in range(3):
            inputs = {name: value.cuda() for name, value in values.items()}
            for index, runtime in enumerate(runtimes):
                current = dict(inputs)
                if index:
                    current["value"] = current["value"].reshape(tokens, hidden // prod(lanes), *lanes)
                arguments = tuple(
                    outputs[index] if spec["role"] == "result" else current[spec["value"]]
                    for spec in runtime.external_arguments
                )
                Runtime.prepare(runtime, *arguments)
                Runtime.run_into(runtime, *arguments)
                torch.cuda.synchronize()
                assert runtime.resource_report["spill_bytes"] == 0
            assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
            assert all(bool((guard == -123).all()) for guard in guards)
            values["value"].add_(1 / 128)
    finally:
        for runtime in runtimes:
            runtime.close()
