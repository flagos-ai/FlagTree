# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Two FP32 K sub-dots must merge before the semantic BF16 part boundary."""

from math import prod

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.partitioned_linear import PartitionedLinear


def graph(*, k=8, n=8, parts=1, splits=2, lanes=(), input_lanes=(), placement=None, **attrs):
    class Graph(fm.Module):
        def forward(self):
            dtype = fm.vector_type("bfloat16", input_lanes) if input_lanes else fm.DType.BFLOAT16
            x = self.input("x", fm.tensor_type(dtype, (1, k // prod(input_lanes))), id="x")
            w = self.input("w", fm.tensor_type("bfloat16", (n, k)), id="w")
            out = fm.F.nn.partitioned_linear(
                x, w, partitions=parts, accumulation_splits=splits, output_lanes=lanes, name="projection", **attrs
            )
            self.function("main", (x, w), (out,))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def run(module, x, w):
    return TorchEvaluator(DictWeightResolver({})).run(module, {"x": x, "w": w})[0]


@pytest.mark.parametrize("lanes,input_lanes", [((), ()), ((2, 4), (2, 4))])
def test_fp32_subdots_are_explicit_and_survive_emit_resume(tmp_path, lanes, input_lanes):
    module = graph(lanes=lanes, input_lanes=input_lanes)
    x = torch.ones((1, 8), dtype=torch.bfloat16)
    w = torch.ones((8, 8), dtype=torch.bfloat16)
    w[:, 0] = 2**24
    w[:, 4] = -(2**24)
    # The installed independent FP32 matmul primitive exposes the boundary.
    whole = x.float() @ w.float().T
    split = x[:, :4].float() @ w[:, :4].float().T + x[:, 4:].float() @ w[:, 4:].float().T
    assert not torch.equal(whole.bfloat16(), split.bfloat16()), "Fixture must expose the FP32 sub-dot boundary"
    restored = fm.load_module(fm.emit_module(module, tmp_path / "projection.py"))
    assert restored.semantic_hash == module.semantic_hash
    assert restored.node_map["projection"].attrs["accumulation_splits"] == 2
    result = run(restored, x.reshape(1, 8 // prod(input_lanes), *input_lanes), w)
    torch.testing.assert_close(result.reshape(1, 8), split.bfloat16(), rtol=0, atol=0)


def test_subdots_do_not_round_to_bf16_before_the_part_sum():
    module = graph(n=1)
    x = torch.ones((1, 8), dtype=torch.bfloat16)
    w = torch.zeros((1, 8), dtype=torch.bfloat16)
    w[0, 0] = 1
    w[0, 1] = 1 / 256
    w[0, 4] = 1 / 256
    result = run(module, x, w)
    assert result.item() == 1.0078125
    premature = (
        (x[:, :4].float() @ w[:, :4].float().T).bfloat16().float()
        + (x[:, 4:].float() @ w[:, 4:].float().T).bfloat16().float()
    ).bfloat16()
    assert premature.item() == 1.0


@pytest.mark.parametrize("splits", [0, -1, True, 1.5, 3])
def test_invalid_split_count_is_rejected(splits):
    with pytest.raises(IRSchemaError, match="PartitionedLinear"):
        graph(splits=splits)


def test_default_one_is_old_checkpoint_compatible():
    module = graph(splits=1)
    assert "accumulation_splits" not in module.node_map["projection"].attrs
    x = torch.ones((1, 8), dtype=torch.bfloat16)
    w = torch.ones((8, 8), dtype=torch.bfloat16)
    w[:, 0] = 2**24
    w[:, 4] = -(2**24)
    torch.testing.assert_close(run(module, x, w), (x.float() @ w.float().T).bfloat16(), rtol=0, atol=0)
    from python.test.flagmega.ir.ops.nn.test_partitioned_linear import graph as legacy_graph

    assert legacy_graph(k=8, n=8, parts=1).semantic_hash == module.semantic_hash


def test_pattern_helper_reads_the_explicit_split_contract():
    from triton.flagmega import pattern_match as pm

    module = graph()
    assert (
        pm.try_match_root(module.node_map["projection"], pm.F.nn.is_partitioned_linear(accumulation_splits=2), module)
        is not None
    )


def test_split_cost_counts_additions_without_new_global_materialization():
    one, two = graph(k=2048, n=128, splits=1), graph(k=2048, n=128, splits=2)

    def cost(module):
        node = module.node_map["projection"]
        return PartitionedLinear.cost_factors(tuple(module.node_map[x] for x in node.inputs), node.attrs, node.type)

    a, b = cost(one), cost(two)
    assert b.cpu_cycles - a.cpu_cycles == 128
    assert a.chip_aggregate_memory_load_bytes == b.chip_aggregate_memory_load_bytes
    assert a.chip_aggregate_memory_store_bytes == b.chip_aggregate_memory_store_bytes


@pytest.mark.parametrize(
    "k,parts,expected_variant", [(320, 4, "mma"), (2048, 4, "mma"), (4096, 4, "wgmma_local"), (12288, 4, "wgmma_local")]
)
def test_candidate_checks_each_fp32_subdot_geometry(tmp_path, k, parts, expected_variant):
    import ast

    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler

    module = Compiler().compile(graph(k=k, n=80, parts=parts)).module
    (definition,) = (d for d in module.kernel_definitions if d.dispatch.semantic_op == "nn.partitioned_linear")
    assert definition.dispatch.microkernel.variant == expected_variant
    assert definition.dispatch.semantic_attrs["accumulation_splits"] == 2
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    ast.parse((artifact / "generated_kernels.py").read_text())
    restored = fm.load_module(fm.emit_module(module, tmp_path / "compiled.py"))
    assert restored.semantic_hash == module.semantic_hash


@pytest.mark.parametrize(
    "k,n,parts,lanes",
    [(320, 83, 4, ()), (4096, 83, 4, ()), (4096, 80, 4, (2, 4)), (12288, 80, 4, (2, 4)), (2048, 4096, 1, (2, 4))],
)
def test_device_subdots_round_once_and_keep_guards(tmp_path, k, n, parts, lanes):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

    module = (
        Compiler(CompileOptions(placement=fm.Placement((8, 16), "yx", "bb")))
        .compile(graph(k=k, n=n, parts=parts, lanes=lanes, input_lanes=lanes))
        .module
    )
    runtime = load(
        write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True), device="cuda:0"
    )
    generator = torch.Generator().manual_seed(9318)
    x = (torch.randint(-7, 8, (1, k), generator=generator).float() / 128).bfloat16()
    w = (torch.randint(-7, 8, (n, k), generator=generator).float() / 128).bfloat16()
    values = {}
    storages = {}
    for spec in runtime.external_arguments:
        kind = fm.logical_type(module.node_map[spec["value"]].type)
        shape = tuple(d.fixed_value for d in kind.shape) + tuple(getattr(kind.dtype, "lanes", ()))
        storage = torch.full((prod(shape) + 32,), 123.0, dtype=torch.bfloat16, device="cuda")
        storages[spec["value"]] = storage
        values[spec["value"]] = storage[16:-16].reshape(shape)
    result_name = next(spec["value"] for spec in runtime.external_arguments if spec["role"] == "result")
    try:
        for mutation in range(4):
            if mutation in (1, 2):
                x.add_(1 / 128)
            if mutation == 3:
                x.fill_(1)
                w.zero_()
                w[:, 0] = 1
                w[:, 1] = 1 / 256
                w[:, k // parts // 2] = 1 / 256
            values["x"].copy_(x.reshape(values["x"].shape).cuda())
            values["w"].copy_(w.cuda())
            values[result_name].fill_(float("nan"))
            arguments = tuple(values[spec["value"]] for spec in runtime.external_arguments)
            GeneratedTirCallGraphModule.prepare(runtime, *arguments)
            GeneratedTirCallGraphModule.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            span = k // parts
            inner = span // 2
            expected = torch.zeros((1, n), dtype=torch.float32)
            for part in range(parts):
                start = part * span
                subtotal = x[:, start : start + inner].float() @ w[:, start : start + inner].float().T
                subtotal = (
                    subtotal + x[:, start + inner : start + span].float() @ w[:, start + inner : start + span].float().T
                )
                expected = expected + subtotal.bfloat16().float()
            expected = expected.bfloat16()
            torch.testing.assert_close(values[result_name].cpu().reshape(1, n), expected, rtol=0, atol=0)
            if mutation == 3:
                assert torch.all(expected == 1.0078125)
            for storage in storages.values():
                assert bool(torch.all(storage[:16] == 123.0)) and bool(torch.all(storage[-16:] == 123.0))
            assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
