# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Actual kernels preserve activation values and round decoded weights first."""

import json
from dataclasses import replace
from math import prod

import pytest
import torch
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.selection import apply_plan, override_plan

from python.test.flagmega.ir.ops.math.test_dequantized_block_scaled_matmul import (
    DequantizedProjection,
    boundary_operands,
)


def _artifact(tmp_path, module, variant, *, options=None, checkpoint=None, dequant_mma_splits=None):
    compiler = Compiler(options) if options is not None else Compiler()
    if dequant_mma_splits is not None:
        from triton.flagmega.targets import NvidiaSm90Target
        base = NvidiaSm90Target().triton_implementation_model
        catalog = replace(base, implementations=tuple(
            replace(value, parameters={**value.parameters, "dequant_mma_splits": dequant_mma_splits})
            if value.id == "tir.block_fp8.mma" else value for value in base.implementations))
        compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    proposed = compiler.compile(module, stop_after="propose-tir").module
    assert not any(node.op == "tensors.pack" for node in proposed.nodes)
    choices = [(point.id, variant) for point in proposed.selection_points
               if any(candidate.id == variant for candidate in point.candidates)]
    assert choices
    selected = apply_plan(proposed, override_plan(proposed, choices))
    path = write_artifact(compiler.compile(selected).module, tmp_path / "dequantized",
                          target="nvidia-sm90", checkpoint=checkpoint, emit_executable=True)
    return path


def _run(path, values, shape, dtype):
    runtime = load(path, device="cuda:0")
    try:
        output = torch.empty(shape, dtype=getattr(torch, dtype), device="cuda")
        # Packing may replace a public weight parameter with its packed ABI.
        # Bind the final entry's parameter positions and physical lane shape.
        abi = json.loads((path / "artifact.json").read_text())["abi"]
        entry = next(function for function in abi["functions"] if function["name"] == abi["entry"])
        device = {}
        for spec, value in zip(entry["parameters"], values.values(), strict=True):
            tensor = spec["type"].get("tensor", spec["type"])
            lanes = tuple(tensor["dtype"].get("lanes", ())) if isinstance(tensor["dtype"], dict) else ()
            physical_shape = tuple(dim["value"] for dim in tensor["shape"]) + lanes
            device[spec["node"]] = value.cuda().reshape(physical_shape)
        args = tuple(output if spec["role"] == "result" else device[spec["value"]]
                     for spec in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *args)
        GeneratedTirCallGraphModule.run_into(runtime, *args)
        torch.cuda.synchronize()
        assert runtime.resource_report["spill_bytes"] == 0
        return output.cpu()
    finally:
        runtime.close()


def _require_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")


@pytest.mark.parametrize("packed", [False, True])
def test_dequantized_mma_keeps_the_accumulator_across_scale_blocks(tmp_path, packed):
    _require_sm90()
    n, k = 32, 384
    value = torch.ones((1, k), dtype=torch.bfloat16)
    weight = torch.zeros((n, k), dtype=torch.float32)
    weight[:, 0] = 1.
    weight[:, 128:256] = 1.
    weight[:, 256] = -1.
    weight[n // 2:, -1] = 2. ** -9
    scales = torch.tensor([[2. ** 24, 2. ** -6, 2. ** 24]]).repeat(2, 1)
    weight = weight.to(torch.float8_e4m3fn)
    decoded = (weight.float() * scales.repeat_interleave(16, 0).repeat_interleave(128, 1)).bfloat16()
    # An independent full BF16 GEMM retains its FP32 MMA accumulator across
    # all three scale blocks. Restarting each block and adding partial dots
    # preserves a small sum that the specified running accumulator rounds off.
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        expected = (value.cuda().expand(16, -1).contiguous() @ decoded.cuda().T)[:1].cpu()
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
    assert torch.equal(expected[:, :n // 2], torch.zeros((1, n // 2), dtype=torch.bfloat16))
    assert torch.equal(expected[:, n // 2:], torch.full((1, n // 2), 32768., dtype=torch.bfloat16))
    physical_weight = weight.reshape(n, k // 32, 2, 16) if packed else weight
    values = {"value": value, "weight": physical_weight, "scale": scales}
    graph = DequantizedProjection(n=n, k=k, packed_weights=packed).build()
    actual = _run(_artifact(tmp_path, graph, "tir.block_fp8.mma"), values, (1, n), "bfloat16")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_dequantized_mma_preserves_the_selected_split_partitions(tmp_path, packed):
    _require_sm90()
    n, k = 32, 512
    value = torch.ones((1, k), dtype=torch.bfloat16)
    weight = torch.zeros((n, k), dtype=torch.float32)
    weight[:, 0], weight[:, 256:384], weight[:, 384] = 1., 1., -1.
    weight[n // 2:, -1] = 2. ** -9
    scale = torch.tensor([[2. ** 24, 1., 2. ** -6, 2. ** 24]]).repeat(2, 1)
    encoded = weight.to(torch.float8_e4m3fn)
    decoded = (encoded.float() * scale.repeat_interleave(16, 0).repeat_interleave(128, 1)).bfloat16()
    # Within each chosen partition the exact partial sum is representable in
    # FP32. An FP64 oracle is therefore independent of MMA scheduling here.
    expected = sum((value[:, start:start + 256].double() @ decoded[:, start:start + 256].double().T).float()
                   for start in (0, 256)).bfloat16()
    assert torch.equal(expected[:, :16], torch.full((1, 16), 2., dtype=torch.bfloat16))
    physical = encoded.reshape(n, k // 32, 2, 16) if packed else encoded
    values = {"value": value, "weight": physical, "scale": scale}
    graph = DequantizedProjection(n=n, k=k, packed_weights=packed).build()
    split = _run(_artifact(tmp_path / "split", graph, "tir.block_fp8.mma", dequant_mma_splits=2),
                  values, (1, n), "bfloat16")
    full = _run(_artifact(tmp_path / "full", graph, "tir.block_fp8.mma"), values, (1, n), "bfloat16")
    torch.testing.assert_close(split, expected, rtol=0, atol=0)
    assert torch.equal(full[:, :16], torch.zeros((1, 16), dtype=torch.bfloat16))
    assert torch.equal(split[:, 16:], full[:, 16:])


@pytest.mark.parametrize("variant", ["tir.block_fp8.simt", "tir.block_fp8.mma"])
@pytest.mark.parametrize("boundary", ["activation", "weight"])
@pytest.mark.parametrize("packed,lanes,n,k", [(False, (), 17, 129), (True, (8,), 40, 160),
                                              (True, (2, 4), 40, 160)])
def test_dequantized_precision_survives_gpu_packing_and_tail_geometry(tmp_path, variant, boundary, packed, lanes, n, k):
    _require_sm90()
    values, expected = boundary_operands(n, k, boundary=boundary, packed=packed)
    module = DequantizedProjection(packed_weights=packed, lanes=lanes, n=n, k=k).build()
    path = _artifact(tmp_path, module, variant)
    output = _run(path, values, (1, n // prod(lanes), *lanes), "bfloat16")
    torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["tir.block_fp8.simt", "tir.block_fp8.mma"])
def test_matching_float16_inputs_use_the_declared_weight_rounding(tmp_path, variant):
    _require_sm90()
    values, expected = boundary_operands(dtype="float16")
    module = DequantizedProjection(dtype="float16", weight_dtype="float16").build()
    output = _run(_artifact(tmp_path, module, variant), values, (1, 32), "float16")
    torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)


def test_float32_activation_keeps_bits_that_bfloat16_would_discard(tmp_path):
    _require_sm90()
    values, _ = boundary_operands(dtype="float32")
    values["value"][0, 1] += 2 ** -20
    module = DequantizedProjection(dtype="float32").build()
    output = _run(_artifact(tmp_path, module, "tir.block_fp8.simt"), values, (1, 32), "float32")
    torch.testing.assert_close(output, torch.full_like(output, .81640625 + 2 ** -20), rtol=0, atol=0)


def test_mma_selects_one_repeated_row_without_overflowing_an_extra_sum(tmp_path):
    _require_sm90()
    values, _ = boundary_operands()
    values["value"].zero_()
    values["value"][0, 1] = 1e38
    module = DequantizedProjection().build()
    output = _run(_artifact(tmp_path, module, "tir.block_fp8.mma"), values, (1, 32), "bfloat16")
    torch.testing.assert_close(output, values["value"][:, 1:2].expand_as(output), rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["tir.block_fp8.simt", "tir.block_fp8.mma"])
@pytest.mark.parametrize("axis", ["n", "k"])
def test_dequantization_uses_global_scale_coordinates_for_distributed_owners(tmp_path, variant, axis):
    from triton.flagmega import ir as fm
    from triton.flagmega.options import CompileOptions, parse_placement

    _require_sm90()
    placement = parse_placement("x=2/b")
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous((0,), granularity=32)
    reduction = fm.SBP.split_contiguous((0,), granularity=128)
    packed_reduction = fm.SBP.split_contiguous((0,), granularity=4)

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.DistributedType(fm.tensor_type("bfloat16", (1, 256)),
                               (broadcast, reduction if axis == "k" else broadcast), placement), id="value")
            weight = self.input("weight", fm.DistributedType(
                fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (64, 8)),
                (split if axis == "n" else broadcast, packed_reduction if axis == "k" else broadcast), placement),
                id="weight")
            scale = self.input("scale", fm.DistributedType(fm.tensor_type("float32", (4, 2)),
                                (broadcast, broadcast), placement), id="scale")
            result = fm.F.math.packed_block_scaled_matmul(
                value, weight, scale, k_pack=2, k_vector=16, output_lanes=(2, 4),
                weight_block_n=16, weight_block_k=128, activation_quantization="none",
                weight_dequant_dtype="bfloat16")
            if axis == "k":
                assert result.type.partial == fm.SBP.partial((0,))
            result = fm.F.distributed.boxing(result, result.type.tensor)
            self.function("main", (value, weight, scale), (result,))

    module = Graph(dialect="high_level", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    local_weight = fm.local_shape(module.node_map["weight"].type)
    assert local_weight[0].fixed_value == (32 if axis == "n" else 64)
    assert local_weight[1].fixed_value == (4 if axis == "k" else 8)
    value = torch.zeros((1, 256), dtype=torch.bfloat16)
    weight = torch.zeros((64, 256))
    weight[:, 0], weight[:, 128] = 1., 1.
    value[0, 0] = 129.
    scale = torch.full((4, 2), 1.00390625)
    if axis == "k":
        value[0, 128] = 1.
        scale[:, 1] = 2.
        expected = torch.full((1, 64), 131., dtype=torch.bfloat16)
    else:
        scale[2:, 0] = 2.0078125
        expected = torch.full((1, 64), 129., dtype=torch.bfloat16)
        expected[:, 32:] = 258.
    values = {"value": value, "weight": weight.to(torch.float8_e4m3fn).reshape(64, 8, 2, 16), "scale": scale}
    path = _artifact(tmp_path, module, variant, options=CompileOptions(placement=placement))
    output = _run(path, values, (1, 8, 2, 4), "bfloat16").reshape(1, 64)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("quantization,weight_dtype,expected", [("dynamic_fp8", "bfloat16", .859375),
                                                               ("none", None, .8203125)])
def test_simt_preserves_independent_activation_and_weight_precision(tmp_path, quantization, weight_dtype, expected):
    _require_sm90()
    values, _ = boundary_operands()
    values["scale"].fill_(1.00390625)
    module = DequantizedProjection(quantization=quantization, weight_dtype=weight_dtype).build()
    output = _run(_artifact(tmp_path, module, "tir.block_fp8.simt"), values, (1, 32), "bfloat16")
    torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)


def test_readonly_weight_artifact_keeps_fp8_bytes_and_rounds_only_in_the_consumer(tmp_path):
    from triton.flagmega import ir as fm
    from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
    from triton.flagmega.ir.distributed_inference import tensor_of
    from triton.flagmega.ir.ops.core import tensor_nbytes

    _require_sm90()
    values, expected = boundary_operands(boundary="weight")
    weights = {name: values[name] for name in ("weight", "scale")}
    infos = {name: TensorInfo(name, fm.DType.FLOAT8_E4M3FN if name == "weight" else fm.DType.FLOAT32,
                               tuple(value.shape), "memory") for name, value in weights.items()}
    checkpoint = MemoryCheckpoint({}, infos, weights)
    frozen = Compiler().compile(DequantizedProjection(constants=True).build(), stop_after="freeze-constants").module
    assets = [node for node in frozen.nodes if node.op == "builtin.const_asset"]
    codes = [node for node in assets if getattr(tensor_of(node.type).dtype, "elem_type", tensor_of(node.type).dtype)
             == fm.DType.FLOAT8_E4M3FN]
    assert len(codes) == 1 and tensor_nbytes(tensor_of(codes[0].type)) == 32 * 128
    assert not any(node.op == "tensors.cast" for recipe in frozen.constant_recipes for node in recipe.nodes)
    output = _run(_artifact(tmp_path, frozen, "tir.block_fp8.mma", checkpoint=checkpoint),
                   {"value": values["value"]}, (1, 32), "bfloat16")
    torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
