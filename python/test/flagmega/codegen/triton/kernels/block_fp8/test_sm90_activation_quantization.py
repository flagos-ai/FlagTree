# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Actual scalar/packed projection paths and the complete BF16 amax domain."""

import importlib.util

import pytest
import torch
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.ir.ops.math._sm90_quantization import sm90_bf16_quantize

from python.test.flagmega.codegen.triton.kernels.block_fp8.test_dequantized_runtime import (
    _artifact,
    _require_sm90,
    _run,
)
from python.test.flagmega.ir.ops.math.test_dequantized_block_scaled_matmul import DequantizedProjection
from python.test.flagmega.ir.ops.math.test_sm90_activation_quantization import Sm90Projection, operands


@pytest.mark.parametrize("family", ("linear", "glu"))
@pytest.mark.parametrize("packed", (False, True))
def test_sm90_dot_preserves_fp8_wgmma_accumulation(tmp_path, family, packed):
    _require_sm90()
    generator = torch.Generator().manual_seed(721)
    values = operands(family=family, packed=packed)
    values["value"] = torch.randn((1, 128), generator=generator).bfloat16()
    for name in ("weight", "up"):
        if name in values:
            values[name] = torch.randn((32, 128), generator=generator).to(torch.float8_e4m3fn).reshape_as(values[name])
    for name in ("scale", "up_scale"):
        if name in values:
            values[name] = torch.tensor([[.75], [1.25]])
    q, scale = sm90_bf16_quantize(values["value"])
    rhs = q.cuda().expand(16, 128).contiguous().T
    one = torch.ones(1, device="cuda")

    def projection(name, scales):
        weight = torch.zeros((64, 128), dtype=torch.float8_e4m3fn, device="cuda")
        weight[:32].copy_(values[name].reshape(32, 128))
        dot = torch._scaled_mm(weight, rhs, scale_a=one, scale_b=one,
                               out_dtype=torch.float32, use_fast_accum=True)
        return (dot[:32, 0] * scale.cuda().reshape(())
                * values[scales].cuda().repeat_interleave(16, 0).reshape(32)).bfloat16()

    expected = projection("weight", "scale")
    if family == "glu":
        expected = torch.nn.functional.silu(expected) * projection("up", "up_scale")
    candidate = f"tir.{'block_fp8' if family == 'linear' else 'matmul_glu'}.mma"
    path = _artifact(tmp_path, Sm90Projection(family=family, packed=packed).build(), candidate)
    actual = _run(path, values, (1, 32), "bfloat16")
    torch.testing.assert_close(actual, expected.cpu().reshape(1, 32), rtol=0, atol=0)


@pytest.mark.parametrize("family", ("linear", "glu"))
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("variant", ("simt", "mma"))
def test_actual_projection_uses_sm90_quantization(tmp_path, family, packed, variant):
    _require_sm90()
    candidate = f"tir.{'block_fp8' if family == 'linear' else 'matmul_glu'}.{variant}"
    path = _artifact(tmp_path, Sm90Projection(family=family, packed=packed).build(), candidate)
    output = _run(path, operands(family=family, packed=packed), (1, 32), "bfloat16")
    expected = torch.full_like(output, .052978515625)
    if family == "glu":
        expected = torch.nn.functional.silu(expected) * expected
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_complete_finite_bf16_amax_domain_matches_sm90_instructions(tmp_path):
    _require_sm90()
    source = "import triton\nimport triton.language as tl\n" + TritonTemplateRegistry().render(
        "kernels/_activation_quantization.py.jinja", {}) + '''
@triton.jit
def domain(X, Q, S):
    group = tl.program_id(0)
    lane = tl.arange(0, 128)
    x = tl.load(X + group * 128 + lane).to(tl.float32)
    q, s = _flagmega_quantize_block_fp8(x, "dynamic_fp8_sm90")
    tl.store(Q + group * 128 + lane, q)
    tl.store(S + group, s)
'''
    path = tmp_path / "quant_domain.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("sm90_quant_domain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    amax = torch.arange(32640, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = torch.zeros((32640, 128), dtype=torch.bfloat16)
    values[:, 0], values[:, 3] = amax, -amax
    values[:, 1], values[:, 2] = (amax.float() / 3).bfloat16(), (-amax.float() / 7).bfloat16()
    edges = torch.zeros((4, 128), dtype=torch.bfloat16)
    edges[0].fill_(float("nan"))
    edges[1, :2] = torch.tensor((float("inf"), -float("inf")), dtype=torch.bfloat16)
    edges[2, :2] = torch.tensor((float("nan"), 1.), dtype=torch.bfloat16)
    edges[3, :2] = torch.tensor((float("nan"), -0.), dtype=torch.bfloat16)
    values = torch.cat((values, edges), 0)
    expected_codes, expected_scale = sm90_bf16_quantize(values)
    x = values.cuda()
    codes = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((len(values),), device="cuda")
    kernel = module.domain[(len(values),)](x, codes, scale, num_warps=4)
    torch.testing.assert_close(codes.cpu().view(torch.uint8), expected_codes.view(torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(scale.cpu().view(torch.int32), expected_scale.flatten().view(torch.int32), atol=0, rtol=0)
    assert kernel.n_spills == 0


@pytest.mark.parametrize("variant", ("simt", "mma"))
@pytest.mark.parametrize("packed,n,k,lanes", ((False, 17, 129, ()), (True, 40, 160, (2, 4))))
def test_quantized_projection_preserves_tails_and_multidimensional_output_lanes(tmp_path, variant, packed, n, k, lanes):
    from math import prod

    _require_sm90()
    value = torch.zeros((1, k), dtype=torch.bfloat16)
    value[0, 0], value[0, 1] = .050537109375, .134765625
    weight = torch.zeros((n, k))
    weight[:, 0] = 1.
    weight = weight.to(torch.float8_e4m3fn)
    if packed:
        weight = weight.reshape(n, k//32, 2, 16)
    values = {"value": value, "weight": weight, "scale": torch.ones(((n+15)//16, (k+127)//128))}
    module = DequantizedProjection(packed_weights=packed, n=n, k=k, lanes=lanes,
                                  quantization="dynamic_fp8_sm90", weight_dtype=None).build()
    output = _run(_artifact(tmp_path, module, f"tir.block_fp8.{variant}"), values,
                  (1, n//prod(lanes), *lanes), "bfloat16")
    torch.testing.assert_close(output, torch.full_like(output, .052978515625), atol=0, rtol=0)


def test_quantization_mode_survives_readonly_weight_publication(tmp_path):
    from triton.flagmega import ir as fm
    from triton.flagmega.importer import MemoryCheckpoint, TensorInfo

    _require_sm90()
    values = operands()
    weights = {key: values[key] for key in ("weight", "scale")}
    infos = {key: TensorInfo(key, fm.DType.FLOAT8_E4M3FN if key == "weight" else fm.DType.FLOAT32,
                             tuple(value.shape), "memory") for key, value in weights.items()}
    checkpoint = MemoryCheckpoint({}, infos, weights)
    module = DequantizedProjection(constants=True, quantization="dynamic_fp8_sm90", weight_dtype=None).build()
    path = _artifact(tmp_path, module, "tir.block_fp8.mma", checkpoint=checkpoint)
    output = _run(path, {"value": values["value"]}, (1, 32), "bfloat16")
    torch.testing.assert_close(output, torch.full_like(output, .052978515625), atol=0, rtol=0)


@pytest.mark.parametrize("variant", ("simt", "mma"))
@pytest.mark.parametrize("axis", ("n", "k"))
def test_sm90_quantization_preserves_distributed_group_coordinates(tmp_path, variant, axis):
    from triton.flagmega import ir as fm
    from triton.flagmega.options import CompileOptions, parse_placement

    _require_sm90()
    placement = parse_placement("x=2/b")
    broadcast = fm.SBP.broadcast()
    n_split = fm.SBP.split_contiguous((0,), granularity=32)
    k_split = fm.SBP.split_contiguous((0,), granularity=128)
    packed_k_split = fm.SBP.split_contiguous((0,), granularity=4)

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.DistributedType(fm.tensor_type("bfloat16", (1, 256)),
                               (broadcast, k_split if axis == "k" else broadcast), placement), id="value")
            weight = self.input("weight", fm.DistributedType(
                fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (64, 8)),
                (n_split if axis == "n" else broadcast, packed_k_split if axis == "k" else broadcast), placement),
                id="weight")
            scale = self.input("scale", fm.DistributedType(fm.tensor_type("float32", (4, 2)),
                                (broadcast, broadcast), placement), id="scale")
            result = fm.F.math.packed_block_scaled_matmul(value, weight, scale, k_pack=2, k_vector=16,
                output_lanes=(2, 4), weight_block_n=16, weight_block_k=128,
                activation_quantization="dynamic_fp8_sm90")
            result = fm.F.distributed.boxing(result, result.type.tensor)
            self.function("main", (value, weight, scale), (result,))

    module = Graph(dialect="high_level", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    value = torch.zeros((1, 256), dtype=torch.bfloat16)
    value[0, 0], value[0, 1] = .050537109375, .134765625
    weight = torch.zeros((64, 256))
    weight[:, 0], weight[:, 128] = 1., 1.
    scale = torch.ones((4, 2))
    expected = torch.full((1, 64), .052978515625, dtype=torch.bfloat16)
    if axis == "k":
        value[0, 128], value[0, 129] = .050537109375, .134765625
        scale[:, 1] = 2.
        expected.fill_(.1591796875)
    else:
        scale[2:, 0] = 2.
        expected[:, 32:] = .10595703125
    values = {"value": value, "weight": weight.to(torch.float8_e4m3fn).reshape(64, 8, 2, 16), "scale": scale}
    path = _artifact(tmp_path, module, f"tir.block_fp8.{variant}", options=CompileOptions(placement=placement))
    output = _run(path, values, (1, 8, 2, 4), "bfloat16").reshape(1, 64)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
