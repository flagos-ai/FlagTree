# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SIMT/MMA kernels must store every scalar of a packed activation result."""

from math import ceil, prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.math._block_scaled import block_scaled_linear
from triton.flagmega.options import CompileOptions, parse_placement
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.selection import apply_plan, override_plan
from python.test.flagmega.devices import is_nvidia_sm90


def linear_module(*, dtype="bfloat16", output_lanes=(), consumer_lanes=(),
                  packed_weights=True, n=56, k=160):
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type(dtype, (1, k)), id="value")
            kind = (fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (n, k // 32))
                    if packed_weights else fm.tensor_type("float8_e4m3fn", (n, k)))
            weight = self.input("weight", kind, id="weight")
            scale = self.input("weight_scale", fm.tensor_type("float32", (ceil(n / 16), ceil(k / 128))), id="weight_scale")
            kwargs = dict(weight_block_n=16, weight_block_k=128, output_lanes=output_lanes, name="projection")
            output = (fm.F.math.packed_block_scaled_matmul(value, weight, scale, k_pack=2, k_vector=16, **kwargs)
                      if packed_weights else fm.F.math.block_scaled_matmul(value, weight, scale, **kwargs))
            if consumer_lanes:
                output = fm.F.tensors.pack(output, consumer_lanes, axes=(-1,) * len(consumer_lanes))
            self.function("main", (value, weight, scale), (output,))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def artifact(tmp_path, name, module, variant, *, placement=None):
    compiler = Compiler(CompileOptions(placement=placement))
    proposals = compiler.compile(module, stop_after="propose-tir").module
    choices = [(point.id, variant) for point in proposals.selection_points
               if any(candidate.id == variant for candidate in point.candidates)]
    assert choices, "the requested device implementation must actually be tested"
    selected = apply_plan(proposals, override_plan(proposals, choices))
    lowered = Compiler(CompileOptions(placement=placement)).compile(selected).module
    return write_artifact(lowered, tmp_path / name, target="nvidia-sm90", emit_executable=True)


def run(path, values, shape, torch):
    runtime = load(path, device="cuda:0")
    output = torch.empty(shape, dtype=values["value"].dtype, device="cuda")
    arguments = tuple(output if spec["role"] == "result" else values[spec["value"]]
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()
    return output


@pytest.mark.parametrize("variant", ["tir.block_fp8.simt", "tir.block_fp8.mma"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("lanes", [(8,), (2, 4), (1,)])
def test_fp8_packed_store_matches_scalar_kernel_and_evaluator(tmp_path, variant, dtype, lanes):
    torch = pytest.importorskip("torch")
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA required")
    generator = torch.Generator().manual_seed(31)
    values = {"value": torch.randn(1, 160, generator=generator).to(getattr(torch, dtype)),
              "weight": torch.randn(56, 160, generator=generator).to(torch.float8_e4m3fn).reshape(56, 5, 2, 16),
              "weight_scale": torch.rand(4, 2, generator=generator) + .25}
    expected = TorchEvaluator(DictWeightResolver({})).run(linear_module(dtype=dtype), values)[0]
    device_values = {name: value.cuda() for name, value in values.items()}
    results = []
    for name, kwargs in (("scalar", {}), ("demand", {"consumer_lanes": lanes}), ("direct", {"output_lanes": lanes})):
        path = artifact(tmp_path, name, linear_module(dtype=dtype, **kwargs), variant)
        source = (path / "generated_kernels.py").read_text()
        assert f"# flagmega-kernel: block_fp8/{variant.rsplit('.', 1)[-1]}" in source
        assert "# flagmega-kernel: pack/" not in source
        shape = (1, 56) if name == "scalar" else (1, 56 // prod(lanes), *lanes)
        results.append(run(path, device_values, shape, torch).reshape(1, 56).cpu())
    torch.testing.assert_close(results[1], results[0], rtol=0, atol=0)
    torch.testing.assert_close(results[2], results[0], rtol=0, atol=0)
    torch.testing.assert_close(results[0], expected, rtol=2e-5 if dtype == "float32" else 8e-3,
                               atol=2e-5 if dtype == "float32" else 8e-3)
    if dtype == "float32":
        assert not torch.equal(results[0], results[0].bfloat16().float()), "FP32 output must not silently round to BF16"


@pytest.mark.parametrize("variant", ["tir.block_fp8.simt", "tir.block_fp8.mma"])
def test_fp8_packed_output_covers_distributed_n_owners(tmp_path, variant):
    torch = pytest.importorskip("torch")
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA required")
    generator = torch.Generator().manual_seed(37)
    values = {"value": torch.randn(1, 160, generator=generator).bfloat16().cuda(),
              "weight": torch.randn(64, 160, generator=generator).to(torch.float8_e4m3fn).reshape(64, 5, 2, 16).cuda(),
              "weight_scale": (torch.rand(4, 2, generator=generator) + .25).cuda()}
    module = linear_module(n=64, output_lanes=(2, 4))
    path = artifact(tmp_path, "n-owners", module, variant, placement=parse_placement("yx=1x2/bb"))
    actual = run(path, values, (1, 8, 2, 4), torch).reshape(1, 64)
    expected = block_scaled_linear(values["value"], values["weight"].reshape(64, 160), values["weight_scale"],
                                   block_n=16, block_k=128)
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize("variant", ["tir.block_fp8.simt", "tir.block_fp8.mma"])
def test_fp8_packed_output_preserves_k_partial_and_scale_owner(tmp_path, variant):
    torch = pytest.importorskip("torch")
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA required")
    placement = parse_placement("x=2/b")
    broadcast = fm.SBP.broadcast()

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.DistributedType(fm.tensor_type("float32", (1, 256)),
                               (broadcast, fm.SBP.split_contiguous((0,), granularity=128)), placement), id="value")
            weight = self.input("weight", fm.DistributedType(
                fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (64, 8)),
                (broadcast, fm.SBP.split_contiguous((0,), granularity=4)), placement), id="weight")
            scale = self.input("weight_scale", fm.DistributedType(fm.tensor_type("float32", (4, 2)),
                                (broadcast, broadcast), placement), id="weight_scale")
            partial = fm.F.math.packed_block_scaled_matmul(value, weight, scale, k_pack=2, k_vector=16,
                       weight_block_n=16, weight_block_k=128, output_lanes=(2, 4), name="projection")
            assert partial.type.partial == fm.SBP.partial((0,))
            result = fm.F.distributed.boxing(partial, partial.type.tensor)
            self.function("main", (value, weight, scale), (result,))

    module = Graph(dialect="high_level", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    generator = torch.Generator().manual_seed(41)
    values = {"value": torch.randn(1, 256, generator=generator).cuda(),
              "weight": torch.randn(64, 256, generator=generator).to(torch.float8_e4m3fn).reshape(64, 8, 2, 16).cuda(),
              "weight_scale": (torch.rand(4, 2, generator=generator) + .25).cuda()}
    path = artifact(tmp_path, "k-owners", module, variant, placement=placement)
    actual = run(path, values, (1, 8, 2, 4), torch).reshape(1, 64)
    expected = block_scaled_linear(values["value"], values["weight"].reshape(64, 256), values["weight_scale"],
                                   block_n=16, block_k=128)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
