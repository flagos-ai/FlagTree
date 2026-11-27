# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Row vectorization must preserve the scalar kernel's FP32 reduction tree."""

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.fixture
def tiled_model(monkeypatch):
    tutorial = Path(__file__).parents[5] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    factory = import_module("local_optimizations.reduction_rows_schedule").implementation_model
    return factory(NvidiaSm90Target().triton_implementation_model)


def test_row_tiling_is_a_separate_storage_preserving_choice(tiled_model):
    ordinary = tiled_model.implementation("tir.reduce_sum.local")
    tiled = tiled_model.implementation("tir.reduce_sum.tiled_rows")
    assert tiled.contract == ordinary.contract
    assert tiled.facts.get("result_memory_scope", "inferred") == "inferred"
    assert tiled.parameters["elements_per_program"] == ordinary.parameters["elements_per_program"]
    assert tiled_model.preferences["reduce_sum"][0] == tiled.id
    assert NvidiaSm90Target().triton_implementation_model.implementation(tiled.id) is None


@pytest.mark.parametrize("shape,axes,keep_dims,lanes", [
    ((4, 1, 4096), (0,), False, ()),
    ((7, 777), (0,), False, ()),
    ((1, 65), (0,), False, ()),
    ((32, 72), (0,), False, (8,)),
    ((128, 3), (0,), True, ()),
    ((256, 3), (0,), False, ()),
    ((513, 3), (0,), False, ()),
    ((2, 7, 3), (2, 0), True, ()),
    ((3, 4, 72), (1,), False, (2, 4)),
    ((3, 0, 7), (1,), False, ()),
    ((0, 7), (1,), True, ()),
    ((3, 72), (), False, (8,)),
])
@pytest.mark.parametrize("warps", [4, 8])
def test_tiled_rows_match_scalar_fp32_bits(tmp_path, tiled_model, shape, axes, keep_dims, lanes, warps):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = replace(primitive_module(fm.get_definition("math.reduce_sum"), (fm.tensor_type("float32", shape),),
                                      axes=axes, keep_dims=keep_dims, output_lanes=lanes), stage="frozen_constants")
    runtimes = []
    try:
        for label, model in (("scalar", NvidiaSm90Target().triton_implementation_model), ("tiled", tiled_model)):
            model = replace(model, implementations=tuple(
                replace(value, parameters={**value.parameters, "compute_num_warps": warps})
                if value.family == "reduce_sum" else value for value in model.implementations))
            compiler = Compiler()
            compiler.target = NvidiaSm90Target(triton_implementation_model=model)
            compiled = compiler.compile(module).module
            path = write_artifact(compiled, tmp_path / label, target="nvidia-sm90", emit_executable=True)
            if label == "tiled":
                source = (path / "generated_kernels.py").read_text()
                assert "# flagmega-kernel: reduce_sum/tiled_rows" in source
                assert "tl.sum(_fm_values," in source or not axes
            runtimes.append(load(path, device="cuda:0"))
        generator = torch.Generator(device="cuda").manual_seed(4837)
        for mutation in range(3):
            value = torch.randn(shape, generator=generator, dtype=torch.float32, device="cuda")
            if mutation == 1:
                # Cancellation and widely separated exponents expose tree changes.
                exponents = torch.arange(value.numel(), device="cuda").reshape(shape) % 41 - 20
                value = torch.ldexp(value, exponents.int())
            elif mutation == 2:
                value = (torch.arange(value.numel(), device="cuda").reshape(shape) % 31 - 15).float() / 8
            results = []
            for runtime in runtimes:
                runtime.prepare(value)
                results.append(runtime.run(value).clone())
                torch.cuda.synchronize()
                assert runtime.resource_report["spill_bytes"] == 0
            assert torch.equal(results[0].view(torch.uint8), results[1].view(torch.uint8))
    finally:
        for runtime in runtimes:
            runtime.close()


@pytest.mark.parametrize("reduce_split", [False, True])
def test_tiled_rows_keep_packed_owner_and_partial_components(tmp_path, tiled_model, reduce_split):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    mesh = fm.Placement((2, 2), "xy", "bb")
    policies = (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 2) if reduce_split else fm.SBP.broadcast(),
                fm.SBP.split_block_cyclic((1,), 16))

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.DistributedType(fm.tensor_type("float32", (3, 4, 72)), policies, mesh))
            partial = fm.F.math.reduce_sum(value, axes=(1,), keep_dims=False, output_lanes=(2, 4))
            result = fm.F.distributed.force_boxing(partial, partial.type.tensor)
            self.function("main", (value,), (result,))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()
    values = torch.arange(3 * 4 * 72, device="cuda").remainder(31).float().reshape(3, 4, 72) / 8
    results, runtimes = [], []
    try:
        for label, model in (("scalar", NvidiaSm90Target().triton_implementation_model), ("tiled", tiled_model)):
            compiler = Compiler(CompileOptions(placement=mesh))
            compiler.target = NvidiaSm90Target(triton_implementation_model=model)
            path = write_artifact(compiler.compile(module).module, tmp_path / label,
                                  target="nvidia-sm90", emit_executable=True)
            runtime = load(path, device="cuda:0")
            runtimes.append(runtime)
            runtime.prepare(values)
            results.append(runtime.run(values).clone())
            torch.cuda.synchronize()
            assert runtime.resource_report["spill_bytes"] == 0
        assert torch.equal(results[0].view(torch.uint8), results[1].view(torch.uint8))
        torch.testing.assert_close(results[1].reshape(3, 72), values.sum(1), rtol=0, atol=0)
    finally:
        for runtime in runtimes:
            runtime.close()
