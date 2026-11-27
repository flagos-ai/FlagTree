# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ordered BF16 MMA exposes its FP32 result through the ordinary compiler."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia.dense_matmul import packed_mma_accumulator_implementations

from python.test.flagmega.codegen.triton.kernels.dense_matmul.test_small_bf16_schedule import _graph

IDENTITY = "tir.dense_matmul.packed_k_major_mma_accumulator"


def _compile(mesh, *, k=256, n=24, m=1, splits=1, axes=(), block_k=128, weight_load_layout=None):
    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            proposal = super().propose_tir(module)
            return apply_plan(proposal, override_plan(proposal, [("tir.projection", IDENTITY)],
                policy="test.fp32_mma", rationale="Preserve the FP32 projection accumulator."))

    base = NvidiaSm90Target().triton_implementation_model
    implementation, = packed_mma_accumulator_implementations()
    implementation = replace(implementation, parameters={**implementation.parameters,
        "accumulation_splits": splits, "block_k": block_k})
    if weight_load_layout is not None:
        implementation = replace(implementation, parameters={**implementation.parameters,
            "weight_load_layout": weight_load_layout})
    catalog = replace(base, implementations=(*base.implementations, implementation))
    compiler = Compiler(CompileOptions(placement=mesh))
    compiler.target = Target(triton_implementation_model=catalog)
    return compiler.compile(_graph(m, k, n, mesh, output="float32", axes=axes)).module


@pytest.mark.parametrize("splits", [1, 2])
def test_accumulator_choice_roundtrips_and_renders(tmp_path, splits):
    module = _compile(fm.Placement((1, 1), "yx", "bb"), splits=splits)
    assert module.selection_map["tir.projection"].candidate_id == IDENTITY
    restored = fm.load_module(fm.emit_module(module, tmp_path / "module.py"))
    assert restored.semantic_hash == module.semantic_hash
    write_artifact(restored, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


@pytest.mark.parametrize("splits", [0, -1, 3, True, 4])
def test_invalid_accumulation_partition_is_rejected(tmp_path, splits):
    module = _compile(fm.Placement((1, 1), "yx", "bb"), splits=splits)
    with pytest.raises(CodegenError, match="splits require"):
        write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


@pytest.mark.parametrize("layout", [False, "automatic", 1])
def test_invalid_weight_load_layout_is_rejected(tmp_path, layout):
    module = _compile(fm.Placement((1, 1), "yx", "bb"), weight_load_layout=layout)
    with pytest.raises(CodegenError, match="weight_load_layout"):
        write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


@pytest.mark.parametrize("k,n,m,splits,axes", [
    (80, 24, 2, 1, ()),
    (256, 24, 2, 1, ()),
    (4096, 128, 1, 2, (1,)),
    (4096, 2048, 1, 2, (0, 1)),
])
def test_device_accumulator_preserves_precision_and_guards(tmp_path, k, n, m, splits, axes):
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    mesh = fm.Placement((8, 16), "yx", "bb")
    module = _compile(mesh, k=k, n=n, m=m, splits=splits, axes=axes)
    path = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(path, device="cuda:0")
    try:
        for mutation in range(3):
            x = (((torch.arange(m * k, device="cuda") % 13) - 6 + mutation) / 128).reshape(m, k).bfloat16()
            w = (((torch.arange(k * n, device="cuda") % 17) - 8) / 128).reshape(k, n).bfloat16()
            if mutation == 2:
                x.zero_(); w.zero_()
                x[:, 0] = 129 / 128; x[:, 1] = 1.
                w[0].fill_(129 / 128); w[1].fill_(-65 / 64)
            packed = w.reshape(k // 16, 2, 8, n // 8, 8).permute(0, 3, 4, 1, 2).contiguous()
            output_storage = torch.full((m * n + 32,), 123., device="cuda")
            copied_storage = torch.full((160,), 123., device="cuda")
            specs = [s for s in runtime.external_arguments if s["role"] == "result"]
            kind = fm.logical_type(module.node_map[specs[0]["value"]].type)
            output_shape = (*[d.fixed_value for d in kind.shape], *getattr(kind.dtype, "lanes", ()))
            values = {"x": x, "w": packed, "sentinel": torch.arange(128, device="cuda", dtype=torch.float32)}
            values.update({specs[0]["value"]: output_storage[:m * n].reshape(output_shape),
                           specs[1]["value"]: copied_storage[:128]})
            arguments = tuple(values[s["value"]] for s in runtime.external_arguments)
            GeneratedTirCallGraphModule.prepare(runtime, *arguments)
            GeneratedTirCallGraphModule.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            # Dyadic operands keep every partial sum exactly representable;
            # this independently catches a hidden BF16 result conversion.
            expected = (x.cpu().double() @ w.cpu().double()).float().cuda()
            torch.testing.assert_close(output_storage[:m * n].reshape(m, n), expected, rtol=0, atol=0)
            torch.testing.assert_close(copied_storage[:128], values["sentinel"] * 2, rtol=0, atol=0)
            assert torch.all(output_storage[m * n:] == 123.) and torch.all(copied_storage[128:] == 123.)
            assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


def test_coalesced_weight_load_reduces_pointer_liveness(tmp_path):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    k, n = 4096, 2048
    mesh = fm.Placement((8, 16), "yx", "bb")
    generator = torch.Generator(device="cuda").manual_seed(19217)
    x = torch.randn((1, k), generator=generator, device="cuda").bfloat16()
    w = torch.randn((k // 16, n // 8, 8, 2, 8), generator=generator, device="cuda").bfloat16()
    runtimes, reports, results = [], [], []
    try:
        for layout in ("mma", "coalesced"):
            module = _compile(mesh, k=k, n=n, splits=2, axes=(0, 1), weight_load_layout=layout)
            runtime = load(write_artifact(module, tmp_path / layout, target="nvidia-sm90", emit_executable=True),
                           device="cuda:0")
            runtimes.append(runtime)
            specs = [s for s in runtime.external_arguments if s["role"] == "result"]
            kind = fm.logical_type(module.node_map[specs[0]["value"]].type)
            shape = (*[d.fixed_value for d in kind.shape], *kind.dtype.lanes)
            output, copied = torch.full((n + 32,), 123., device="cuda"), torch.full((160,), 123., device="cuda")
            values = {"x": x, "w": w, "sentinel": torch.arange(128, dtype=torch.float32, device="cuda"),
                      specs[0]["value"]: output[:n].reshape(shape), specs[1]["value"]: copied[:128]}
            arguments = tuple(values[s["value"]] for s in runtime.external_arguments)
            GeneratedTirCallGraphModule.prepare(runtime, *arguments)
            GeneratedTirCallGraphModule.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            assert torch.all(output[n:] == 123.) and torch.all(copied[128:] == 123.)
            assert runtime.resource_report["spill_bytes"] == 0
            results.append(output[:n])
            reports.append(runtime.resource_report)
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
        # With N16 and eight warps, an MMA operand layout repeats addresses
        # in four warp groups. Global loads need independent K/N ownership;
        # only the conversion to the MMA fragment should replicate values.
        assert reports[1]["registers_per_thread"] < reports[0]["registers_per_thread"]
    finally:
        for runtime in runtimes:
            runtime.close()
