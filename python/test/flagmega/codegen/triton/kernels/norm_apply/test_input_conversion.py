# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Numerator conversion stays inside the actual NormApply device kernel."""

import subprocess
from math import prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRVerificationError


def norm_graph(*, explicit, lanes=(), use_mean=False, round_before_scale=False, output_dtype="bfloat16", family="local",
               source_dtype="float32", virtual_dtype="bfloat16"):
    mesh = fm.Placement((8, 16), "yx", "bb")
    broad = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous((1,) if family == "add" else (0, 1),
                                   (256 if family == "add" else 32) // prod(lanes))
    value_type = fm.tensor_type(fm.vector_type(source_dtype, lanes) if lanes else source_dtype, (1, 4096 // prod(lanes)))
    parameter_type = fm.tensor_type(fm.vector_type("bfloat16", lanes) if lanes else "bfloat16", (4096 // prod(lanes),))
    stats_type = fm.tensor_type("float32", (2 if use_mean else 1, 1, 1))

    class Graph(fm.Module):
        def forward(self):
            roots = tuple(self.input(name, kind, id=name) for name, kind in
                          (("x", value_type), ("stats", stats_type), ("scale", parameter_type), ("bias", parameter_type)))
            values = tuple(fm.F.distributed.sharded_view(value, fm.DistributedType(value.type, policy, mesh))
                           for value, policy in zip(roots, ((broad, split), (broad, broad, broad), (split,), (split,)), strict=True))
            x, stats, scale, bias = values
            material = None
            if family == "add":
                parts_type = fm.tensor_type("float32", (8, 1, 4096))
                parts = self.input("parts", parts_type, id="parts")
                roots = (*roots, parts)
                parts = fm.F.distributed.sharded_view(parts, fm.DistributedType(parts_type,
                    (fm.SBP.split_contiguous((0,), 1), broad, fm.SBP.split_contiguous((1,), 256)), mesh))
                partial = fm.F.math.reduce_sum(parts, axes=(0,), keep_dims=False)
                if lanes:
                    partial = fm.F.tensors.pack(partial, axes=(-1,), lanes=lanes)
                combined = fm.F.ntt.add_norm_stats(partial, x, axis=-1, use_mean=use_mean)
                x, stats = fm.F.tensors.get_items(combined, 0, 1)
                material = x
            elif family == "stats":
                stats = fm.F.nn.norm_stats(x, axis=-1, use_mean=use_mean)
            if family != "local":
                stats = fm.F.distributed.boxing(stats, fm.DistributedType(stats_type, (broad, broad, broad), mesh))
            if explicit:
                x = fm.F.tensors.cast(x, fm.vector_type(virtual_dtype, lanes) if lanes else virtual_dtype, name="converted")
            norm = fm.F.nn.norm_apply(x, stats, scale, bias, axis=-1, epsilon=1e-6, use_mean=use_mean,
                round_before_scale=round_before_scale, output_dtype=output_dtype,
                input_dtype=None if explicit else virtual_dtype, name="norm")
            output = fm.F.distributed.force_boxing(norm, fm.logical_type(norm.type), name="result")
            outputs = ((fm.F.distributed.force_boxing(material, fm.logical_type(material.type), name="material"), output)
                       if material is not None else (output,))
            self.function("main", roots, outputs)
    return Graph(dialect="distributed", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()


@pytest.mark.parametrize("lanes", [(), (8,)])
@pytest.mark.parametrize("use_mean,round_before_scale", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("family", ["local", "stats", "add"])
def test_input_rounding_matches_explicit_gpu_conversion(tmp_path, lanes, use_mean, round_before_scale, family):
    _compare_input_conversion(tmp_path, lanes, use_mean, round_before_scale, family)


@pytest.mark.parametrize("lanes", [(), (8,)])
@pytest.mark.parametrize("use_mean,round_before_scale", [(False, False), (False, True), (True, False), (True, True)])
def test_input_widening_matches_explicit_gpu_conversion(tmp_path, lanes, use_mean, round_before_scale):
    _compare_input_conversion(tmp_path, lanes, use_mean, round_before_scale, "local",
                              source_dtype="bfloat16", virtual_dtype="float32")


def _compare_input_conversion(tmp_path, lanes, use_mean, round_before_scale, family,
                              *, source_dtype="float32", virtual_dtype="bfloat16"):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    context = torch.empty((1,), device="cuda")
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    active = [line for line in processes.splitlines() if line.split(",")[0].strip() == uuid]
    assert len(active) == 1, f"Cooperative kernels require an exclusive GPU: {active}"
    del context
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

    output_dtype = "float32" if round_before_scale or virtual_dtype == "float32" else "bfloat16"
    runtimes, outputs, arguments, guards = [], [], [], []
    rng = torch.Generator().manual_seed(1729)
    source = torch.randn((1, 4096), generator=rng) * 3
    square = source.square().sum(-1, keepdim=True)
    stats = torch.stack((source.sum(-1, keepdim=True), square)) if use_mean else square.unsqueeze(0)
    values = {"x": source.to(getattr(torch, source_dtype)).reshape(1, 4096 // prod(lanes), *lanes).cuda(), "stats": stats.cuda(),
              "scale": torch.randn((4096,), generator=rng).bfloat16().reshape(4096 // prod(lanes), *lanes).cuda(),
              "bias": torch.randn((4096,), generator=rng).bfloat16().reshape(4096 // prod(lanes), *lanes).cuda()}
    if family == "add":
        values["parts"] = torch.randn((8, 1, 4096), generator=rng).cuda()
    try:
        for label, explicit in (("reference", True), ("fused", False)):
            try:
                module = Compiler().compile(norm_graph(explicit=explicit, lanes=lanes, use_mean=use_mean,
                    round_before_scale=round_before_scale, output_dtype=output_dtype, family=family,
                    source_dtype=source_dtype, virtual_dtype=virtual_dtype)).module
            except IRVerificationError as error:
                if (explicit and family == "add" and use_mean
                        and "No reviewed Triton TIR candidate was proposed for op 'ntt.add_norm_stats'" in str(error)):
                    pytest.xfail("Existing SM90 add-norm-stats reference supports RMS only; LayerNorm device comparison is unavailable")
                raise
            runtime = load(write_artifact(module, tmp_path / label, target="nvidia-sm90", emit_executable=True), device="cuda:0")
            runtimes.append(runtime)
            results = {}
            result_specs = [spec for spec in runtime.external_arguments if spec["role"] == "result"]
            for spec in result_specs:
                dtype = torch.float32 if spec["value"] == "material" else getattr(torch, output_dtype)
                storage = torch.full((4096 + 64,), -123., device="cuda", dtype=dtype)
                results[spec["value"]] = storage[32:-32].reshape(1, 4096 // prod(lanes), *lanes)
                guards.extend((storage[:32], storage[-32:]))
            outputs.append(tuple(results.values()))
            operands = {**values, **results}
            arguments.append(tuple(operands[spec["value"]] for spec in runtime.external_arguments))
            Runtime.prepare(runtime, *arguments[-1])
            assert runtime.resource_report["spill_bytes"] == 0
            if not explicit:
                assert not any(d.dispatch.semantic_op in {"tensors.cast", "ntt.vectorized_cast"} for d in module.kernel_definitions)
                expected_family = {"local": "norm_apply", "stats": "gather_reduce_norm_apply",
                                   "add": "gather_reduce_add_norm_apply"}[family]
                assert any(d.dispatch.microkernel.family == expected_family for d in module.kernel_definitions)
        for mutation in range(3):
            if mutation:
                values["x"].mul_(1.03125).add_(1 / 128)
            for runtime, operands, results in zip(runtimes, arguments, outputs, strict=True):
                for output in results:
                    output.fill_(float("nan"))
                Runtime.run_into(runtime, *operands)
            torch.cuda.synchronize()
            for reference, actual in zip(*outputs, strict=True):
                assert torch.isfinite(actual).all()
                assert torch.equal(reference.view(torch.uint8), actual.view(torch.uint8))
            assert all(bool((guard == -123).all()) for guard in guards)
    finally:
        for runtime in runtimes:
            runtime.close()
