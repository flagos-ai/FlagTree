# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One selected TIR graph may require distinct physical launch programs."""

from dataclasses import replace
import ast
import importlib.util

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.codegen.triton.package_plan import plain_package_value
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError, CodegenError
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target


def graph(*, export_mid=True, packed=False):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 128)), id="x")
            if packed:
                dtype = fm.vector_type("bfloat16", (8, 2, 8))
                a = self.input("a", fm.tensor_type(dtype, (8, 8)), id="a")
                b = self.input("b", fm.tensor_type(dtype, (8, 4)), id="b")
                none = fm.F.builtin.none()
                metadata = {
                    "selected_vectorization": "vectorization.matmul.n",
                    "selected_vector_axes": (1,),
                    "selected_vector_lanes": (8,),
                }
                mid = fm.F.ntt.packed_matmul(
                    x, a, none, none, output_data_type="bfloat16", name="first", metadata=metadata
                )
                out = fm.F.ntt.packed_matmul(
                    x, b, none, none, output_data_type="bfloat16", name="second", metadata=metadata
                )
            else:
                a = self.input("a", fm.tensor_type("bfloat16", (64, 128)), id="a")
                b = self.input("b", fm.tensor_type("bfloat16", (32, 64)), id="b")
                mid = fm.F.math.matmul(x, a, transpose_b=True, name="first")
                out = fm.F.math.matmul(mid, b, transpose_b=True, name="second")
            self.function("main", (x, a, b), (mid, out) if export_mid else (out,))

    return Graph(dialect="nn", stage="frozen_constants", entry="main").build()


def compile_programs(*, export_mid=True, packed=False, dependencies=None):
    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            module = super().propose_tir(module)
            choices = [
                (point.id, "tir.dense_matmul.packed_k_major_mma")
                for point in module.selection_points
                if point.kind == "tir"
                and point.owner in module.node_map
                and module.node_map[point.owner].op == "ntt.packed_matmul"
            ]
            module = (
                apply_plan(
                    module,
                    override_plan(
                        module,
                        choices,
                        policy="test.kernel-programs",
                        rationale="Exercise separate warp layout namespaces on the same selected MMA family.",
                    ),
                )
                if choices
                else module
            )
            owners = {p.owner for p in module.selection_points if p.kind == "tir"}
            calls = tuple(node.id for node in module.nodes if node.id in owners)
            assert len(calls) == 2, calls
            programs = (
                dict(symbol="program_first", calls=calls[:1], num_warps=8),
                dict(symbol="program_second", calls=calls[1:], num_warps=4),
            )
            if dependencies is not None:
                programs = tuple({**program, "depends_on": parents}
                                 for program, parents in zip(programs, dependencies, strict=True))
            return replace(
                module,
                metadata={
                    **module.metadata,
                    "kernel_program_partition": programs,
                },
            )

    compiler = Compiler()
    compiler.target = Target()
    return compiler.compile(graph(export_mid=export_mid, packed=packed)).module


@pytest.fixture(scope="module")
def compiled():
    return compile_programs()


def import_source(path):
    spec = importlib.util.spec_from_file_location("_flagmega_program_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normal_codegen_outlines_two_entries_with_independent_warp_layouts(tmp_path, compiled):
    compiled = compile_programs(packed=True)
    assert all(d.dispatch.microkernel.variant == "packed_k_major_mma" for d in compiled.kernel_definitions)
    package = render_triton_package(compiled, tmp_path)
    assert len(package.get("kernels", ())) == 2
    assert [v["symbol"] for v in package["kernels"]] == ["program_first", "program_second"]
    assert [v["num_warps"] for v in package["kernels"]] == [8, 4]
    assert all(not v["requires_grid_wide_sync"] for v in package["kernels"])
    source = import_source(tmp_path / "generated_kernels.py")
    assert source.program_first is not source.program_second
    assert tuple(source.FLAGMEGA_KERNEL_PROGRAMS) == tuple(package["kernels"])
    reloaded = fm.load_module(fm.emit_module(compiled, tmp_path / "roundtrip.py"))
    assert reloaded.semantic_hash == compiled.semantic_hash
    assert describe_tir_package(reloaded)["kernels"] == package["kernels"]


@pytest.mark.parametrize(
    "cause",
    ["missing", "duplicate", "reordered", "unknown", "warps", "below_minimum", "symbol", "reserved", "physical_grid"],
)
def test_rejects_invalid_program_partition(compiled, cause):
    if cause == "below_minimum":
        compiled = compile_programs(packed=True)
    plan = plain_package_value(compiled.metadata["codegen_package_plan"])
    # Checkpoints freeze mappings recursively; reconstruct only the mutable
    # target-owned requests this test intentionally corrupts.
    programs = [dict(p) for p in plan["programs"]]
    if cause == "missing":
        programs.pop()
    elif cause == "duplicate":
        programs[1]["calls"] = programs[0]["calls"]
    elif cause == "reordered":
        programs.reverse()
    elif cause == "unknown":
        programs[0]["calls"] = ("not_a_call",)
    elif cause == "warps":
        programs[0]["num_warps"] = 3
    elif cause == "below_minimum":
        programs[0]["num_warps"] = 1
    elif cause == "symbol":
        programs[1]["symbol"] = programs[0]["symbol"]
    elif cause == "reserved":
        programs[0]["symbol"] = "FLAGMEGA_KERNEL_PROGRAMS"
    else:
        programs[0]["grid"] = (512, 1, 1)
    plan["programs"] = programs
    invalid = replace(
        compiled, metadata={**compiled.metadata, "codegen_package_plan": plan, "kernel_program_partition": programs}
    )
    with pytest.raises(CodegenError, match="[Pp]rogram"):
        describe_tir_package(invalid)


def test_runtime_prepares_each_entry_with_its_declared_resource_contract(tmp_path, compiled, monkeypatch):
    import triton.flagmega.runtime.module as runtime_module

    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact)
    seen = []

    class Prepared:
        resource_report = {}

    def prepare(kernel, args, dynamic, **options):
        seen.append(options)
        return Prepared()

    monkeypatch.setattr(runtime_module, "prepare_jit_kernel", prepare)
    monkeypatch.setattr(runtime, "_validate_external_arguments", lambda args: None)
    runtime._pool_values = {str(v["name"]): object() for v in runtime.runtime_binding["pools"]}
    runtime._mark_loaded("cpu")
    Runtime.prepare(runtime, *(object() for _ in runtime.external_arguments))
    assert [v["num_warps"] for v in seen] == [8, 4]
    assert [v["contract"].compute_num_warps for v in seen] == [8, 4]
    assert all(v["contract"].forbid_spills for v in seen)


def test_runtime_rejects_manifest_source_warp_mismatch(tmp_path, compiled):
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact)
    runtime.codegen["kernels"][0]["num_warps"] = 4
    with pytest.raises(ArtifactError, match="[Pp]rogram"):
        runtime._kernel_launch_specs()


@pytest.mark.parametrize("export_mid", [True, False])
def test_device_serial_cut_preserves_intermediates_rebinding_and_graph(tmp_path, compiled, export_mid):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    if not export_mid:
        compiled = compile_programs(export_mid=False)
    runtime = load(
        write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True), device="cuda:0"
    )
    generator = torch.Generator(device="cuda").manual_seed(907)
    values = {
        "x": torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        "a": torch.empty((64, 128), device="cuda", dtype=torch.bfloat16),
        "b": torch.empty((32, 64), device="cuda", dtype=torch.bfloat16),
    }
    guards = []
    outputs = []
    for shape in ((1, 64), (1, 32)) if export_mid else ((1, 32),):
        storage = torch.full((shape[1] + 32,), -123.0, device="cuda", dtype=torch.bfloat16)
        guards.append(storage)
        outputs.append(storage[16:-16].reshape(shape))

    def arguments():
        results = iter(outputs)
        return tuple(next(results) if s["role"] == "result" else values[s["value"]] for s in runtime.external_arguments)

    try:
        Runtime.prepare(runtime, *arguments())
        assert [v["num_warps"] for v in runtime.resource_report] == [8, 4]
        assert all(v["spill_bytes"] == 0 for v in runtime.resource_report)
        for mutation in range(6):
            for name, v in values.items():
                v.copy_((torch.randint(-8, 9, v.shape, generator=generator, device="cuda").float() / 16).bfloat16())
            if mutation == 5:
                values["x"].zero_()
            for value in outputs:
                value.fill_(float("nan"))
            Runtime.run_into(runtime, *arguments())
            torch.cuda.synchronize()
            mid = (values["x"].double() @ values["a"].double().T).bfloat16()
            gold = (mid.double() @ values["b"].double().T).bfloat16()
            assert torch.equal(outputs[-1], gold)
            if export_mid:
                assert torch.equal(outputs[0], mid)
            assert all(bool((v[:16] == -123).all() & (v[-16:] == -123).all()) for v in guards)
            if mutation == 2:
                values = {k: v.clone() for k, v in values.items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture := torch.cuda.CUDAGraph(), stream=stream):
            Runtime.run_into(runtime, *arguments())
        for _ in range(3):
            capture.replay()
        torch.cuda.synchronize()
        assert torch.equal(outputs[-1], gold)
    finally:
        runtime.close()


def compile_loop_program(stop):
    from python.test.flagmega.ir.ops.test_for import loop_module
    from triton.flagmega.passes.functions import function_nodes

    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            module = super().propose_tir(module)
            owners = {p.owner for p in module.selection_points if p.kind == "tir"}
            items = tuple(
                f"{module.entry}.{n.id}_iv" if n.op == "control.for" else n.id
                for n in function_nodes(module, module.function_map[module.entry])
                if n.id in owners or n.op in {"distributed.boxing", "control.for"}
            )
            return replace(
                module,
                metadata={
                    **module.metadata,
                    "kernel_program_partition": ({"symbol": "loop_program", "calls": items, "num_warps": 4},),
                },
            )

    compiler = Compiler()
    compiler.target = Target()
    return compiler.compile(loop_module(stop=stop)).module


@pytest.mark.parametrize("stop", [0, 1, 3])
def test_program_keeps_a_whole_structured_loop_and_reusable_body(tmp_path, stop):
    compiled = compile_loop_program(stop)
    package = render_triton_package(compiled, tmp_path)
    assert len(package["kernels"]) == 1
    source = import_source(tmp_path / "generated_kernels.py")
    assert source.loop_program
    tree = ast.parse((tmp_path / "generated_kernels.py").read_text())
    assert (
        len(
            [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "loop_iv"
            ]
        )
        == 1
    )
    restored = fm.load_module(fm.emit_module(compiled, tmp_path / "loop.py"))
    assert restored.semantic_hash == compiled.semantic_hash
    runtime = load(write_artifact(restored, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True))
    assert runtime._kernel_launch_specs()[0]["calls"][-1] == "main.loop_iv"


@pytest.mark.parametrize("stop", [0, 1, 3])
def test_device_program_loop_preserves_zero_and_multiple_trips(tmp_path, stop):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    runtime = load(
        write_artifact(compile_loop_program(stop), tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
        device="cuda:0",
    )
    value = torch.arange(4, dtype=torch.float32, device="cuda")
    increment = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    output = torch.full_like(value, float("nan"))
    values = {"value": value, "increment": increment}
    arguments = tuple(
        output if spec["role"] == "result" else values[spec["value"]] for spec in runtime.external_arguments
    )
    try:
        Runtime.prepare(runtime, *arguments)
        assert runtime.resource_report["spill_bytes"] == 0
        for mutation in range(3):
            value.add_(mutation)
            output.fill_(float("nan"))
            Runtime.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, value + stop * increment, rtol=0, atol=0)
    finally:
        runtime.close()


def test_device_programs_keep_packed_mma_layouts_private(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    compiled = compile_programs(packed=True)
    assert all(d.dispatch.microkernel.variant == "packed_k_major_mma" for d in compiled.kernel_definitions)
    runtime = load(
        write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True), device="cuda:0"
    )
    generator = torch.Generator(device="cuda").manual_seed(712)
    values = {
        "x": torch.empty((1, 128), dtype=torch.bfloat16, device="cuda"),
        "a": torch.empty((8, 8, 8, 2, 8), dtype=torch.bfloat16, device="cuda"),
        "b": torch.empty((8, 4, 8, 2, 8), dtype=torch.bfloat16, device="cuda"),
    }
    stores = [torch.full((n + 32,), -123.0, dtype=torch.bfloat16, device="cuda") for n in (64, 32)]
    outputs = [v[16:-16].reshape(1, -1, 8) for v in stores]

    def args():
        results = iter(outputs)
        return tuple(
            next(results) if spec["role"] == "result" else values[spec["value"]] for spec in runtime.external_arguments
        )

    try:
        Runtime.prepare(runtime, *args())
        assert [v["num_warps"] for v in runtime.resource_report] == [8, 4]
        assert all(v["spill_bytes"] == 0 for v in runtime.resource_report)
        for mutation in range(6):
            for v in values.values():
                v.copy_((torch.randint(-8, 9, v.shape, generator=generator, device="cuda").float() / 16).bfloat16())
            if mutation == 5:
                values["x"].zero_()
            Runtime.run_into(runtime, *args())
            torch.cuda.synchronize()
            for name, result in zip(("a", "b"), outputs, strict=True):
                weight = values[name].permute(0, 3, 4, 1, 2).reshape(128, -1)
                gold = (values["x"].double() @ weight.double()).bfloat16()
                assert torch.equal(result.reshape(1, -1), gold)
            assert all(bool((v[:16] == -123).all() & (v[-16:] == -123).all()) for v in stores)
    finally:
        runtime.close()
