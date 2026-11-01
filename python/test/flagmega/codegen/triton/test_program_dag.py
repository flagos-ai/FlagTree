# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dependency, physical lifetime and device acceptance for program fork/join."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError, CodegenError, IRVerificationError, RuntimeContractError
from triton.flagmega.ir import verify_buffer_plan
from triton.flagmega.ir.program_order import program_ancestors, program_stream_indices
from triton.flagmega.runtime import load
from triton.flagmega.runtime.kernel_programs import verified_kernel_programs
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
from triton.flagmega.runtime.prepared import ResourceContract
from triton.flagmega.runtime.program_dag import verify_joint_resources
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.codegen.triton.test_kernel_programs import compile_programs, import_source


def compile_fork_join(*, common_prefix=False):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 128)), id="x")
            bias = self.input("bias", fm.tensor_type("bfloat16", (1, 128)), id="bias") if common_prefix else None
            a = self.input("a", fm.tensor_type("bfloat16", (64, 128)), id="a")
            b = self.input("b", fm.tensor_type("bfloat16", (32, 64)), id="b")
            c = self.input("c", fm.tensor_type("bfloat16", (64, 128)), id="c")
            d = self.input("d", fm.tensor_type("bfloat16", (32, 64)), id="d")
            value = fm.F.math.add(x, bias, name="prefix") if common_prefix else x
            left0 = fm.F.math.matmul(value, a, transpose_b=True, name="left0")
            left1 = fm.F.math.matmul(left0, b, transpose_b=True, name="left1")
            right0 = fm.F.math.matmul(value, c, transpose_b=True, name="right0")
            right1 = fm.F.math.matmul(right0, d, transpose_b=True, name="right1")
            joined = fm.F.math.add(left1, right1, name="joined")
            self.function("main", (x, bias, a, b, c, d) if common_prefix else (x, a, b, c, d),
                          (left1, right1, joined))

    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            module = super().propose_tir(module)
            owners = {p.owner for p in module.selection_points if p.kind == "tir"}
            calls = tuple(node.id for node in module.nodes if node.id in owners)
            expected = ("left0", "left1", "right0", "right1", "joined")
            assert calls == (("prefix", *expected) if common_prefix else expected), calls
            parent = ("prefix",) if common_prefix else ()
            dependencies = {
                "prefix": (),
                "left0": parent,
                "left1": ("left0",),
                "right0": parent,
                "right1": ("right0",),
                "joined": ("left1", "right1"),
            }
            return replace(
                module,
                metadata={
                    **module.metadata,
                    "kernel_program_partition": tuple(
                        {
                            "symbol": f"program_{name}",
                            "calls": (name,),
                            "num_warps": 4,
                            "depends_on": tuple(f"program_{p}" for p in dependencies[name]),
                        }
                        for name in calls
                    ),
                },
            )

    compiler = Compiler()
    compiler.target = Target()
    return compiler.compile(Graph(dialect="nn", stage="frozen_constants", entry="main").build()).module


def test_fork_join_replans_storage_of_serially_dead_branch_temporaries(tmp_path):
    module = compile_fork_join()
    plan = verify_buffer_plan(module)
    values = dict(plan.function_map[module.entry].values)
    left, right = (plan.buffer_map[values[name][0]].mem_span.buffer for name in ("left0", "right0"))
    assert left.id != right.id
    assert left.offset + left.nbytes <= right.offset or right.offset + right.nbytes <= left.offset
    kernels = render_triton_package(module, tmp_path)["kernels"]
    assert [p["stream_index"] for p in kernels] == [0, 0, 1, 1, 0]
    assert kernels[-1]["depends_on"] == ["program_left1", "program_right1"]


def test_fork_after_materialized_prefix_is_preserved_by_normal_codegen(tmp_path):
    kernels = render_triton_package(compile_fork_join(common_prefix=True), tmp_path)["kernels"]
    by_symbol = {spec["symbol"]: spec for spec in kernels}
    assert by_symbol["program_left0"]["depends_on"] == ["program_prefix"]
    assert by_symbol["program_right0"]["depends_on"] == ["program_prefix"]
    assert by_symbol["program_left0"]["stream_index"] != by_symbol["program_right0"]["stream_index"]


@pytest.mark.parametrize(
    "dependencies",
    [((), ("first",)), (("second",), ()), ((), ("missing",)), ((), ("second",)), ((), ("first", "first"))],
)
def test_program_dependency_parser_rejects_missing_forward_self_and_duplicate_edges(dependencies):
    values = [{"symbol": "first", "depends_on": dependencies[0]}, {"symbol": "second", "depends_on": dependencies[1]}]
    if dependencies == ((), ("first",)):
        assert program_ancestors(values)["second"] == {"first"}
        assert program_stream_indices(values) == {"first": 0, "second": 0}
    else:
        with pytest.raises(IRVerificationError, match="dependencies"):
            program_ancestors(values)


def test_program_dependency_parser_requires_explicit_roots():
    with pytest.raises(IRVerificationError, match="dependencies"):
        program_ancestors([{"symbol": "first"}, {"symbol": "second", "depends_on": ()}])


def test_independent_programs_get_distinct_streams_and_declared_writes(tmp_path):
    module = compile_programs(packed=True, dependencies=((), ()))
    codegen = render_triton_package(module, tmp_path)
    assert codegen["kernel_program_schema"] == "flagmega.kernel-programs/v3-dag-completion"
    assert all(not p["device_completion"] for p in codegen["kernels"])
    assert [p["stream_index"] for p in codegen["kernels"]] == [0, 1]
    assert [p["depends_on"] for p in codegen["kernels"]] == [[], []]
    arguments = codegen["runtime_binding"]["arguments"]
    writes = [[arguments[i]["value"] for i in p["external_writes"]] for p in codegen["kernels"]]
    assert writes == [["first"], ["second"]]


def test_program_dag_round_trip_reconstructs_effects_and_source_contract(tmp_path):
    module = compile_programs(packed=True, dependencies=((), ()))
    reloaded = fm.load_module(fm.emit_module(module, tmp_path / "selected.py"))
    assert reloaded.semantic_hash == module.semantic_hash
    package = render_triton_package(reloaded, tmp_path / "source")
    source = import_source(tmp_path / "source/generated_kernels.py")
    kernels = (source.program_first, source.program_second)
    assert len(verified_kernel_programs(package, kernels, reloaded)) == 2
    forged = deepcopy(package)
    forged["kernels"][0]["external_writes"] = []
    # Matching source/manifest literals do not waive the typed effect proof.
    source.FLAGMEGA_KERNEL_PROGRAMS = forged["kernels"]
    with pytest.raises(ArtifactError, match="dependencies differ"):
        verified_kernel_programs(forged, kernels, reloaded)


def test_missing_data_dependency_is_rejected_from_real_physical_effects(tmp_path):
    module = compile_programs(dependencies=((), ()))
    with pytest.raises(CodegenError, match="unordered physical memory hazard"):
        render_triton_package(module, tmp_path)


def test_dag_entry_allocation_ends_after_its_last_ordered_reader(tmp_path):
    module = compile_programs(export_mid=False, dependencies=((), ("program_first",)))
    plan = verify_buffer_plan(module)
    first = plan.buffer_map[dict(plan.function_map[module.entry].values)["first"][0]]
    from triton.flagmega.passes.functions.graph import function_nodes

    indexes = {n.id: i for i, n in enumerate(function_nodes(module, module.function_map[module.entry]))}
    assert first.mem_span.buffer.live_start == indexes["first"]
    assert first.mem_span.buffer.live_end == indexes["second"]
    assert render_triton_package(module, tmp_path)["kernels"][1]["stream_index"] == 0


@pytest.mark.parametrize("registers,shared,passes", [(100, 100000, True), (130, 100000, False), (100, 120000, False)])
def test_joint_residency_uses_rounded_stream_peaks(registers, shared, passes):
    contract = ResourceContract(8, 1)
    specs = [{"symbol": str(i), "depends_on": (), "grid": (128, 1, 1),
              "requires_grid_wide_sync": True, "stream_index": i} for i in range(2)]
    kernels = [
        SimpleNamespace(
            contract=contract,
            resource_report={"registers_per_thread": registers, "shared_memory_bytes": shared, "num_warps": 8},
        )
        for _ in specs
    ]
    if passes:
        assert verify_joint_resources(specs, kernels, 132)["registers"] == 2 * 104 * 256
    else:
        with pytest.raises(RuntimeContractError, match="joint SM90 residency"):
            verify_joint_resources(specs, kernels, 132)


def test_joint_residency_rejects_unproved_subpartition_packing():
    contract = ResourceContract(2, 1)
    specs = [{"symbol": "first", "depends_on": (), "grid": (128, 1, 1),
              "requires_grid_wide_sync": True, "stream_index": 0}]
    kernel = SimpleNamespace(
        contract=contract, resource_report={"registers_per_thread": 32, "shared_memory_bytes": 0, "num_warps": 2}
    )
    with pytest.raises(RuntimeContractError, match="complete four-warp groups"):
        verify_joint_resources(specs, (kernel,), 132)


def test_device_program_dag_matches_independent_fp64_and_captures(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    module = compile_programs(packed=True, dependencies=((), ()))
    runtime = load(
        write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True), device="cuda:0"
    )
    generator = torch.Generator(device="cuda").manual_seed(6123)
    x = torch.empty((1, 128), dtype=torch.bfloat16, device="cuda")
    a = torch.empty((8, 8, 8, 2, 8), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((8, 4, 8, 2, 8), dtype=torch.bfloat16, device="cuda")
    storages = [
        torch.full((64 + 32,), 128.0, dtype=torch.bfloat16, device="cuda"),
        torch.full((32 + 32,), 128.0, dtype=torch.bfloat16, device="cuda"),
    ]
    values = {
        "x": x,
        "a": a,
        "b": b,
        "first": storages[0][16:-16].reshape(1, 64),
        "second": storages[1][16:-16].reshape(1, 32),
    }
    actuals = tuple(values[s["value"]] for s in runtime.external_arguments)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            Runtime.prepare(runtime, *actuals)
            assert len(runtime._program_dag.streams) == 2
            assert all(p.resource_report["spill_bytes"] == 0 for p in runtime._prepared)
            for mutation in range(6):
                for value in (x, a, b):
                    value.copy_(torch.randint(-8, 9, value.shape, device="cuda", generator=generator).float() / 8)
                for value in (values["first"], values["second"]):
                    value.fill_(float("nan"))
                Runtime.run_into(runtime, *actuals)
                stream.synchronize()
                for name, weight in (("first", a), ("second", b)):
                    # K-major Vec(8,2,8) maps [K/16,N/8,Nlane,Kpack,Klane].
                    dense = weight.permute(1, 2, 0, 3, 4).reshape(weight.shape[1] * 8, 128)
                    gold = (x.double() @ dense.double().T).bfloat16()
                    torch.testing.assert_close(values[name], gold, rtol=0, atol=0)
                assert all(bool((v[:16] == 128).all() & (v[-16:] == 128).all()) for v in storages)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                Runtime.run_into(runtime, *actuals)
                Runtime.run_into(runtime, *actuals)
            graph.replay()
            stream.synchronize()
            for name, weight in (("first", a), ("second", b)):
                dense = weight.permute(1, 2, 0, 3, 4).reshape(weight.shape[1] * 8, 128)
                torch.testing.assert_close(values[name], (x.double() @ dense.double().T).bfloat16(), rtol=0, atol=0)
            aliases = list(actuals)
            first = next(i for i, s in enumerate(runtime.external_arguments) if s["value"] == "first")
            second = next(i for i, s in enumerate(runtime.external_arguments) if s["value"] == "second")
            aliases[second] = actuals[first].reshape(-1)[:32].reshape(1, 32)
            with pytest.raises(RuntimeContractError, match="writable external aliases"):
                Runtime.run_into(runtime, *aliases)
    finally:
        torch.cuda.synchronize()
        runtime.close()


def test_device_program_dag_fork_join_orders_repeated_captured_requests(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    runtime = load(
        write_artifact(compile_fork_join(), tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
        device="cuda:0",
    )
    generator = torch.Generator(device="cuda").manual_seed(1527)
    values = {
        name: (torch.randint(-8, 9, shape, device="cuda", generator=generator).float() / 8).bfloat16()
        for name, shape in {"x": (1, 128), "a": (64, 128), "b": (32, 64), "c": (64, 128), "d": (32, 64)}.items()
    }
    for name in ("left1", "right1", "joined"):
        values[name] = torch.full((1, 32), float("nan"), dtype=torch.bfloat16, device="cuda")
    actuals = tuple(values[s["value"]] for s in runtime.external_arguments)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            Runtime.prepare(runtime, *actuals)
            assert all(p.resource_report["spill_bytes"] == 0 for p in runtime._prepared)
            for _ in range(4):
                values["x"].copy_(
                    (torch.randint(-8, 9, (1, 128), device="cuda", generator=generator).float() / 8).bfloat16()
                )
                Runtime.run_into(runtime, *actuals)
                stream.synchronize()
                left0 = (values["x"].double() @ values["a"].double().T).bfloat16()
                right0 = (values["x"].double() @ values["c"].double().T).bfloat16()
                left1 = (left0.double() @ values["b"].double().T).bfloat16()
                right1 = (right0.double() @ values["d"].double().T).bfloat16()
                for name, gold in (("left1", left1), ("right1", right1), ("joined", (left1 + right1).bfloat16())):
                    torch.testing.assert_close(values[name], gold, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(3):
                    Runtime.run_into(runtime, *actuals)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profile:
                graph.replay()
                stream.synchronize()
            path = tmp_path / "fork-join.trace.json"
            profile.export_chrome_trace(str(path))
            events = [event for event in json.loads(path.read_text())["traceEvents"] if event.get("cat") == "kernel"]
            by_name = {
                name: sorted((e for e in events if e["name"].startswith("program_" + name)), key=lambda e: e["ts"])
                for name in ("left0", "left1", "right0", "right1", "joined")
            }
            assert len(events) == 15 and all(len(v) == 3 for v in by_name.values())
            end = lambda e: e["ts"] + e["dur"]
            for index in range(3):
                for source, target in (
                    ("left0", "left1"),
                    ("right0", "right1"),
                    ("left1", "joined"),
                    ("right1", "joined"),
                ):
                    assert end(by_name[source][index]) <= by_name[target][index]["ts"] + 0.005
                assert by_name["left0"][index]["args"]["stream"] != by_name["right0"][index]["args"]["stream"]
                if index:
                    for source in ("left0", "right0"):
                        assert end(by_name["joined"][index - 1]) <= by_name[source][index]["ts"] + 0.005
            torch.testing.assert_close(values["joined"], (left1 + right1).bfloat16(), rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        runtime.close()
