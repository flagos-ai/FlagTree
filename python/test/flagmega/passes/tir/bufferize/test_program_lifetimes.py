# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Program completion allows reuse without overlapping forked execution."""
from dataclasses import replace
from itertools import combinations, permutations
from types import SimpleNamespace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.functions import function_nodes
from triton.flagmega.targets import NvidiaSm90Target


def repeated_private_frame_graph():
    class Graph(fm.Module):
        def forward(self):
            ty = fm.tensor_type("bfloat16", (1, 128))
            p = self.input("p", ty, id="p")
            wa = self.input("wa", fm.tensor_type("bfloat16", (64, 128)), id="wa")
            wb = self.input("wb", fm.tensor_type("bfloat16", (32, 64)), id="wb")
            mid = fm.F.math.matmul(p, wa, transpose_b=True, name="mid")
            out = fm.F.math.matmul(mid, wb, transpose_b=True, name="out")
            self.function("worker", (p, wa, wb), (out,), attrs={"reusable": True, "noinline": True})
            x0 = self.input("x0", ty, id="x0")
            x1 = self.input("x1", ty, id="x1")
            a = self.input("a", wa.type, id="a")
            b = self.input("b", wb.type, id="b")
            calls = [fm.F.builtin.call(x, a, b, result_type=out.type, callee="worker", name=f"call{i}")
                     for i, x in enumerate((x0, x1))]
            self.function("main", (x0, x1, a, b), tuple(calls))
    return Graph(dialect="nn", stage="frozen_constants", entry="main").build()


def compile_private_frames(*, ordered=True):
    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            proposed = super().propose_tir(module)
            return replace(proposed, metadata={**proposed.metadata, "kernel_program_partition": (
                dict(symbol="first", calls=("call0",), num_warps=4, depends_on=()),
                dict(symbol="second", calls=("call1",), num_warps=4, depends_on=("first",) if ordered else ()))})
    compiler = Compiler()
    compiler.target = Target()
    return compiler.compile(repeated_private_frame_graph()).module


def test_completed_program_private_frame_is_reused_by_the_next_invocation():
    module = compile_private_frames()
    plan = fm.verify_buffer_plan(module)
    physical = {p.id: p for p in plan.physical_buffers}
    left, right = (physical[f"callframe:main:call{i}:workspace"] for i in range(2))
    assert left.nbytes > 0 and left.nbytes == right.nbytes
    assert left.live_end < right.live_start
    assert left.offset == right.offset
    assert plan.workspace_bytes == plan.function_map["worker"].workspace_bytes


def test_unordered_programs_cannot_share_private_frames():
    module = compile_private_frames(ordered=False)
    plan = fm.verify_buffer_plan(module)
    physical = {p.id: p for p in plan.physical_buffers}
    left, right = (physical[f"callframe:main:call{i}:workspace"] for i in range(2))
    assert left.live_end >= right.live_start
    assert left.offset + left.nbytes <= right.offset or right.offset + right.nbytes <= left.offset


def projection(parents):
    from triton.flagmega.passes.tir.bufferize.program_lifetimes import program_use_ends

    programs = tuple(dict(symbol=f"p{i}", calls=(f"call{i}",), depends_on=tuple(f"p{p}" for p in ps))
                     for i, ps in enumerate(parents))
    nodes = tuple(node for i in range(len(parents)) for node in (
        SimpleNamespace(id=f"call{i}", op="tir.call"), SimpleNamespace(id=f"view{i}", op="builtin.get_item")))
    module = SimpleNamespace(entry="main", stage="test", metadata={"kernel_program_partition": programs})
    return program_use_ends(module, SimpleNamespace(name="main"), nodes)


def test_projected_interference_is_sound_for_every_four_program_dag_and_topological_order():
    edges = tuple(combinations(range(4), 2))
    for mask in range(1 << len(edges)):
        selected = tuple(edge for i, edge in enumerate(edges) if mask & (1 << i))
        parents = [tuple(a for a, b in selected if b == i) for i in range(4)]
        positions = [{v: i for i, v in enumerate(order)} for order in permutations(range(4))
                     if all(order.index(a) < order.index(b) for a, b in selected)]
        ends = projection(parents)
        allocations = []
        for definition in range(4):
            later = [v for v in range(4) if v != definition and all(p[definition] < p[v] for p in positions)]
            for readers in range(1 << len(later)):
                uses = {definition, *(v for i, v in enumerate(later) if readers & (1 << i))}
                allocations.append((definition, uses, max(ends[f"call{v}"] for v in uses)))
        for left, right in combinations(allocations, 2):
            for earlier, later in ((left, right), (right, left)):
                if earlier[2] < later[0] * 2:
                    assert all(max(p[v] for v in earlier[1]) < p[later[0]] for p in positions), (selected, earlier, later)


def test_all_readers_contribute_even_when_the_last_reader_is_ordered_before_reuse():
    # p1 and p2 both read p0's allocation. p3 follows p2 but can overlap p1.
    ends = projection(((), (0,), (0,), (2,)))
    assert ends["call1"] >= 6
    assert ends["call2"] < 6
    assert max(ends[f"call{i}"] for i in (0, 1, 2)) >= 6


def test_forty_five_layer_fork_join_scratch_stops_before_the_next_layer():
    parents = []
    for layer in range(45):
        i = layer * 6
        parents.extend(((i - 1,) if layer else (), (i,), (i + 1,), (i,), (i + 3,), (i + 2, i + 4)))
    ends = projection(parents)
    for layer in range(45):
        i = layer * 6
        assert ends[f"call{i}"] == i * 2
        assert all(ends[f"call{j}"] < (i + 6) * 2 for j in range(i, i + 6))


@pytest.mark.parametrize("case", ["missing", "duplicate", "foreign", "reordered"])
def test_incomplete_or_invalid_legalized_entry_partitions_are_rejected_before_allocation(case):
    from triton.flagmega.errors import IRVerificationError
    from triton.flagmega.passes.tir.bufferize.program_lifetimes import program_use_ends

    programs = [dict(symbol="p0", calls=("a",), depends_on=()),
                dict(symbol="p1", calls=("b",), depends_on=("p0",))]
    if case == "missing":
        programs.pop()
    elif case == "duplicate":
        programs[1]["calls"] = ("a",)
    elif case == "foreign":
        programs[0]["calls"] = ("callee_only",)
    else:
        programs[0]["calls"], programs[1]["calls"] = ("b",), ("a",)
    module = SimpleNamespace(entry="main", stage="test", metadata={"kernel_program_partition": programs})
    nodes = (SimpleNamespace(id="a", op="tir.call"), SimpleNamespace(id="b", op="tir.call"))
    with pytest.raises(IRVerificationError, match="program lifetimes"):
        program_use_ends(module, SimpleNamespace(name="main"), nodes)


@pytest.mark.parametrize("ordered", [False, True])
def test_device_private_frame_reuse_preserves_distinct_outputs_and_graph(tmp_path, ordered):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

    module = compile_private_frames(ordered=ordered)
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True), device="cuda:0")
    generator = torch.Generator(device="cuda").manual_seed(523)
    inputs = {name: torch.empty(shape, dtype=torch.bfloat16, device="cuda") for name, shape in
              (("x0", (1, 128)), ("x1", (1, 128)), ("a", (64, 128)), ("b", (32, 64)))}
    storages = [torch.full((64,), -123., dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    outputs = [value[16:-16].reshape(1, 32) for value in storages]
    stream = torch.cuda.Stream()
    def args():
        results = iter(outputs)
        return tuple(next(results) if s["role"] == "result" else inputs[s["value"]] for s in runtime.external_arguments)
    try:
        Runtime.prepare(runtime, *args())
        assert all(report["spill_bytes"] == 0 for report in runtime.resource_report)
        for mutation in range(6):
            for value in inputs.values():
                value.copy_((torch.randint(-4, 5, value.shape, generator=generator, device="cuda").float()/16).bfloat16())
            if mutation == 5:
                inputs["x0"].zero_()
            for value in outputs:
                value.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            Runtime.run_into(runtime, *args(), stream=stream.cuda_stream)
            torch.cuda.synchronize()
            gold = [((inputs[name].double() @ inputs["a"].double().T).bfloat16().double()
                     @ inputs["b"].double().T).bfloat16() for name in ("x0", "x1")]
            assert all(torch.equal(value, expected) for value, expected in zip(outputs, gold, strict=True))
            assert all(bool((v[:16] == -123).all() & (v[-16:] == -123).all()) for v in storages)
            if mutation == 2:
                inputs = {name: value.clone() for name, value in inputs.items()}
        with torch.cuda.graph(capture := torch.cuda.CUDAGraph(), stream=stream):
            Runtime.run_into(runtime, *args(), stream=stream.cuda_stream)
        for _ in range(3):
            capture.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(value, expected) for value, expected in zip(outputs, gold, strict=True))
    finally:
        runtime.close()
