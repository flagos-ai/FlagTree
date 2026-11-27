# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
import ast
from pathlib import Path
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.errors import CodegenError
from python.test.flagmega.codegen.triton.kernels.mhc._independent_pre_fixture import compile_graph

root = Path(__file__).resolve().parent


def artifact(tmp_path, independent=None, *, declared=None):
    g = stream_graph()
    m = compile_graph(g, independent, declared=declared)
    path = write_artifact(m, tmp_path, target="nvidia-sm90", emit_executable=True)
    return m, (path / "generated_kernels.py").read_text()


def test_independent_pre_schedule_has_only_the_projection_publication(tmp_path):
    m, source = artifact(tmp_path, True)
    tree = ast.parse(source)
    (f,) = (
        f
        for f in tree.body
        if isinstance(f, ast.FunctionDef)
        and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.endswith("__project")
            for n in ast.walk(f)
        )
    )
    assert sum(isinstance(n, ast.Call) and ast.unparse(n.func) == "tle.distributed_barrier" for n in ast.walk(f)) == 1
    (dispatch,) = (d.dispatch for d in m.kernel_definitions if d.dispatch.microkernel.family == "mhc_pre")
    assert dispatch.microkernel.facts["internal_grid_barriers"] == 1


@pytest.mark.parametrize("value", [0, 1, "yes"])
def test_independent_pre_rejects_untyped_parameters(tmp_path, value):
    with pytest.raises(CodegenError, match="independent_collapse_pre"):
        artifact(tmp_path, value)


def test_independent_pre_requires_matching_synchronization_facts(tmp_path):
    with pytest.raises(CodegenError, match="one declared"):
        artifact(tmp_path, True, declared=2)


def test_default_schedule_keeps_the_same_source(tmp_path):
    _, legacy = artifact(tmp_path / "implicit")
    _, explicit = artifact(tmp_path / "explicit", False)
    assert legacy == explicit


def test_emitted_task_rotation_preserves_every_writer(tmp_path):
    _, source = artifact(tmp_path, True)
    tree = ast.parse(source)
    (f,) = (f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name.endswith("__collapse"))
    (loop,) = (
        n
        for n in ast.walk(f)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "_fm_task"
    )
    start = loop.iter.args[0]
    definitions = {
        t.id: n.value for n in ast.walk(f) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)
    }

    def evaluate(n, env):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id] if n.id in env else evaluate(definitions[n.id], env)
        assert isinstance(n, ast.BinOp), ast.dump(n)
        a, b = evaluate(n.left, env), evaluate(n.right, env)
        return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mod: lambda: a % b}[type(n.op)]()

    for ctas in (1, 2, 8, 128):
        for tasks in (1, 3, 16, 129, 528):
            writes = [
                t
                for cta in range(ctas)
                for t in range(evaluate(start, {"_fm_cta": cta, "_fm_ctas": ctas}), tasks, ctas)
            ]
            assert sorted(writes) == list(range(tasks)), (ctas, tasks)


def test_nondefault_schedule_round_trips_before_rendering(tmp_path):
    graph = stream_graph()
    compiled = compile_graph(graph, True)
    restored = fm.load_module(fm.emit_module(compiled, tmp_path / "candidate.py"))
    assert restored.semantic_hash == compiled.semantic_hash
    path = write_artifact(restored, tmp_path / "resumed", target="nvidia-sm90", emit_executable=True)
    tree = ast.parse((path / "generated_kernels.py").read_text())
    (wrapper,) = (
        f
        for f in tree.body
        if isinstance(f, ast.FunctionDef)
        and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.endswith("__project")
            for n in ast.walk(f)
        )
    )
    assert (
        sum(isinstance(n, ast.Call) and ast.unparse(n.func) == "tle.distributed_barrier" for n in ast.walk(wrapper))
        == 1
    )


def test_independent_schedule_rejects_a_noncooperative_renderer():
    from triton.flagmega.codegen.triton.call_abi import describe_kernel_call_abis
    from triton.flagmega.codegen.triton.mhc_renderer import mhc_pre_call

    graph = stream_graph()
    compiled = compile_graph(graph, True)
    raw = next(v for v in describe_kernel_call_abis(compiled) if v["family"] == "mhc_pre")
    for group in ("inputs", "outputs", "workspaces"):
        for value in raw[group]:
            for buffer in value["buffers"]:
                buffer.setdefault("runtime_argument", value["formal"])
    raw["variant"] = "stream_fma"
    with pytest.raises(CodegenError, match="independent_collapse_pre requires"):
        mhc_pre_call(raw)


def stream_graph():
    from python.test.flagmega.codegen.triton.kernels.mhc.test_cooperative_stream_fma import distributed_module

    mesh = fm.Placement((8, 16), "yx", "bb")
    return distributed_module(
        tokens=1,
        hc=4,
        hidden=512,
        splits=4,
        output_lanes=(8,),
        placement=mesh,
        collapsed_dtype="float32",
        collapse_input_dtype="bfloat16",
        collapse_reduction="sequential_fma",
    )
