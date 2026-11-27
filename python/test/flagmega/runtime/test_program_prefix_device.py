# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A real materialized prefix, two branches and repeated captured requests."""

import json

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

from python.test.flagmega.codegen.triton.test_program_dag import compile_fork_join


def test_device_prefix_fork_join_matches_fp64_and_orders_repeated_requests(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    runtime = load(write_artifact(compile_fork_join(common_prefix=True), tmp_path / "artifact",
                                  target="nvidia-sm90", emit_executable=True), device="cuda:0")
    generator = torch.Generator(device="cuda").manual_seed(19373)
    shapes = dict(x=(1, 128), bias=(1, 128), a=(64, 128), b=(32, 64), c=(64, 128), d=(32, 64))
    values = {name: torch.empty(shape, dtype=torch.bfloat16, device="cuda") for name, shape in shapes.items()}
    guards = [torch.full((64,), 128., dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    for name, storage in zip(("left1", "right1", "joined"), guards):
        values[name] = storage[16:-16].reshape(1, 32)
    arguments = tuple(values[spec["value"]] for spec in runtime.external_arguments)
    caller = torch.cuda.Stream()
    caller.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(caller):
            Runtime.prepare(runtime, *arguments)
            assert len(runtime._program_dag.streams) == 2
            assert all(kernel.resource_report["spill_bytes"] == 0 for kernel in runtime._prepared)
            for mutation in range(6):
                for name in shapes:
                    values[name].copy_(torch.randint(-8, 9, shapes[name], generator=generator,
                                                    device="cuda").float()/8)
                for name in ("left1", "right1", "joined"):
                    values[name].fill_(float("nan"))
                Runtime.run_into(runtime, *arguments)
                caller.synchronize()
                source = (values["x"].double()+values["bias"].double()).bfloat16().double()
                left = ((source @ values["a"].double().T).bfloat16().double() @ values["b"].double().T).bfloat16()
                right = ((source @ values["c"].double().T).bfloat16().double() @ values["d"].double().T).bfloat16()
                expected = dict(left1=left, right1=right, joined=(left.double()+right.double()).bfloat16())
                for name, gold in expected.items():
                    torch.testing.assert_close(values[name], gold, rtol=0, atol=0)
                assert all(bool((guard[:16] == 128).all() & (guard[-16:] == 128).all()) for guard in guards)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=caller):
                for _ in range(3):
                    Runtime.run_into(runtime, *arguments)
            graph.replay()
            caller.synchronize()
            for name, gold in expected.items():
                torch.testing.assert_close(values[name], gold, rtol=0, atol=0)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                graph.replay()
                caller.synchronize()
            path = tmp_path / "prefix.trace.json"
            profile.export_chrome_trace(str(path))
            events = [event for event in json.loads(path.read_text())["traceEvents"] if event.get("cat") == "kernel"]
            assert len(events) == 18
            names = ("prefix", "left0", "left1", "right0", "right1", "joined")
            phases = {name: sorted((e for e in events if e["name"] == "program_"+name), key=lambda e: e["ts"])
                      for name in names}
            assert all(len(group) == 3 for group in phases.values())
            finish = lambda event: event["ts"]+event["dur"]
            for index in range(3):
                for left_name, right_name in (("prefix", "left0"), ("prefix", "right0"),
                    ("left0", "left1"), ("right0", "right1"), ("left1", "joined"), ("right1", "joined")):
                    assert finish(phases[left_name][index]) <= phases[right_name][index]["ts"]+.005
                assert phases["left0"][index]["args"]["stream"] != phases["right0"][index]["args"]["stream"]
                if index:
                    assert finish(phases["joined"][index-1]) <= phases["prefix"][index]["ts"]+.005
    finally:
        torch.cuda.synchronize()
        runtime.close()
