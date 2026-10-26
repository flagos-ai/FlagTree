# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compatible dense final-axis packets share one reusable input ABI."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions.function_boundary_layout import propagate_function_boundary_layouts
from triton.flagmega.passes.functions.graph import function_nodes
from python.test.flagmega.devices import requires_nvidia_sm90


def module(*, axis=1, raw_output=False, compute=False):
    kind = fm.tensor_type("float32", (16, 16))

    class Graph(fm.Module):
        def forward(self):
            x = self.input("worker_x", kind, id="worker_x", metadata={"function_parameter": "worker"})
            small = fm.F.tensors.pack(x, lanes=(4,), axes=(axis,), name="small")
            wide = fm.F.tensors.pack(x, lanes=(2, 4), axes=(axis, axis), name="wide")
            computed = (fm.F.math.add(x, fm.F.builtin.splat_const(kind, 1.0)),) if compute else ()
            outputs = (small, wide, *computed, *((x,) if raw_output else ()))
            self.function("worker", (x,), outputs, attrs={"reusable": True})
            actual = tuple(self.input(f"x{i}", kind, id=f"x{i}") for i in range(2))
            calls = tuple(
                fm.F.builtin.call(
                    value, callee="worker", result_type=fm.TupleType(tuple(v.type for v in outputs)), name=f"call{i}"
                )
                for i, value in enumerate(actual)
            )
            results = tuple(v for call in calls for v in fm.F.tensors.get_items(call, *range(len(outputs))))
            if compute:
                exposed = []
                for i, value in enumerate(results):
                    if isinstance(value.type.dtype, fm.VectorType):
                        value = fm.F.tensors.unpack(value, axes=(1,) * len(value.type.dtype.lanes))
                    exposed.append(fm.F.math.add(value, fm.F.builtin.splat_const(kind, float(i * 11 + 3))))
                results = (*exposed, fm.F.math.add(*actual))
            self.function("main", actual, results)

    return Graph(dialect="high_level", stage="imported" if compute else "packed", entry="main").build()


@pytest.mark.parametrize("raw_output", [False, True])
def test_compatible_packets_change_the_abi_and_preserve_all_consumers(raw_output):
    source = module(raw_output=raw_output)
    result = propagate_function_boundary_layouts(source)
    worker = result.function_map["worker"]
    assert result.node_map[worker.parameters[0]].type.dtype == fm.VectorType(fm.DType.FLOAT32, (2, 4))
    assert not any(n.op == "tensors.pack" for n in function_nodes(result, worker))
    assert any(node.op == "tensors.bitcast" for node in result.nodes)
    inputs = {f"x{i}": (torch.arange(256, dtype=torch.float32).reshape(16, 16) + i * 257) for i in range(2)}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    expected = evaluator.run(source, inputs)
    actual = evaluator.run(result, inputs)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert propagate_function_boundary_layouts(result).semantic_hash == result.semantic_hash


def test_different_nonfinal_axis_packets_are_not_byte_views():
    source = module(axis=0)
    result = propagate_function_boundary_layouts(source)
    assert result.node_map["worker_x"].type == source.node_map["worker_x"].type
    assert all(result.node_map[key].op == "tensors.pack" for key in ("small", "wide"))


def test_repeated_logical_views_use_one_physical_result_binding():
    from triton.flagmega.compiler import Compiler

    class Repeated(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type("float32", (16,)), id="value")
            result = fm.F.math.add(value, value)
            self.function("main", (value,), (result, result))

    compiled = Compiler().compile(Repeated(dialect="high_level", stage="imported", entry="main").build()).module
    worker = compiled.execution_function_map["main"]
    logical = tuple(
        buffer for _, buffers in fm.verify_buffer_plan(compiled).function_map["main"].outputs for buffer in buffers
    )
    assert len(set(logical)) < len(logical)
    assert worker.results == tuple(dict.fromkeys(logical))


def test_conflicting_physical_result_bindings_are_rejected():
    from types import SimpleNamespace

    from triton.flagmega.errors import IRVerificationError
    from triton.flagmega.passes.tir.materialize_execution_functions import _nested_call

    record = SimpleNamespace(arguments=(), results=(("result", "first"), ("result", "second")))
    with pytest.raises(IRVerificationError, match="binds physical result.*different caller buffers"):
        _nested_call("main", SimpleNamespace(id="call"), record, None, ())


@requires_nvidia_sm90
def test_compatible_input_views_execute_through_reusable_device_calls(tmp_path):
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load

    source = module(raw_output=True, compute=True)
    inputs = {f"x{i}": torch.arange(256, dtype=torch.float32).reshape(16, 16) * 0.25 + i * 257 for i in range(2)}
    expected = TorchEvaluator(DictWeightResolver({})).run(source, inputs)
    compiled = Compiler().compile(source).module

    class Constants:
        def load_tensor(self, key, *, device="cpu"):
            raise AssertionError(f"unexpected checkpoint access: {key}")

    artifact = write_artifact(
        compiled, tmp_path / "artifact", target="nvidia-sm90", checkpoint=Constants(), emit_executable=True
    )
    runtime = load(artifact, device="cuda:0")
    try:
        values = {key: value.cuda() for key, value in inputs.items()}
        outputs = tuple(torch.empty_like(value, device="cuda") for value in expected)
        specs = [spec for spec in runtime.external_arguments if spec["role"] == "result"]
        values.update({spec["value"]: value for spec, value in zip(specs, outputs, strict=True)})
        args = tuple(values[spec["value"]] for spec in runtime.external_arguments)
        runtime.prepare(*args)
        runtime.run_into(*args)
        for actual, wanted in zip(outputs, expected, strict=True):
            torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
