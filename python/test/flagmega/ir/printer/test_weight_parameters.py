# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager
from triton.flagmega.ir.printer import il_source, script_source


def parameter_module(*, dynamic=False, nested=False):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", (4, ))
    x = builder.var("input", tensor, id="input")
    weights = [builder.weight(str(i), tensor, source="missing", key=str(i), id=f"w{i}") for i in range(2)]
    p = builder.var("parameter", tensor, id="parameter")
    h = builder.var("activation", tensor, id="activation")
    transformed = builder.call("math.silu", (p, ), tensor, id="parameter_transform")
    out = builder.call("math.add", (h, transformed), tensor, id="compute")
    builder.function("worker", (p, h), (out, ))
    callee = "worker"
    if nested:
        wp = builder.var("wrapper_parameter", tensor, id="wrapper_parameter")
        wx = builder.var("wrapper_input", tensor, id="wrapper_input")
        call = builder.call("builtin.call", (wp, wx), tensor, attrs={"callee": callee}, id="inner_call")
        builder.function("wrapper", (wp, wx), (call, ))
        callee = "wrapper"
    calls = [
        builder.call("builtin.call", (x if dynamic and i else weight, x), tensor, attrs={"callee": callee},
                     id=f"call{i}") for i, weight in enumerate(weights)
    ]
    builder.function("main", (x, ), calls)
    return fm.verify_module(builder.build(entry="main"))


@pytest.mark.parametrize("render", (il_source, script_source))
@pytest.mark.parametrize("nested", (False, True))
def test_all_calls_prove_parameter_preprocessing_even_for_different_weights(render, nested):
    source = render(parameter_module(nested=nested))
    weights, compute = source.split("  // compute\n", 1)
    assert "    %w0 = Math.Silu(%parameter)" in weights
    assert "  %0 = Math.Add(%activation, %w0)" in compute


@pytest.mark.parametrize("render", (il_source, script_source))
def test_one_dynamic_call_or_missing_caller_prevents_weight_parameter_classification(render):
    module = parameter_module(dynamic=True, nested=True)
    assert "weights {" not in render(module)
    fragment = fm.function_module(parameter_module(), "worker")
    assert "weights {" not in render(fragment)


def test_dump_manager_keeps_parent_proof_without_changing_python_or_adding_files(tmp_path):
    module = parameter_module(nested=True)
    manager = DumpManager(tmp_path, DumpFlags.PASS_IR)
    manager.root.dump_module(module, "After", category=DumpFlags.PASS_IR)
    source = (tmp_path / "After/worker.il").read_text()
    assert "    %w0 = Math.Silu(%parameter)" in source
    assert sorted(p.name for p in (tmp_path / "After").iterdir()) == [
        "main.il", "main.py", "worker.il", "worker.py", "wrapper.il", "wrapper.py"
    ]
    assert fm.load_module(tmp_path / "After").semantic_hash == module.semantic_hash
    manager.root.dump_module(module, "Before", category=DumpFlags.PASS_IR)
    assert (tmp_path / "Before/worker.il").read_text() == source


def test_parent_analysis_is_shared_and_invalidated_when_calls_change(tmp_path, monkeypatch):
    from triton.flagmega.ir.print_weights import WeightPrintAnalysis
    analyze = WeightPrintAnalysis.analyze
    owners = []

    def observe(cls, module):
        owners.append(module)
        return analyze(module)

    monkeypatch.setattr(WeightPrintAnalysis, "analyze", classmethod(observe))
    constant = parameter_module(nested=True)
    dynamic = parameter_module(nested=True, dynamic=True)
    manager = DumpManager(tmp_path, DumpFlags.PASS_IR)
    for name, module in (("Before", constant), ("Repeated", constant), ("After", dynamic)):
        manager.root.dump_module(module, name, category=DumpFlags.PASS_IR)
    assert owners == [constant, dynamic]
    assert "weights {" in (tmp_path / "Before/worker.il").read_text()
    assert "weights {" not in (tmp_path / "After/worker.il").read_text()
