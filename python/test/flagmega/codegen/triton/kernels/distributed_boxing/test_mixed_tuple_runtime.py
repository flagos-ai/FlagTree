# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Mixed nested TensorStore/TensorLoad leaves share no transfer-family fiction."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.fixture(scope="module")
def mixed_boxing_module():
    tensor = fm.tensor_type("float32", (2, 259))
    placement = fm.Placement((8, 16), "yx", "bb")
    local = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 1)), placement)
    mixed_type = fm.TupleType((tensor, fm.TupleType((local, tensor))))

    class Transfer(fm.Module):
        def forward(self):
            a, b, c = (self.input(name, tensor) for name in ("a", "b", "c"))
            local_a = fm.F.distributed.force_boxing(a, local, name="load_a")
            value = fm.F.builtin.tuple(local_a, fm.F.builtin.tuple(b, c))
            mixed = fm.F.distributed.force_boxing(value, mixed_type, name="mixed")
            output_a = fm.F.tensors.get_item(mixed, 0)
            nested = fm.F.tensors.get_item(mixed, 1)
            output_b = fm.F.distributed.force_boxing(fm.F.tensors.get_item(nested, 0), tensor, name="store_b")
            output_c = fm.F.tensors.get_item(nested, 1)
            self.function("main", (a, b, c), (output_a, output_b, output_c))

    module = Transfer(dialect="distributed", stage="frozen_constants", entry="main",
                      metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    return Compiler().compile(module).module


def test_mixed_tuple_has_independent_transfer_contracts(mixed_boxing_module):
    calls = [call for call in describe_tir_package(mixed_boxing_module)["render_calls"]
             if call["family"] == "distributed_boxing"]
    assert len(calls) == 4  # load a, mixed store a/load b, store b; c aliases.
    assert all(len(call["leaves"]) == 1 for call in calls)
    assert sorted(call["leaves"][0]["transition"] for call in calls) == ["tensor_load"] * 2 + ["tensor_store"] * 2


def test_mixed_nested_tuple_matches_inputs_exactly(tmp_path, mixed_boxing_module):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    artifact = write_artifact(mixed_boxing_module, tmp_path / "mixed", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    values = tuple((torch.arange(518, device="cuda").reshape(2, 259) + 1000 * i).float() for i in range(3))
    allocated = tuple(torch.empty_like(value) for value in values[:2])
    outputs = (*allocated, values[2])
    plan = runtime.buffer_plan
    assert plan.entry_outputs[2][1] == plan.entry_inputs[2][1]
    assert len(runtime.external_arguments) == 5
    runtime.prepare(*values, *allocated)
    for _ in range(3):
        for output in allocated:
            output.fill_(float("nan"))
        runtime.run_into(*values, *allocated)
        torch.cuda.synchronize()
        for actual, expected in zip(outputs, values, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert runtime.resource_report["spill_bytes"] == 0
