# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Nested tuple transfers through real buffer planning and flattened ABI."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.fixture(scope="module", params=(False, True))
def nested_boxing_module(request):
    value_type = fm.tensor_type("float32", (2, 259))
    placement = fm.Placement((8, 16), "yx", "bb")
    local_type = fm.DistributedType(value_type, (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 1)), placement)
    subtree_type = value_type if request.param else local_type
    local_tuple = fm.TupleType((local_type, fm.TupleType((subtree_type, subtree_type))))
    result_type = fm.TupleType((value_type, fm.TupleType((value_type, value_type))))

    class Transfer(fm.Module):
        def forward(self):
            a, b, c = (self.input(name, value_type) for name in ("a", "b", "c"))
            value = fm.F.builtin.tuple(a, fm.F.builtin.tuple(b, c))
            local = fm.F.distributed.force_boxing(value, local_tuple, name="load")
            result = fm.F.distributed.force_boxing(local, result_type, name="store")
            self.function("main", (a, b, c), (result,))

    module = Transfer(dialect="distributed", stage="frozen_constants", entry="main",
                      metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    return Compiler().compile(module).module


def test_nested_tuple_codegen_retains_all_tensor_leaves(nested_boxing_module):
    package = describe_tir_package(nested_boxing_module)
    boxing_calls = [call for call in package["render_calls"] if call["family"] == "distributed_boxing"]
    # Identity subtrees retain their input aliases. Every changed leaf owns
    # its individual transfer/effect contract, as in nncase GenerateBoxingValue.
    assert len(boxing_calls) in (2, 6)
    assert all(len(call["leaves"]) == 1 for call in boxing_calls)


def test_nested_tuple_transfers_every_leaf_on_gpu(tmp_path, nested_boxing_module):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    artifact = write_artifact(nested_boxing_module, tmp_path / "nested", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    values = tuple((torch.arange(518, device="cuda").reshape(2, 259) + 1000 * i).float() for i in range(3))
    result_count = len(runtime.external_arguments) - len(values)
    assert result_count in (1, 3)
    allocated = tuple(torch.empty_like(value) for value in values[:result_count])
    # This low-level API takes physical external buffers, not duplicated alias
    # results. The unchanged subtree is returned through its original buffers.
    outputs = (*allocated, *values[result_count:])
    plan = runtime.buffer_plan
    output_buffers = plan.entry_outputs[0][1]
    input_buffers = tuple(buffers[0] for _, buffers in plan.entry_inputs)
    assert output_buffers[result_count:] == input_buffers[result_count:]
    runtime.prepare(*values, *allocated)
    for _ in range(3):
        for output in allocated:
            output.fill_(float("nan"))
        runtime.run_into(*values, *allocated)
        torch.cuda.synchronize()
        for output, expected in zip(outputs, values, strict=True):
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert runtime.resource_report["spill_bytes"] == 0
