# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""tensors.gather: evaluator numerics and the executable Triton kernel."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.tensors.gather import Gather
from triton.flagmega.runtime import load

VALUE_EXTENT = 288
ROWS = 8
SELECTED = 8


def _gather_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported", metadata={"model": "gather-smoke"})
    value_type = fm.tensor_type("float32", [ROWS, VALUE_EXTENT])
    index_type = fm.tensor_type("int32", [ROWS, SELECTED])
    value = builder.var("value", value_type, id="value")
    indices = builder.var("indices", index_type, id="indices")
    gathered = builder.call("tensors.gather", [value, indices], fm.tensor_type("float32", [ROWS, SELECTED]),
                            id="gathered", attrs={"axis": -1})
    builder.function("main", [value, indices], [gathered])
    return builder.build(entry="main")


def _sample(seed: int):
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(ROWS, VALUE_EXTENT, generator=generator)
    indices = torch.randint(0, VALUE_EXTENT, (ROWS, SELECTED), generator=generator).to(torch.int32)
    return value, indices


def test_gather_evaluator_matches_torch():
    module = _gather_module()
    value, indices = _sample(7)
    outputs = TorchEvaluator(DictWeightResolver({})).run(module, {"value": value, "indices": indices})
    expected = torch.gather(value, -1, indices.to(torch.int64))
    torch.testing.assert_close(outputs[0], expected)


def test_gather_allows_an_index_axis_larger_than_the_source():
    value = fm.Node("value", "builtin.var", (), fm.tensor_type("float32", [1, 4]))
    indices = fm.Node("indices", "builtin.var", (), fm.tensor_type("int32", [1, 5]))
    assert Gather.infer_type((value, indices), {"axis": -1}) == fm.tensor_type("float32", [1, 5])


def test_gather_rejects_mixed_distribution_contract():
    placement = fm.Placement((2,), "x", "b")
    value = fm.Node("value", "builtin.var", (), fm.tensor_type("float32", [1, 4]))
    indices = fm.Node(
        "indices", "builtin.var", (),
        fm.DistributedType(fm.tensor_type("int32", [1, 2]), (fm.SBP.broadcast(),) * 2, placement),
    )
    with pytest.raises(IRSchemaError, match="same distribution contract"):
        Gather.infer_type((value, indices), {"axis": -1})


@pytest.mark.cuda
def test_gather_executable_artifact_matches_torch():
    module = _gather_module()
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, Path(tempfile.mkdtemp(prefix="flagmega-gather-")), target="nvidia-sm90",
                              emit_executable=True)
    runtime = load(artifact, device="cuda:0")

    value, indices = _sample(11)
    device_value = value.to("cuda:0")
    device_indices = indices.to("cuda:0")
    output = torch.empty((ROWS, SELECTED), dtype=torch.float32, device="cuda:0")
    runtime.prepare(device_value, device_indices, output=output)
    runtime.run_into(output, device_value, device_indices)
    torch.cuda.synchronize()
    expected = torch.gather(value, -1, indices.to(torch.int64))
    torch.testing.assert_close(output.cpu(), expected)
