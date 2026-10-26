# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A numeric cast preserves vector elements without inventing packing axes."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.vectorization import vectorization_contract
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.rules.neutral._utility import make_node
from python.test.flagmega.devices import is_nvidia_sm90


class TypedCastGraph(fm.Module):
    def __init__(self, shape=(1, 4), lanes=(2, 2), source="float32", target="bfloat16"):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.shape, self.lanes, self.source, self.target = shape, lanes, source, target

    def forward(self):
        value = self.input("value", fm.tensor_type(fm.vector_type(self.source, self.lanes), self.shape), id="value")
        result = fm.F.tensors.cast(value, fm.vector_type(self.target, self.lanes), name="cast")
        self.function("main", (value,), (result,))


@pytest.mark.parametrize("shape", ((), (1, 4), (2, 1, 3)))
def test_plain_typed_cast_has_an_element_lane_contract_without_agent_metadata(shape):
    node = TypedCastGraph(shape=shape).build().node_map["cast"]
    contract = vectorization_contract(node)
    assert contract["kind"] == "typed_elements"
    assert contract["axes"] == ()
    assert contract["lanes"] == (2, 2) and contract["lane_count"] == 4


@pytest.mark.parametrize("source_lanes,target_lanes", (((), (4,)), ((4,), ()), ((2, 2), (4,))))
def test_numeric_cast_cannot_silently_repack_vector_elements(source_lanes, target_lanes):
    source = fm.vector_type("float32", source_lanes) if source_lanes else fm.DType.FLOAT32
    target = fm.vector_type("bfloat16", target_lanes) if target_lanes else fm.DType.BFLOAT16
    value = fm.Node("value", "builtin.var", (), fm.tensor_type(source, (4,)))
    with pytest.raises(IRSchemaError, match="lanes"):
        make_node("tensors.cast", "cast", (value,), {"dtype": target}, {})


@pytest.mark.parametrize("shape", ((), (1, 4), (2, 1, 3)))
@pytest.mark.parametrize("source,target", (("float32", "bfloat16"), ("bfloat16", "float32")))
def test_plain_vector_cast_executes_every_element_lane(tmp_path, shape, source, target):
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load

    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    graph = TypedCastGraph(shape=shape, source=source, target=target)
    compiled = Compiler().compile(graph.build()).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    try:
        value = torch.randn((*shape, *graph.lanes), dtype=getattr(torch, source), device="cuda")
        runtime.prepare(value)
        got = runtime.run(value)
        torch.testing.assert_close(got, value.to(getattr(torch, target)), atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
