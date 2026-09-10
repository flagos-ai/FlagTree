# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target
import pytest


@pytest.mark.parametrize("generated", [False, True])
@pytest.mark.parametrize("lanes,producer_lanes", [((4,), (4,)), ((2, 4), (8,))])
def test_newly_exposed_callee_pack_reaches_caller_producer(generated, lanes, producer_lanes):

    class Graph(fm.Module):

        def forward(self):
            scalar = fm.tensor_type("float32", (1, 32))
            param = self.input("param", scalar, id="param")
            result = fm.F.tensors.pack(
                fm.F.math.sigmoid(param), lanes, axes=(1,) * len(lanes),
                metadata={"vectorization_internal": True, "vectorization_root": "consumer"} if generated else {})
            self.function("layer", (param, ), (result, ), attrs={"reusable": True})
            ids = self.input("ids", fm.tensor_type("int32", (1, )), id="ids")
            weight = self.weight("weight", fm.tensor_type("bfloat16", (17, 32)), source="weights", key="weight")
            embedding = fm.F.nn.embedding(ids, weight)
            call = fm.F.builtin.call(fm.F.tensors.cast(embedding, "float32"), callee="layer", result_type=result.type)
            self.function("main", (ids, ), (call, ))

    original = Graph(dialect="ntt", stage="packed", entry="main").build()
    result = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    parameter, = result.function_map["layer"].parameters
    assert result.node_map[parameter].type.dtype == fm.vector_type("float32", lanes)
    embedding, = (n for n in result.nodes if n.op == "nn.embedding")
    assert embedding.type.dtype == fm.vector_type("bfloat16", producer_lanes)
    assert all(result.node_map[n.inputs[0]].op == "builtin.weight" for n in result.nodes if n.op == "tensors.pack")
    assert post_function_boundary_pack_propagation(result, NvidiaSm90Target()).semantic_hash == result.semantic_hash
