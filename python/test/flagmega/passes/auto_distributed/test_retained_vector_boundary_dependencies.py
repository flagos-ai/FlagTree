# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts


def test_scalar_slice_bitcast_remains_a_boundary_before_packed_matmul():
    mesh = fm.Placement((8, 16), "yx", "bb")
    b = fm.SBP.broadcast()
    value_type = fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", (4,)), (1, 32)), (b, b), mesh)
    rhs_type = fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (4, 8)), (b, b), mesh)

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", value_type, id="value")
            rhs = self.input("rhs", rhs_type, id="rhs")
            metadata = {"vectorized_from": "tensors.slice", "vectorization_root": "value",
                        "vectorization_semantic_id": "slice", "vectorization_candidate": "vectorization.last_axis",
                        "vector_axes": (1,), "vector_lanes": (4,), "vectorization_inputs": ("value",),
                        "vectorization_attrs": {"starts": (0,), "ends": (64,), "axes": (1,), "steps": (1,)}}
            sliced = fm.F.tensors.slice(value, starts=(0,), ends=(16,), axes=(1,), name="slice.physical",
                                       metadata={**metadata, "vectorization_internal": True})
            scalar = fm.F.tensors.bitcast(sliced, "bfloat16", name="slice", metadata=metadata)
            none = fm.F.builtin.none(name="none")
            output = fm.F.ntt.packed_matmul(scalar, rhs, none, none, name="projection")
            self.function("main", (value, rhs), (output,))

    original = Graph(dialect="ntt", stage="distributed", entry="main").build()
    result = fm.verify_module(lower_vectorization_contracts(original))
    assert result.node_map["slice"].op == "tensors.bitcast"
    assert result.node_map["slice.physical"] == original.node_map["slice.physical"]
    assert result.node_map["projection"].inputs[0] == "slice"


def test_retained_result_boundary_keeps_its_physical_slice_dependency():

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type(fm.vector_type("bfloat16", (8, )), (2, 4)), id="value")
            compute = fm.F.math.vectorized_unary(
                value, unary_op="sigmoid", name="compute", metadata={
                    "vectorization_root": "boundary", "vectorization_semantic_id": "sigmoid", "vectorized_from":
                    "math.sigmoid", "vectorization_internal": True, "vectorization_candidate":
                    "vectorization.last_axis", "vector_axes": (1, ), "vector_lanes": (8, )
                })
            sliced = fm.F.tensors.slice(
                value, starts=(1, ), ends=(3, ), axes=(1, ), name="slice", metadata={
                    "vectorization_root": "old_crop", "vectorization_semantic_id": "scalar_crop", "vectorized_from":
                    "tensors.slice", "vectorization_internal": True, "vectorization_candidate":
                    "vectorization.propagated", "vector_axes": (1, ), "vector_lanes": (8, ), "vectorization_inputs":
                    ("value", ), "vectorization_attrs":
                    {"starts": (8, ), "ends": (24, ), "axes": (1, ), "steps": (1, )}
                })
            output = fm.F.tensors.concat(
                compute, sliced, axis=1, name="concat", metadata={
                    "vectorization_internal": True, "vectorized_from": "tensors.concat", "vectorization_semantic_id":
                    "boundary"
                })
            self.function("main", (value, ), (output, ))

    original = Graph(dialect="ntt", stage="distributed", entry="main").build()
    result = fm.verify_module(lower_vectorization_contracts(original))
    assert result.node_map["slice"] == original.node_map["slice"]
    assert result.node_map["concat"].inputs == ("compute", "slice")


def test_native_consumer_retains_authoritative_vector_reshape_bitcast_chain():
    mesh = fm.Placement((8, 16), "yx", "bb")
    value_type = fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 16, 32)),
                                    (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 1), fm.SBP.broadcast()), mesh)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", value_type, id="x")
            product = fm.F.math.vectorized_binary(x, x, binary_op="mul", name="product.compute", metadata={
                "vectorized_from": "math.mul", "vectorization_root": "product",
                "vectorization_semantic_id": "product", "vectorization_internal": True,
                "vectorization_candidate": "vectorization.last_axis", "vector_axes": (2,), "vector_lanes": (8,),
            })
            metadata = {"vectorized_from": "tensors.reshape", "vectorization_root": "flatten",
                        "vectorization_semantic_id": "flatten", "vectorization_candidate": "vectorization.propagated",
                        "vector_axes": (1,), "vector_lanes": (8,), "vectorization_inputs": ("product",),
                        "vectorization_attrs": {"shape": (1, 4096)}}
            reshaped = fm.F.tensors.reshape(product, (1, 512), name="flatten.physical",
                                            metadata={**metadata, "vectorization_internal": True})
            flat = fm.F.tensors.bitcast(reshaped, "bfloat16", name="flatten", metadata=metadata)
            stats = fm.F.nn.norm_stats(flat, axis=1, use_mean=False, name="stats")
            self.function("main", (x,), (stats,))

    original = Graph(dialect="ntt", stage="distributed", entry="main").build()
    result = fm.verify_module(lower_vectorization_contracts(original))
    assert result.node_map["flatten"].op == "tensors.bitcast"
    assert result.node_map["flatten.physical"] == original.node_map["flatten.physical"]
    assert not any(node.op == "tensors.unpack" for node in result.nodes)


def test_unannotated_native_consumer_keeps_concat_reshape_across_boxing():
    mesh = fm.Placement((8, 16), "yx", "bb")
    b = fm.SBP.broadcast()
    value_type = fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 4)), (b, b), mesh)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", value_type, id="x")
            compute = fm.F.math.vectorized_unary(x, unary_op="sigmoid", name="sigmoid.compute", metadata={
                "vectorization_root": "sigmoid", "vectorization_semantic_id": "sigmoid",
                "vectorized_from": "math.sigmoid", "vectorization_internal": True,
                "vectorization_candidate": "vectorization.last_axis", "vector_axes": (1,), "vector_lanes": (8,),
            })
            concat = fm.F.tensors.concat(compute, compute, axis=1, name="pair.physical", metadata={
                "vectorization_root": "pair", "vectorization_semantic_id": "pair",
                "vectorized_from": "tensors.concat", "vectorization_internal": True,
                "vectorization_candidate": "vectorization.propagated", "vector_axes": (1,), "vector_lanes": (8,),
                "vectorization_inputs": ("sigmoid", "sigmoid"), "vectorization_attrs": {"axis": 1},
            })
            reshaped = fm.F.tensors.reshape(concat, (1, 1, 8), name="heads.physical", metadata={
                "vectorization_root": "heads", "vectorization_semantic_id": "heads",
                "vectorized_from": "tensors.reshape", "vectorization_internal": True,
                "vectorization_candidate": "vectorization.propagated", "vector_axes": (2,), "vector_lanes": (8,),
                "vectorization_inputs": ("pair",), "vectorization_attrs": {"shape": (1, 1, 64)},
            })
            boxed = fm.F.distributed.force_boxing(reshaped, reshaped.type, name="boxed")
            output = fm.F.nn.norm_stats(boxed, axis=2, use_mean=False, name="stats")
            self.function("main", (x,), (output,))

    original = Graph(dialect="ntt", stage="distributed", entry="main").build()
    result = fm.verify_module(lower_vectorization_contracts(original))
    assert result.node_map["pair.physical"].type == original.node_map["pair.physical"].type
    assert result.node_map["heads.physical"].type == original.node_map["heads.physical"].type
    assert not any(node.op == "tensors.pack" for node in result.nodes)
