# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts


def test_cloned_result_boundary_retains_its_exact_packed_compute_operands():
    scalar = fm.tensor_type("bfloat16", (1, 16))
    vector = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 2))
    lhs = fm.Node("lhs", "builtin.var", (), scalar, attrs={"name": "lhs"})
    rhs = fm.Node("rhs", "builtin.var", (), scalar, attrs={"name": "rhs"})
    lhs_pack = fm.Node(
        "lhs.pack",
        "tensors.pack",
        (lhs.id,),
        vector,
        attrs={"lanes": (8,), "axis": 1},
        metadata={
            "vectorization_internal": True,
            "vectorization_role": "pack",
            "vectorization_root": "sum",
        },
    )
    rhs_pack = fm.Node(
        "rhs.pack",
        "tensors.pack",
        (rhs.id,),
        vector,
        attrs={"lanes": (8,), "axis": 1},
        metadata={
            "vectorization_internal": True,
            "vectorization_role": "pack",
            "vectorization_root": "sum",
        },
    )
    original = fm.Node(
        "sum.compute",
        "math.add",
        (lhs_pack.id, rhs_pack.id),
        vector,
        metadata={
            "vectorization_internal": True,
            "vectorization_role": "compute",
            "vectorization_root": "sum",
        },
    )
    cloned = fm.Node(
        "sum.compute.specialized",
        "math.add",
        (lhs_pack.id, rhs_pack.id),
        vector,
        metadata={
            "cloned_for_function_variant": "packed_boundary",
            "vectorization_candidate": "vectorization.add.last_axis",
            "vectorization_attrs": {},
            "vectorization_internal": True,
            "vectorization_role": "compute",
            "vectorization_root": "sum",
            "vector_axes": (1,),
            "vector_lanes": (8,),
            "vectorized_from": "math.add",
        },
    )
    boundary = fm.Node(
        "sum.specialized",
        "tensors.bitcast",
        (cloned.id,),
        scalar,
        metadata={
            "cloned_for_function_variant": "packed_boundary",
            "vectorization_candidate": "vectorization.add.last_axis",
            "vectorization_attrs": {},
            "vector_axes": (1,),
            "vector_lanes": (8,),
            "vectorized_from": "math.add",
        },
    )
    module = fm.IRModule(
        "ntt",
        "norm_bindings_finalized",
        (lhs, rhs, lhs_pack, rhs_pack, original, cloned, boundary),
        (),
        "main",
    )

    lowered = lower_vectorization_contracts(module)

    assert lowered.node_map[cloned.id].op == "math.add"
    assert lowered.node_map[cloned.id].inputs == (lhs_pack.id, rhs_pack.id)
    assert lowered.node_map[cloned.id].type == vector
    assert lowered.node_map[boundary.id].op == "tensors.bitcast"
    assert lowered.node_map[boundary.id].inputs == (cloned.id,)
    assert lowered.node_map[boundary.id].type == scalar
    assert lhs_pack.id in lowered.node_map
    assert rhs_pack.id in lowered.node_map


def test_native_vector_dependency_survives_a_distributed_view_bridge():
    """Distributed views must not hide typed-vector producer liveness."""

    scalar = fm.tensor_type("bfloat16", (1, 16))
    vector = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 2))
    value = fm.Node("value", "builtin.var", (), scalar, attrs={"name": "value"})
    packed_reshape = fm.Node(
        "packed_reshape",
        "tensors.reshape",
        (value.id,),
        vector,
        attrs={"shape": (1, 2)},
        metadata={
            "vectorization_internal": True,
            "vectorization_role": "propagated-reshape",
            "vectorization_root": "native",
            "vectorization_semantic_id": "semantic_reshape",
            "vectorization_candidate": "vectorization.propagated",
            "vectorization_attrs": {"shape": (1, 16)},
            "vector_axes": (1,),
            "vector_lanes": (8,),
            "vectorized_from": "tensors.reshape",
        },
    )
    view = fm.Node(
        "view",
        "distributed.sharded_view",
        (packed_reshape.id,),
        vector,
        metadata={"realization": "sharded_view"},
    )
    structural = fm.Node(
        "structural",
        "builtin.tuple",
        (view.id,),
        fm.TupleType((vector,)),
        metadata={
            "vectorization_internal": True,
            "vectorization_role": "tuple",
            "vectorization_root": "native",
        },
    )
    native = fm.Node(
        "native",
        "ntt.vectorized_rope",
        (structural.id,),
        vector,
        metadata={
            "vectorization_candidate": "vectorization.native",
            "vectorization_attrs": {},
            "vector_axes": (1,),
            "vector_lanes": (8,),
            "vectorized_from": "nn.rope",
        },
    )
    module = fm.IRModule(
        "ntt",
        "norm_bindings_finalized",
        (value, packed_reshape, view, structural, native),
        (),
        "main",
    )

    lowered = lower_vectorization_contracts(module)

    retained = lowered.node_map[packed_reshape.id]
    assert retained.op == "tensors.reshape"
    assert retained.type == vector
    assert retained.metadata["vectorization_internal"] is True
    assert lowered.node_map[view.id].inputs == (packed_reshape.id,)
    assert all(
        input_id in lowered.node_map
        for node in lowered.nodes
        for input_id in node.inputs
    )
