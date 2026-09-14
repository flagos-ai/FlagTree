# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.candidates import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.inference_providers import TypeInferenceCandidateProvider
from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine


@pytest.mark.parametrize("lanes", [1, 8])
@pytest.mark.parametrize("split", [False, True])
def test_dense_reshape_candidate_has_no_runtime_work(lanes, split):
    placement = fm.Placement((2, 4), "yx", "bb")
    dtype = "bfloat16" if lanes == 1 else fm.vector_type("bfloat16", (lanes,))
    source_type = fm.DistributedType(
        fm.tensor_type(dtype, (1, 256)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 64) if split else fm.SBP.broadcast()),
        placement,
    )
    builder = fm.IRBuilder(dialect="high_level", stage="unused_functions_removed")
    source = builder.var("source", source_type)
    definition = fm.get_definition("tensors.reshape")
    attrs = {"shape": (1, 8, 32)}
    result_type = definition.infer_type((source,), attrs)
    result = builder.call("tensors.reshape", (source,), result_type, attrs=attrs)
    builder.function("main", (source,), (result,))
    module = builder.build(entry="main")
    model = NvidiaSm90Machine().distributed_operation_cost_model()
    context = DistributedCandidateContext(module, result, placement, ((source_type,),), operation_cost_model=model)
    candidates = TypeInferenceCandidateProvider(frozenset({"tensors.reshape"})).get_candidates(context)
    assert candidates
    assert all(candidate.operation_cost == 0 for candidate in candidates)
    factors = definition.cost_factors((source,), attrs, result_type)
    assert factors.block_local_memory_load_bytes == 0
    assert factors.block_local_memory_store_bytes == 0


def test_non_dense_reshape_keeps_materialization_cost():
    definition = fm.get_definition("tensors.reshape")
    source_type = fm.tensor_type("float32", (4, 4), layout=fm.TensorLayout(strides=(1, 4)))
    source = fm.Node("source", "builtin.var", (), source_type, attrs={"name": "source"})
    result_type = definition.infer_type((source,), {"shape": (2, 8)})
    factors = definition.cost_factors((source,), {"shape": (2, 8)}, result_type)
    assert factors.block_local_memory_load_bytes > 0
    assert factors.block_local_memory_store_bytes > 0


@pytest.mark.parametrize("heads,axis,block", [(16, 1, 1), (2, 0, 1), (17, 0, 2)])
def test_reshape_consumer_demand_inverts_to_an_exact_flat_layout(heads, axis, block):
    mesh = fm.Placement((8, 16), "yx", "bb")
    dtype = fm.vector_type("bfloat16", (8,))
    tensor = fm.tensor_type(dtype, (1, heads * 32))
    broad = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, mesh)
    target = fm.DistributedType(fm.tensor_type(dtype, (1, heads, 32)), (
        fm.SBP.broadcast(), fm.SBP.split_block_cyclic((axis,), block), fm.SBP.broadcast(),
    ), mesh)
    flat = fm.DistributedType(tensor, (
        fm.SBP.broadcast(), fm.SBP.split_block_cyclic((axis,), block * 32),
    ), mesh)
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    source = builder.var("source", tensor)
    node = builder.call("tensors.reshape", (source,), target.tensor, attrs={"shape": (1, heads, 32)})
    builder.function("main", (source,), (node,))
    module = builder.build(entry="main")
    provider = TypeInferenceCandidateProvider(frozenset({"tensors.reshape"}))
    context = DistributedCandidateContext(module, node, mesh, ((broad,),))
    assert target in provider.get_return_candidate_types(context, (target,))
    tuples = provider.try_get_input_type_tuples(context, target)
    assert any(value.input_types == (flat,) for value in tuples)


@pytest.mark.parametrize("layout", [fm.TensorLayout(strides=(1, 4)), fm.TensorLayout(tag="opaque")])
def test_non_dense_reshape_is_not_lowered_as_an_alias(layout):
    from triton.flagmega.codegen.triton.lowering import _is_zero_copy_reshape

    builder = fm.IRBuilder(dialect="high_level", stage="selected_tir_variants")
    source = builder.var("source", fm.tensor_type("float32", (4, 4), layout=layout))
    result_type = fm.get_definition("tensors.reshape").infer_type((source,), {"shape": (2, 8)})
    result = builder.call("tensors.reshape", (source,), result_type, attrs={"shape": (2, 8)})
    builder.function("main", (source,), (result,))
    module = fm.verify_module(builder.build(entry="main"))
    assert not _is_zero_copy_reshape(result, module)
