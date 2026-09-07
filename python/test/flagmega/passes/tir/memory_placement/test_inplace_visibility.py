# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Optional in-place reuse must preserve the result's memory placement."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir import (
    MEMORY_SPACE_METADATA,
    materialize_kernel_prim_functions,
    plan_function_memory,
)
from triton.flagmega.passes.tir.bufferize import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


def _case(*, widening, source_scope=fm.MemorySharingScope.BLOCK):
    generic = BufferizationOptions.generic(alignment=64)
    workspace, rdata, external = generic.memory_spaces
    options = BufferizationOptions(
        (
            replace(workspace, sharing_scope=fm.MemorySharingScope.CHIP),
            replace(workspace, name="private_data", sharing_scope=source_scope),
            rdata,
            external,
        ),
        block_local="private_data" if source_scope is fm.MemorySharingScope.BLOCK else None,
    )
    placement = fm.Placement((2, 4), "yx", "bb")
    split = fm.DistributedType(
        fm.tensor_type("float32", (1, 64)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 8)),
        placement,
    )
    broadcast = fm.DistributedType(
        split.tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    source = builder.var("source", split, id="source")
    rhs = builder.var("rhs", split, id="rhs")

    def kernel(name, inputs, value_type, op="math.silu"):
        return builder.call(
            "tir.kernel", inputs, value_type, id=name,
            attrs={
                "semantic_op": op,
                "candidate": "tir.silu.local",
                "parameters": {"family": "elementwise", "variant": "silu"},
                "facts": {}, "semantic_attrs": {},
            },
        )

    temporary = kernel("temporary", (source,), split)
    added = kernel("added", (temporary, rhs), split, "math.add")
    consumer_input = added
    if widening:
        consumer_input = builder.call(
            "distributed.sharded_view", (added,), broadcast, id="widened",
            attrs={"new_type": broadcast},
        )
    result = kernel("result", (consumer_input,), consumer_input.type)
    builder.function("main", (source, rhs), (result,))
    module = materialize_kernel_prim_functions(builder.build(entry="main"))
    if source_scope is fm.MemorySharingScope.BLOCK:
        module = plan_function_memory(module, options)
    else:
        module = replace(module, nodes=tuple(
            replace(node, metadata={MEMORY_SPACE_METADATA: "private_data"})
            if node.id == "temporary" else node for node in module.nodes
        ))
    return module, options


@pytest.mark.parametrize("source_scope", (fm.MemorySharingScope.BLOCK, fm.MemorySharingScope.DIE))
def test_widening_result_cannot_inherit_owner_private_inplace_storage(source_scope):
    module, options = _case(widening=True, source_scope=source_scope)
    assert module.node_map["temporary"].metadata[MEMORY_SPACE_METADATA] == "private_data"
    assert MEMORY_SPACE_METADATA not in module.node_map["added"].metadata

    plan = fm.make_buffer_plan(module, options=options)

    assert plan.buffer_map["temporary"].storage == "private_data"
    assert plan.buffer_map["added"].storage == "workspace"
    assert plan.buffer_map["added"].alias is None
    assert plan.buffer_map["widened"].mem_span.must_alias(plan.buffer_map["added"].mem_span)
    assert not plan.buffer_map["added"].mem_span.may_alias(plan.buffer_map["temporary"].mem_span)


def test_owner_local_inplace_reuse_is_preserved():
    module, options = _case(widening=False)
    plan = fm.make_buffer_plan(module, options=options)

    assert plan.buffer_map["added"].storage == "private_data"
    assert plan.buffer_map["added"].mem_span.must_alias(plan.buffer_map["temporary"].mem_span)


def test_equal_visibility_pool_can_supply_implicit_default_result():
    module, options = _case(widening=True, source_scope=fm.MemorySharingScope.CHIP)
    plan = fm.make_buffer_plan(module, options=options)

    assert plan.buffer_map["added"].storage == "private_data"
    assert plan.buffer_map["added"].mem_span.must_alias(plan.buffer_map["temporary"].mem_span)


def test_resume_rejects_private_inplace_result_without_private_placement():
    module, options = _case(widening=False)
    module = NttBufferizationPolicy(options).bufferize(module)
    fm.verify_buffer_plan(module)
    edited = replace(
        module,
        nodes=tuple(replace(node, metadata={}) if node.id == "added" else node
                    for node in module.nodes),
    )

    with pytest.raises(IRVerificationError, match="In-place result 'added'.*default memory domain"):
        fm.verify_buffer_plan(edited)
