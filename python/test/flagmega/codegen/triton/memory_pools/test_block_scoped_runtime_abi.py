# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.physical_access import emit_storage_pointer
from triton.flagmega.codegen.triton.runtime_binding import (
    describe_function_runtime_binding,
)
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_kernel_prim_functions,
)


def _attrs(candidate: str) -> dict[str, object]:
    return {
        "semantic_op": "math.add",
        "candidate": candidate,
        "parameters": {"family": "elementwise", "variant": "add"},
        "facts": {},
        "semantic_attrs": {},
    }


def _block_scoped_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("float32", (16,))
    lhs = builder.var("lhs", value_type, id="lhs")
    temporary = builder.call(
        "tir.kernel",
        (lhs, lhs),
        value_type,
        id="temporary",
        attrs=_attrs("tir.add.temporary"),
    )
    output = builder.call(
        "tir.kernel",
        (temporary, lhs),
        value_type,
        id="output",
        attrs=_attrs("tir.add.output"),
    )
    builder.function("main", (lhs,), (output,))
    selected = materialize_kernel_prim_functions(builder.build(entry="main"))
    plan = fm.make_buffer_plan(selected)
    spaces = tuple(
        replace(space, sharing_scope=fm.MemorySharingScope.BLOCK)
        if space.name == "workspace"
        else space
        for space in plan.memory_spaces
    )
    plan = replace(plan, memory_spaces=spaces)
    bound = bind_prim_function_buffers(selected, plan=plan)
    return fm.verify_module(replace(
        bound,
        stage="bufferized_tir",
        dialect="bufferized_tir",
        metadata={
            **bound.metadata,
            "buffer_plan": plan.to_data(),
            "launch_contract": {
                "kind": "single_prepared_entry",
                "entry": "flagmega_main",
                "cooperative_grid": True,
                "grid_mesh": {
                    "hierarchy": [2, 4],
                    "hierarchy_levels": "bb",
                    "name": "yx",
                },
            },
        },
    ))


def test_runtime_allocates_one_workspace_scope_per_physical_block():
    binding = describe_function_runtime_binding(_block_scoped_module())
    workspace = next(
        pool for pool in binding["pools"] if pool["storage"] == "workspace"
    )

    assert workspace == {
        "name": "workspace",
        "storage": "workspace",
        "nbytes": 8 * 256,
        "scope": "block",
        "scope_nbytes": 256,
        "scope_count": 8,
        "scope_index": "program_id_x",
        "scope_local": False,
    }


def test_local_pointer_selects_the_current_block_pool_before_value_offset():
    binding = describe_function_runtime_binding(_block_scoped_module())
    call = binding["call_abi"]["kernel_calls"][0]
    output = call["outputs"][0]["buffers"][0]
    abi = output["abi"]

    assert abi["pool_scope"] == "block"
    assert abi["pool_scope_count"] == 8
    assert abi["pool_scope_stride_bytes"] == 256
    assert emit_storage_pointer(abi, "workspace") == (
        "(workspace.to(tl.pointer_type(tl.float32)) + (shard_index) * 64)"
    )
