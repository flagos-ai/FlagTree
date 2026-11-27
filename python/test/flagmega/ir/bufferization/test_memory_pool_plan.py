# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.bufferization import (
    BUFFER_PLAN_SCHEMA,
    LEGACY_BUFFER_PLAN_SCHEMA,
    BufferPlan,
    CallBufferBinding,
    CallMemoryPoolBinding,
    FunctionBufferPlan,
    FunctionMemoryPool,
)
from triton.flagmega.passes.tir.bufferize import BufferizationOptions


def test_function_and_call_memory_pools_round_trip_without_workspace_projection():
    function = FunctionBufferPlan(
        "worker",
        (),
        (),
        memory_pools=(
            FunctionMemoryPool("chip_data", 512, 256, ("chip-buffer",)),
            FunctionMemoryPool("block_data", 128, 64, ("block-buffer",)),
        ),
    )
    call = CallBufferBinding(
        "invoke",
        "main",
        "worker",
        (),
        (),
        memory_pools=(
            CallMemoryPoolBinding("chip_data", "chip-frame", 256, 512),
            CallMemoryPoolBinding("block_data", "block-frame", 64, 128),
        ),
    )

    function_data = function.to_data()
    call_data = call.to_data()

    assert "workspace_bytes" not in function_data
    assert "workspace_allocation" not in call_data
    assert FunctionBufferPlan.from_data(function_data) == function
    assert CallBufferBinding.from_data(call_data) == call
    with pytest.raises(IRSchemaError, match="multiple pools"):
        _ = function.workspace_bytes


def test_legacy_v5_custom_workspace_identity_resumes_and_reemits_v6():
    generic = BufferizationOptions.generic()
    spaces = tuple(
        replace(space, name="data") if space.name == "workspace" else space
        for space in generic.memory_spaces
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", fm.tensor_type("float32", (16,)), id="source")
    output = builder.call("test.unary", (source,), source.type, id="output")
    builder.function("main", (source,), (output,))
    plan = fm.make_buffer_plan(
        builder.build(entry="main"),
        options=BufferizationOptions(spaces, workspace="data"),
    )
    legacy = plan.to_data()
    legacy["schema"] = LEGACY_BUFFER_PLAN_SCHEMA
    legacy.pop("default_workspace")
    for function in legacy["functions"]:
        pool = function.pop("memory_pools")[0]
        function.update({
            "workspace_bytes": pool["scope_bytes"],
            "workspace_alignment": pool["alignment"],
            "allocations": pool["allocations"],
        })
        for call in function["calls"]:
            pools = call.pop("memory_pools")
            pool = pools[0] if pools else None
            call.update({
                "workspace_allocation": None if pool is None else pool["allocation"],
                "workspace_offset": 0 if pool is None else pool["offset"],
                "workspace_bytes": 0 if pool is None else pool["scope_bytes"],
            })

    resumed = BufferPlan.from_data(legacy)
    encoded = resumed.to_data()

    assert resumed.default_workspace == "data"
    assert resumed.function_map["main"].memory_pools[0].memory_space == "data"
    assert encoded["schema"] == BUFFER_PLAN_SCHEMA
    assert encoded["default_workspace"] == "data"
    assert "workspace_bytes" not in encoded["functions"][0]


def test_nonempty_call_pool_requires_a_physical_frame():
    with pytest.raises(IRSchemaError, match="requires an allocation"):
        CallMemoryPoolBinding("block_data", None, 0, 64)
