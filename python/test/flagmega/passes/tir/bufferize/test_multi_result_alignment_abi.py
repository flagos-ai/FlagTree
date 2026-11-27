# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A strict result must not pad the per-owner stride of its siblings.

Regression for a layer-to-layer physical ABI mismatch: a kernel returning
``(value, statistics)`` declared a 16-byte contract on the value only, but the
requirement was recorded against the whole call result.  The statistics buffer
was then allocated with a padded owner stride in the caller while the callee
computed the unpadded one for its formal parameter, so stitching two layers
together failed the call-argument physical ABI check.
"""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.ops.ntt.matmul_norm_stats import MatMulNormStats
from triton.flagmega.passes.tir import materialize_kernel_prim_functions
from triton.flagmega.passes.tir.bufferize.alignment import (
    transfer_source_alignment_requirements,
)

PLACEMENT = fm.Placement((2, 2), "yx", "bb")


def _broadcast(dtype, shape):
    return fm.DistributedType(
        fm.tensor_type(dtype, shape),
        (fm.SBP.broadcast(),) * len(shape),
        PLACEMENT,
    )


def _n_split(dtype, shape):
    """Split the last axis, so the value result is sharded per owner."""
    return fm.DistributedType(
        fm.tensor_type(dtype, shape),
        (fm.SBP.broadcast(),) * (len(shape) - 1)
        + (fm.SBP.split_contiguous((0, 1)),),
        PLACEMENT,
    )


def _module(*, value_alignment=16):
    """Two layers chained through a (value, statistics) kernel result."""

    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    lhs_type = _broadcast("bfloat16", (1, 16))
    rhs_type = _n_split("bfloat16", (16, 16))
    value_type = _n_split("bfloat16", (1, 16))

    lhs = builder.var("lhs", lhs_type, id="lhs")
    rhs = builder.var("rhs", rhs_type, id="rhs")
    addend = builder.var("addend", value_type, id="addend")
    prepared = MatMulNormStats.prepare(
        (lhs, rhs, addend),
        {"transpose_a": False, "transpose_b": False, "axis": -1, "use_mean": False},
    )
    stats_type = prepared.result_type.fields[1]
    combined = builder.call(
        "tir.kernel", prepared.inputs, prepared.result_type, id="combined",
        attrs={
            "semantic_op": MatMulNormStats.op_name,
            "candidate": "tir.unit.matmul_norm_stats",
            "parameters": {"family": "unit", "variant": "matmul_norm_stats"},
            "facts": {},
            "semantic_attrs": prepared.attrs,
        },
    )
    value = builder.call(
        "builtin.get_item", (combined,), value_type, id="value", attrs={"index": 0})
    stats = builder.call(
        "builtin.get_item", (combined,), stats_type, id="stats", attrs={"index": 1})
    # The layer also consumes incoming statistics, so the same physical ABI is
    # both a formal parameter and a call result.
    incoming = builder.var("incoming_stats", stats_type, id="incoming_stats")
    builder.function("layer", (lhs, rhs, addend, incoming), (value, stats))

    main_lhs = builder.var("main_lhs", lhs_type, id="main_lhs")
    main_rhs = builder.var("main_rhs", rhs_type, id="main_rhs")
    main_addend = builder.var("main_addend", value_type, id="main_addend")
    main_stats = builder.var("main_stats", stats_type, id="main_stats")
    first = builder.call(
        "builtin.call", (main_lhs, main_rhs, main_addend, main_stats),
        prepared.result_type, id="layer_0_call", attrs={"callee": "layer"})
    first_value = builder.call(
        "builtin.get_item", (first,), value_type,
        id="layer_0_value", attrs={"index": 0})
    # The statistics result of layer 0 becomes the statistics parameter of
    # layer 1. This is the edge whose physical ABI used to disagree.
    first_stats = builder.call(
        "builtin.get_item", (first,), stats_type,
        id="layer_0_stats", attrs={"index": 1})
    second = builder.call(
        "builtin.call", (main_lhs, main_rhs, first_value, first_stats),
        prepared.result_type, id="layer_1_call", attrs={"callee": "layer"})
    builder.function(
        "main", (main_lhs, main_rhs, main_addend, main_stats), (second,))

    module = materialize_kernel_prim_functions(builder.build(entry="main"))
    return _declare_alignment(module, "result_0", value_alignment)


def _declare_alignment(module, name, alignment):
    """Declare a contract on one named kernel parameter."""
    definitions = tuple(
        replace(definition, parameters=tuple(
            replace(parameter, alignment_bytes=alignment)
            if parameter.name == name else parameter
            for parameter in definition.parameters
        ))
        for definition in module.kernel_definitions
    )
    return replace(module, kernel_definitions=definitions)


def test_a_strict_result_does_not_constrain_its_sibling_statistics():
    module = _module(value_alignment=16)
    requirements = transfer_source_alignment_requirements(module)

    # The declared 16-byte contract reaches the value leaf of the call result.
    assert requirements.get("layer_0_call.0") == 16
    # It must not reach the statistics leaf, which declared no contract.
    assert requirements.get("layer_0_call.1") in (None, 1)
    assert requirements.get("stats") in (None, 1)


def test_chained_layers_agree_on_the_statistics_physical_abi():
    plan = fm.make_buffer_plan(_module(value_alignment=16))
    formal = plan.buffer_map["incoming_stats"]
    actual = plan.buffer_map[
        dict(plan.function_map["main"].values)["layer_0_stats"][0]
    ]

    assert formal.distributed_storage_kind is actual.distributed_storage_kind
    assert formal.distributed_type == actual.distributed_type
    assert formal.dtype == actual.dtype
    assert formal.shape == actual.shape
    assert formal.strides == actual.strides
    # The padded owner stride is exactly what broke layer-to-layer stitching.
    assert formal.component_stride_bytes == actual.component_stride_bytes


def test_the_declared_value_contract_is_still_honoured():
    plan = fm.make_buffer_plan(_module(value_alignment=16))
    value = plan.buffer_map[
        dict(plan.function_map["main"].values)["layer_0_value"][0]
    ]

    assert value.alignment % 16 == 0
    assert value.component_stride_bytes % 16 == 0


@pytest.mark.parametrize("alignment", [1, 16, 64])
def test_every_call_argument_satisfies_the_physical_abi_verifier(alignment):
    # The planner raises IRVerificationError on a formal/actual ABI mismatch,
    # so a successful plan is the assertion for each declared contract.
    plan = fm.make_buffer_plan(_module(value_alignment=alignment))
    assert plan.call_map["layer_1_call"].callee == "layer"


def test_a_statistics_contract_still_propagates_when_it_is_declared():
    """The fix narrows who is constrained, not whether contracts propagate."""
    module = _declare_alignment(_module(value_alignment=16), "result_1", 64)

    requirements = transfer_source_alignment_requirements(module)
    assert requirements.get("layer_0_call.1") == 64
    # It reaches the callee formal bound to that same storage.
    assert requirements.get("incoming_stats") == 64

    plan = fm.make_buffer_plan(module)
    formal = plan.buffer_map["incoming_stats"]
    assert formal.component_stride_bytes % 64 == 0
