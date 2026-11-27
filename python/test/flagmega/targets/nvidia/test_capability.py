# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.targets import NvidiaSm90Target, Sm90Capability


def _block_fp8_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="distributed")
    activation = fm.tensor_type("bfloat16", (1, 128))
    weight = fm.tensor_type("float8_e4m3fn", (128, 128))
    scale = fm.tensor_type("float32", (1, 1))
    lhs = builder.var("lhs", activation, id="lhs")
    rhs = builder.weight("rhs", weight, source="weights", key="rhs", id="rhs")
    rhs_scale = builder.weight(
        "rhs_scale", scale, source="weights", key="rhs_scale", id="rhs_scale"
    )
    result = builder.call(
        "math.block_scaled_matmul",
        (lhs, rhs, rhs_scale),
        activation,
        id="result",
        attrs={"weight_block_n": 128, "weight_block_k": 128},
    )
    builder.function("main", (lhs,), (result,))
    return builder.build(entry="main")


def test_capability_schema_round_trips_and_derives_features():
    capability = Sm90Capability(
        supports_fp8_mma=False,
        supports_tma=False,
        supports_warp_specialize=False,
    )

    loaded = Sm90Capability.from_data(capability.to_data())

    assert loaded == capability
    assert loaded.features == frozenset({
        "fp8",
        "async_copy",
        "cooperative_grid",
        "grid_sync",
        "mma_v3",
    })
    assert loaded.missing(("tma", "fp8_mma")) == ("fp8_mma", "tma")


def test_capability_rejects_non_sm90_and_invalid_machine_limits():
    with pytest.raises(IRSchemaError, match="9.x"):
        Sm90Capability(compute_capability=(8, 0))
    with pytest.raises(IRSchemaError, match="cannot exceed"):
        Sm90Capability(max_threads_per_block=2048, max_threads_per_sm=1024)
    with pytest.raises(IRSchemaError, match="supports_tma must be bool"):
        Sm90Capability(supports_tma=1)


def test_tir_proposal_filters_unsupported_candidates_and_repairs_default():
    target = NvidiaSm90Target(Sm90Capability(
        supports_fp8_mma=False,
        supports_async_copy=False,
    ))

    proposed = target.propose_tir(_block_fp8_module())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert tuple(value.id for value in point.candidates) == ("tir.block_fp8.simt",)
    assert point.default_candidate == "tir.block_fp8.simt"
    assert proposed.selection_map[point.id].candidate_id == "tir.block_fp8.simt"
    target.verify(proposed)


def test_verifier_rejects_agent_inserted_unsupported_candidate():
    target = NvidiaSm90Target(Sm90Capability(supports_tma=False))
    candidate = fm.Candidate(
        "tir.unit.tma",
        {"family": "unit", "variant": "tma"},
        {"requires": ("tma",)},
    )
    point = fm.SelectionPoint(
        "tir.unit",
        "tir",
        (candidate,),
        candidate.id,
        owner="unit",
    )
    module = fm.IRModule(
        "high_level",
        "selected_tir_variants",
        (),
        (),
        "main",
        selection_points=(point,),
    )

    with pytest.raises(IRVerificationError, match="unavailable features.*tma"):
        target.verify(module)
