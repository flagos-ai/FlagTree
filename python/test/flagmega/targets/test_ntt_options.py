# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.targets import (
    NttTargetOptions,
    NvidiaSm90Target,
    PyNttTargetOptions,
)


def _options() -> NttTargetOptions:
    return NttTargetOptions(
        placements=(fm.Placement((2, 4), "xy", "db"),),
        vector_lane_bytes=32,
        vector_max_axes=1,
        packing_vector_bytes=32,
        packing_k_pack=4,
    )


def _add_module() -> fm.IRModule:
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value_type = fm.tensor_type("bfloat16", (1, 16))
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            output = fm.F.math.add(lhs, rhs, name="output")
            self.function("main", (lhs, rhs), (output,))

    return Graph().build()


def _dense_module() -> fm.IRModule:
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            lhs = self.input(
                "lhs", fm.tensor_type("bfloat16", (1, 128)), id="lhs"
            )
            weight = self.weight(
                "weight",
                fm.tensor_type("bfloat16", (128, 128)),
                source="memory",
                key="weight",
                id="weight",
            )
            output = fm.F.math.matmul(
                lhs, weight, transpose_b=True, name="output"
            )
            self.function("main", (lhs,), (output,))

    return Graph().build()


def test_ntt_options_round_trip_and_configure_backend_rules():
    options = _options()
    loaded = NttTargetOptions.from_data(options.to_data())
    target = NvidiaSm90Target(options=loaded)

    assert loaded == options
    assert target.distributed_placements(_add_module()) == options.placements
    assert target.vectorization_policy.lane_bytes == 32
    assert target.vectorization_policy.max_axes == 1
    assert target.packing_policy.vector_bytes == 32
    assert target.packing_policy.k_pack == 4
    assert target.distribution_policy.identity.startswith(
        "ntt-auto-distributed/v8("
    )
    assert "block_bytes=128" in target.distribution_policy.identity


def test_pyntt_split_policy_is_editable_and_round_trips_with_target_options():
    options = PyNttTargetOptions.from_ntt_options(
        _options(), block_cyclic_block_bytes=256
    )

    loaded = NttTargetOptions.from_data(options.to_data())
    target = NvidiaSm90Target(options=loaded)

    assert loaded == options
    assert isinstance(loaded, PyNttTargetOptions)
    assert target.options.block_cyclic_block_bytes == 256
    assert "block_bytes=256" in target.distribution_policy.identity


def test_nondefault_target_options_change_generic_dense_packing_geometry():
    target = NvidiaSm90Target(options=_options())

    proposed = target.propose_packing(_dense_module())
    point = next(value for value in proposed.selection_points if value.id == "packing.output")
    packed = next(
        value for value in point.candidates
        if value.id == "packing.k_major_n16_k64"
    )

    assert packed.parameters["n_lane"] == 16
    assert packed.parameters["k_lane"] == 64
    assert packed.parameters["payload_groups"] == 4
    assert packed.parameters["payload_width"] == 256
    assert {value.id for value in point.candidates} == {
        "packing.logical",
        "packing.k_major_n16_k64",
    }


def test_ntt_options_reject_invalid_rule_parameters():
    with pytest.raises(IRSchemaError, match="vector_lane_bytes"):
        replace(_options(), vector_lane_bytes=0)
    with pytest.raises(IRSchemaError, match="at least one placement"):
        replace(_options(), placements=())
    with pytest.raises(IRSchemaError, match="positive power of two"):
        PyNttTargetOptions.from_ntt_options(
            _options(), block_cyclic_block_bytes=96
        )


def test_selected_tir_records_options_and_rejects_different_active_options():
    selected = Compiler().compile(_add_module(), stop_after="lower-tir").module
    snapshot = NttTargetOptions.from_data(selected.metadata["target_options"])

    assert snapshot == NvidiaSm90Target().options
    with pytest.raises(IRVerificationError, match="target-options snapshot differs"):
        NvidiaSm90Target(options=_options()).verify(selected)
