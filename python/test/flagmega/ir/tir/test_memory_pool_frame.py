# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.serialization import tir_from_data, tir_to_data


def test_prim_function_call_memory_pool_frames_round_trip_as_typed_tir():
    call = fm.T.prim_function_call(
        "invoke",
        "worker",
        memory_pools=(
            fm.T.memory_pool_frame("chip_data", "chip-frame", 256, 512),
            fm.T.memory_pool_frame("block_data", "block-frame", 64, 128),
        ),
    )

    encoded = tir_to_data(call)
    restored = tir_from_data(encoded)

    assert "workspace_offset" not in encoded
    assert "workspace_nbytes" not in encoded
    assert restored == call
    assert restored.memory_pool_map["block_data"].allocation == "block-frame"


def test_prim_function_call_rejects_duplicate_memory_space_frames():
    frame = fm.T.memory_pool_frame("data", "frame", 0, 64)
    with pytest.raises(IRSchemaError, match="duplicate memory-pool"):
        fm.T.prim_function_call(
            "invoke", "worker", memory_pools=(frame, frame)
        )


def test_legacy_prim_function_call_workspace_fields_upgrade_on_decode():
    restored = tir_from_data({
        "kind": "prim_function_call",
        "call_id": "invoke",
        "callee": "worker",
        "workspace_offset": 256,
        "workspace_nbytes": 128,
    })

    assert restored.memory_pools == (
        fm.T.memory_pool_frame("workspace", None, 256, 128),
    )
    assert "workspace_offset" not in tir_to_data(restored)
