# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import (
    GatedDeltaNetStateConfig,
    GatedDeltaNetStateKind,
    create_gdn_state,
)


torch = pytest.importorskip("torch")


def _config() -> GatedDeltaNetStateConfig:
    return GatedDeltaNetStateConfig(
        num_layers=3,
        num_key_heads=1,
        num_value_heads=2,
        key_head_dim=4,
        value_head_dim=4,
        conv_kernel_size=3,
        hidden_size=8,
    )


def test_state_ir_type_matches_nncase_vector_lane_abi():
    config = _config()
    convolution = config.logical_tensor_type(GatedDeltaNetStateKind.CONVOLUTION)
    recurrent = config.logical_tensor_type(GatedDeltaNetStateKind.RECURRENT)

    assert convolution == fm.tensor_type(fm.vector_type("bfloat16", 8), [3, 2, 2])
    assert recurrent == fm.tensor_type(fm.vector_type("float32", 4), [3, 2, 4, 1])
    assert config.storage_shape(GatedDeltaNetStateKind.CONVOLUTION) == (3, 2, 2, 8)
    assert config.storage_shape(GatedDeltaNetStateKind.RECURRENT) == (3, 2, 4, 1, 4)


def test_convolution_state_pack_round_trip_and_lane_offsets():
    config = _config()
    state = create_gdn_state(config)
    logical = torch.arange(config.conv_dim * 2, dtype=torch.bfloat16).reshape(config.conv_dim, 2)

    state.update_convolution_layer(logical, layer_id=1)

    torch.testing.assert_close(state.convolution_layer(1), logical)
    for channel in range(config.conv_dim):
        for history in range(2):
            assert state.convolution[1, channel // 8, history, channel % 8] == logical[channel, history]
    assert torch.count_nonzero(state.convolution[0]) == 0
    assert torch.count_nonzero(state.convolution[2]) == 0


def test_recurrent_state_pack_round_trip_permuted_layout_and_lane_offsets():
    config = _config()
    state = create_gdn_state(config)
    logical = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)

    state.update_recurrent_layer(logical, layer_id=2)

    torch.testing.assert_close(state.recurrent_layer(2), logical)
    for head in range(2):
        for key in range(4):
            for value in range(4):
                assert state.recurrent[2, head, value, key // 4, key % 4] == logical[head, key, value]
    assert torch.count_nonzero(state.recurrent[:2]) == 0


def test_state_clone_preserves_config_but_not_mutable_storage_aliases():
    state = create_gdn_state(_config())
    cloned = state.clone()

    assert cloned.config is state.config
    assert cloned.convolution.data_ptr() != state.convolution.data_ptr()
    assert cloned.recurrent.data_ptr() != state.recurrent.data_ptr()
