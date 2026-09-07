# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


def _channel(name="rhs", sources=(1,), workspaces=(0,), alignment=16):
    return fm.T.transfer_pipeline_channel(
        name,
        source_argument_indices=sources,
        shared_workspace_indices=workspaces,
        source_alignment_bytes=alignment,
    )


def test_transfer_channel_indices_and_alignment_are_typed():
    channel = _channel(sources=(1, 2), workspaces=(0, 3), alignment=32)

    assert channel.source_argument_indices == (1, 2)
    assert channel.shared_workspace_indices == (0, 3)
    assert channel.source_alignment_bytes == 32

    with pytest.raises(IRSchemaError, match="non-empty, non-negative, and unique"):
        _channel(sources=(1, 1))
    with pytest.raises(IRSchemaError, match="non-empty, non-negative, and unique"):
        _channel(workspaces=())
    with pytest.raises(IRSchemaError, match="positive power of two"):
        _channel(alignment=24)


def test_pipeline_assigns_each_workspace_to_exactly_one_owner():
    contract = fm.T.transfer_pipeline_contract(
        (_channel("lhs", (0,), (0,)), _channel("rhs", (1,), (1,))),
        consumer_shared_workspace_indices=(2,),
        auxiliary_consumer=fm.T.auxiliary_consumer_contract(
            (1,), consumer_shared_workspace_indices=(2,)
        ),
        capacity=4,
    )

    assert contract.source_argument_indices == (0, 1)
    assert contract.shared_workspace_indices == (0, 1)
    assert contract.consumer_shared_workspace_indices == (2,)
    assert contract.capacity == 4

    with pytest.raises(IRSchemaError, match="owned by multiple transfer channels"):
        fm.T.transfer_pipeline_contract(
            (_channel("lhs", (0,), (0,)), _channel("rhs", (1,), (0,)))
        )
    with pytest.raises(IRSchemaError, match="both a transfer channel and the consumer"):
        fm.T.transfer_pipeline_contract(
            (_channel(),), consumer_shared_workspace_indices=(0,)
        )
    with pytest.raises(IRSchemaError, match="outside the transfer channel range"):
        fm.T.transfer_pipeline_contract(
            (_channel(),),
            auxiliary_consumer=fm.T.auxiliary_consumer_contract((1,)),
        )
    with pytest.raises(IRSchemaError, match="not owned by the transfer pipeline consumer"):
        fm.T.transfer_pipeline_contract(
            (_channel(),),
            consumer_shared_workspace_indices=(1,),
            auxiliary_consumer=fm.T.auxiliary_consumer_contract(
                (0,), consumer_shared_workspace_indices=(2,)
            ),
        )


@pytest.mark.parametrize("capacity", (0, -1, True, 2.5))
def test_pipeline_capacity_is_a_positive_integer(capacity):
    with pytest.raises(IRSchemaError, match="capacity must be a positive integer"):
        fm.T.transfer_pipeline_contract((_channel(),), capacity=capacity)


def test_pipeline_capacity_round_trips_and_old_ir_defaults_to_unspecified():
    contract = fm.T.transfer_pipeline_contract(
        (_channel(),), capacity=4
    )

    assert fm.tir_from_data(contract.to_data()) == contract
    legacy = contract.to_data()
    legacy.pop("capacity")
    assert fm.tir_from_data(legacy).capacity is None


def test_microkernel_pipeline_requires_complete_workspace_ownership():
    descriptors = (
        fm.T.shared_workspace_descriptor(
            "rhs_stage", fm.tensor_type("bfloat16", (2, 64)), 16
        ),
        fm.T.shared_workspace_descriptor(
            "accumulator", fm.tensor_type("float32", (64,)), 16
        ),
    )
    with pytest.raises(IRSchemaError, match="assign every shared workspace"):
        fm.T.microkernel_selection(
            implementation="test.pipeline",
            family="dense",
            variant="pipeline",
            shared_workspaces=descriptors,
            transfer_pipeline=fm.T.transfer_pipeline_contract((_channel(),)),
        )
