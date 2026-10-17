# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Distributed handles remain static and agree with the device pointer ABI."""

import pytest
import triton.language as tl

from triton.experimental.tle.language.distributed import device_mesh
from triton.experimental.tle.language.gpu.core import _as_call_args
from triton.runtime import DistributedRtContext


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setattr(DistributedRtContext, "_instance", None)
    monkeypatch.setattr(DistributedRtContext, "_initialized", False)
    monkeypatch.setattr(DistributedRtContext, "_init_count", 0)
    return DistributedRtContext(11, 22)


def test_tle_pointer_indices_match_packed_device_handles(context):
    assert context.comm_ptr == 11
    assert context.mem_ptr == 22
    assert (context[0], context[1]) == context.get_packed_data() == (22, 11)
    assert context._get_needed_params() == {"device_comm_ptr": 11, "device_mem_ptr": 22}


def test_warp_specialize_keeps_a_nested_context_static(context):
    args = _as_call_args(tl.tuple([tl.constexpr(context)]))
    assert len(args) == 1
    assert isinstance(args[0], tl.constexpr)
    assert args[0].value is context


def test_context_can_be_recreated_after_communicator_cleanup(context):
    context._reset()
    assert context.comm_ptr is None
    assert context.mem_ptr is None
    recreated = DistributedRtContext(33, 44)
    assert recreated.get_packed_data() == (44, 33)
    assert (recreated[0], recreated[1]) == (44, 33)


def test_context_lookup_before_communicator_creation_does_not_freeze_empty_handles(context):
    context._reset()
    # Ordinary Triton launches query this singleton before FlagCX exists.
    empty = DistributedRtContext()
    assert empty.comm_ptr is None
    assert empty.mem_ptr is None
    bound = DistributedRtContext(33, 44)
    assert bound is empty
    assert bound.get_packed_data() == (44, 33)
    assert DistributedRtContext().get_packed_data() == (44, 33)
    assert bound._init_count == 1


def test_per_rank_launch_grid_excludes_device_rank():
    mesh = device_mesh({"device": 4, "block": [("block_y", 8), ("block_x", 16)]})
    assert mesh.shape == (4, 8, 16)
    assert mesh.launch_shape == (8, 16)


def test_runtime_context_tuple_preserves_handle_order_without_constants():
    import triton.language as tl
    from triton.experimental.tle.language.distributed import _parse_src_arg

    memory, communication = object(), object()
    context = tl.tuple((tl.tensor(memory, tl.int64), tl.tensor(communication, tl.uint64)))
    # No builder is needed: the handles are existing scalar IR operands.
    assert _parse_src_arg(None, context, 0) is memory
    assert _parse_src_arg(None, context, 1) is communication


def test_runtime_context_tuple_rejects_wrong_arity_or_tensor_handles():
    import triton.language as tl
    from triton.experimental.tle.language.distributed import _parse_src_arg

    scalar = tl.tensor(object(), tl.int64)
    with pytest.raises(ValueError, match='two handles'):
        _parse_src_arg(None, tl.tuple((scalar,)))
    vector = tl.tensor(object(), tl.block_type(tl.int64, [2]))
    with pytest.raises(TypeError, match='scalar i64 handles'):
        _parse_src_arg(None, tl.tuple((vector, scalar)))
