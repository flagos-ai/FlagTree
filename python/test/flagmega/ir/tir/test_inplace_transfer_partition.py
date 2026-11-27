# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-owner row snapshots are a typed protocol, not read-only exceptions."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.microkernels.materialization import validate_transfer_pipeline
from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.passes.tir.plan_storage_alignments import implementation_alignments
from triton.flagmega.codegen.triton.call_abi import _transfer_pipeline_abi
from triton.flagmega.codegen.triton.microkernels.materialization import materialize_shared_workspace_buffers


def partitioned_update():
    state_type = fm.RefType("state", (
        ("untouched", fm.tensor_type("float32", (16,))),
        ("recurrent", fm.tensor_type(fm.VectorType(fm.DType.FLOAT32, (4,)), (1, 2, 8, 4))),
    ))
    output_type = fm.DistributedType(fm.tensor_type("bfloat16", (1, 16)),
                                    (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
                                    fm.Placement((2, 4), "yx", "bb"))
    partition = fm.T.inplace_transfer_partition(source_field_path=("recurrent",), source_row_rank=3,
                                               output_index=0, output_axis=1, tile_rows=4)
    workspace = fm.T.shared_workspace_descriptor("state_stage", fm.tensor_type("float32", (2, 4, 16)), 16)
    pipeline = fm.T.transfer_pipeline_contract((fm.T.transfer_pipeline_channel(
        "state", (0,), (0,), 16, inplace_partition=partition),), capacity=2)
    selection = fm.T.microkernel_selection("test.state.pipeline", "state", "pipeline",
                                           shared_workspaces=(workspace,), transfer_pipeline=pipeline)
    dispatch = fm.T.kernel_dispatch("test.state.update", arguments=("state",), outputs=("output",),
        microkernel=selection, reads=("state",), writes=("state", "output"), memory_effects=(
            ("state", fm.MemoryEffect.for_fields(recurrent=fm.MemoryEffect.READ_WRITE)),
            ("output", fm.MemoryEffect.WRITE)))
    function = fm.T.prim_function("update", "triton", (
        fm.T.prim_parameter("state", state_type, fm.PrimParameterRole.INOUT, alignment_bytes=16),
        fm.T.prim_parameter("output", output_type, fm.PrimParameterRole.OUTPUT),
    ), fm.T.sequential((dispatch,)), fm.T.return_((
        fm.T.return_binding(fm.T.value_ref("output", output_type), "output"),)))
    return function, dispatch


def test_inplace_partition_is_verified_before_resource_selection():
    function, dispatch = partitioned_update()
    validate_transfer_pipeline(function, dispatch, dispatch.microkernel, stage="test")
    assert implementation_alignments(function, dispatch.microkernel) == {"state": 16}
    partition = dispatch.microkernel.transfer_pipeline.channels[0].inplace_partition
    assert fm.tir_from_data(partition.to_data()) == partition
    selection = dispatch.microkernel
    buffers = materialize_shared_workspace_buffers(function, selection)
    encoded = _transfer_pipeline_abi(dispatch, buffers)["transfer_pipeline"]["channels"][0]
    assert fm.tir_from_data(encoded["inplace_partition"]) == partition
    assert fm.tir_from_data(function.to_data()) == function


@pytest.mark.parametrize("violation", ("missing_phase", "unknown_field", "output_extent", "partial",
                                      "replicated", "replicated_mesh_axis", "tile_rows", "tile_columns", "dtype", "source_rank",
                                      "readonly_field", "missing_output", "multiple_sources"))
def test_unproved_inplace_partition_cannot_enable_a_pipeline(violation):
    function, dispatch = partitioned_update()
    selection = dispatch.microkernel
    pipeline = selection.transfer_pipeline
    channel, = pipeline.channels
    partition = channel.inplace_partition
    if violation in {"unknown_field", "source_rank", "missing_output"}:
        partition = replace(partition, **{
            "unknown_field": {"source_field_path": ("typo",)}, "source_rank": {"source_row_rank": 4},
            "missing_output": {"output_index": 1},
        }[violation])
        channel = replace(channel, inplace_partition=partition)
    elif violation == "missing_phase":
        channel = replace(channel, inplace_partition=None)
    elif violation in {"output_extent", "partial", "replicated", "replicated_mesh_axis"}:
        output = function.parameters[1]
        changed = replace(output.type, **{
            "output_extent": {"tensor": fm.tensor_type("bfloat16", (1, 32))},
            "partial": {"partial": fm.SBP.partial((0,), fm.ReduceOp.SUM)},
            "replicated": {"axis_policies": (fm.SBP.broadcast(),) * 2},
            "replicated_mesh_axis": {"axis_policies": (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,)))},
        }[violation])
        function = replace(function, parameters=(function.parameters[0], replace(output, type=changed)),
                           results=fm.T.return_((fm.T.return_binding(fm.T.value_ref("output", changed), "output"),)))
    elif violation in {"tile_rows", "tile_columns", "dtype"}:
        workspace, = selection.shared_workspaces
        tensor = fm.tensor_type("bfloat16" if violation == "dtype" else "float32",
                                (2, 8 if violation == "tile_rows" else 4, 8 if violation == "tile_columns" else 16))
        selection = replace(selection, shared_workspaces=(replace(workspace, type=tensor),))
    elif violation == "readonly_field":
        dispatch = replace(dispatch, reads=("state",), writes=("output",), memory_effects=(
            ("state", fm.MemoryEffect.for_fields(recurrent=fm.MemoryEffect.READ)), ("output", fm.MemoryEffect.WRITE)))
    else:
        with pytest.raises(IRSchemaError, match="[Ii]nplace"):
            replace(channel, source_argument_indices=(0, 1))
        return
    selection = replace(selection, transfer_pipeline=replace(pipeline, channels=(channel,)))
    dispatch = replace(dispatch, microkernel=selection)
    with pytest.raises(IRVerificationError, match="[Ii]nplace|non-read-only"):
        validate_transfer_pipeline(function, dispatch, selection, stage="test")


def test_nested_source_path_uses_abi_field_order_not_alphabetical_order():
    function, dispatch = partitioned_update()
    partition = dispatch.microkernel.transfer_pipeline.channels[0].inplace_partition
    scalar = fm.tensor_type("float32", ())
    nested = fm.RefType("outer", (("z", fm.TupleType((scalar, scalar))),
                                  ("a", function.parameters[0].type)))
    index, leaf = replace(partition, source_field_path=("a", "recurrent")).source_leaf(nested)
    assert index == 3
    assert leaf == function.parameters[0].type.fields[1][1]


def test_one_mutable_leaf_cannot_be_prefetched_by_independent_channels():
    function, dispatch = partitioned_update()
    selection = dispatch.microkernel
    channel, = selection.transfer_pipeline.channels
    duplicate = replace(channel, name="second", shared_workspace_indices=(1,))
    selection = replace(selection, shared_workspaces=(*selection.shared_workspaces,
                        replace(selection.shared_workspaces[0], name="second_stage")),
                        transfer_pipeline=replace(selection.transfer_pipeline, channels=(channel, duplicate)))
    with pytest.raises(IRVerificationError, match="overlapping inplace"):
        validate_transfer_pipeline(function, dispatch, selection, stage="test")


def test_readonly_channel_encoding_does_not_change_when_inplace_is_unused():
    channel = fm.T.transfer_pipeline_channel("source", (0,), (0,), 16)
    assert "inplace_partition" not in channel.to_data()
    assert fm.tir_from_data(channel.to_data()) == channel


@pytest.mark.parametrize("change", ({"source_field_path": "recurrent"}, {"source_field_path": (1,)},
                                   {"source_row_rank": 0}, {"output_axis": -1}, {"tile_rows": True}))
def test_partition_schema_is_typed(change):
    _, dispatch = partitioned_update()
    partition = dispatch.microkernel.transfer_pipeline.channels[0].inplace_partition
    with pytest.raises(IRSchemaError, match="Inplace"):
        replace(partition, **change)
